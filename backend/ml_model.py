"""Trainable gesture classifier — optimized for 1–2 beat (multi-part) signs."""
from __future__ import annotations

import json
import threading

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, FEAT_DIM, MIN_SAMPLES, SEQ_LEN
from .recognition import resample, trim_idle, frame_distance

MODEL_PATH = DATA_DIR / 'gesture_model.joblib'
META_PATH = DATA_DIR / 'gesture_model_meta.json'

TIP = [4, 8, 12, 16, 20]
_lock = threading.Lock()
_bundle: dict | None = None


def _channel_indices() -> list[int]:
    idx: list[int] = []
    for hand in (0, 1):
        base = hand * 76
        idx.append(base)
        idx.extend([base + 64, base + 65])
        for tip in TIP:
            o = base + 1 + tip * 3
            idx.extend([o, o + 1, o + 2])
        idx.extend(range(base + 66, base + 76))
    seen = set()
    out = []
    for i in idx:
        if i not in seen and 0 <= i < FEAT_DIM:
            seen.add(i)
            out.append(i)
    return out


CHANNELS = _channel_indices()


def motion_profile(frames) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    motion = np.zeros(arr.shape[0], dtype=np.float64)
    for i in range(1, arr.shape[0]):
        motion[i] = frame_distance(arr[i], arr[i - 1])
    return motion


def count_motion_peaks(motion: np.ndarray, thresh: float = 0.04) -> int:
    if motion.size < 3:
        return 0
    peaks = 0
    active = False
    for v in motion:
        if v >= thresh and not active:
            peaks += 1
            active = True
        elif v < thresh * 0.5:
            active = False
    return peaks


def analyze_gesture(frames) -> dict:
    """Peak / energy stats used by training QA and idle rejection.

    Accepts both moving signs and static held poses (handshape only).
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return {'peaks': 0, 'mean': 0.0, 'max': 0.0, 'ok': False, 'split': 0, 'static': False}
    present = arr[(arr[:, 0] > 0) | (arr[:, 76] > 0)] if arr.shape[1] > 76 else arr
    presence = float(len(present) / max(1, arr.shape[0]))
    pose_spread = 0.0
    if present.shape[0] >= 1:
        mid = present[present.shape[0] // 2]
        # Spatial richness of the held handshape (zeros = no hand)
        pose_spread = float(np.std(mid[1:64])) if mid.shape[0] > 64 else float(np.std(mid))

    trimmed = trim_idle(arr)
    motion = motion_profile(trimmed)
    peaks = int(count_motion_peaks(motion))
    mean = float(motion.mean()) if motion.size else 0.0
    mx = float(motion.max()) if motion.size else 0.0
    split = _best_split(motion)
    motion_ok = mean >= 0.018 and mx >= 0.048 and peaks >= 1 and trimmed.shape[0] >= 5
    # Static OK: hand clearly visible long enough with a real handshape
    static_ok = (
        present.shape[0] >= 8
        and presence >= 0.45
        and pose_spread >= 0.12
    )
    ok = motion_ok or static_ok
    return {
        'peaks': peaks,
        'mean': mean,
        'max': mx,
        'ok': ok,
        'static': bool(static_ok and not motion_ok),
        'presence': round(presence, 3),
        'poseSpread': round(pose_spread, 4),
        'split': int(split),
        'frames': int(trimmed.shape[0]),
    }


def _best_split(motion: np.ndarray) -> int:
    """Valley between two motion beats (part A | part B)."""
    n = int(motion.size)
    if n < 8:
        return n // 2
    lo = max(2, n // 5)
    hi = min(n - 2, (4 * n) // 5)
    if hi <= lo:
        return n // 2
    region = motion[lo:hi]
    # Prefer a quiet valley in the middle third of the clip
    return int(lo + int(np.argmin(region)))


def _part_embed(part: np.ndarray, half_len: int) -> np.ndarray:
    seq = resample(part if part.shape[0] >= 2 else np.zeros((2, FEAT_DIM)), half_len)
    sub = seq[:, CHANNELS]
    vel = np.diff(sub, axis=0, prepend=sub[:1])
    picks = [0, half_len // 2, half_len - 1]
    return np.concatenate([
        sub[picks].reshape(-1),
        sub.mean(axis=0),
        sub.std(axis=0),
        vel.mean(axis=0),
        vel.std(axis=0),
    ]).astype(np.float64)


def embed_sequence(frames) -> np.ndarray:
    """
    Embedding tuned for two-beat signs:
    - explicit part-A / part-B descriptors split at motion valley
    - full-sequence keyframes + peak/duration stats
    """
    raw = np.asarray(frames, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raw = np.zeros((SEQ_LEN, FEAT_DIM))
    trimmed = trim_idle(raw)
    if trimmed.shape[0] < 2:
        trimmed = raw

    motion = motion_profile(trimmed)
    split = _best_split(motion)
    half = max(8, SEQ_LEN // 2)
    part_a = _part_embed(trimmed[: max(2, split + 1)], half)
    part_b = _part_embed(trimmed[min(trimmed.shape[0] - 2, split):], half)

    seq = resample(trimmed, SEQ_LEN)
    sub = seq[:, CHANNELS]
    vel = np.diff(sub, axis=0, prepend=sub[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    picks = [0, SEQ_LEN // 4, SEQ_LEN // 2, (3 * SEQ_LEN) // 4, SEQ_LEN - 1]
    keyframes = sub[picks].reshape(-1)

    # Four temporal quarters for finer multi-part shape
    quarters = []
    q = SEQ_LEN // 4
    for i in range(4):
        chunk = sub[i * q:(i + 1) * q]
        quarters.append(chunk.mean(axis=0))
        quarters.append(chunk.std(axis=0))
    quarters_vec = np.concatenate(quarters)

    peaks = float(count_motion_peaks(motion))
    motion_energy = np.linalg.norm(vel, axis=1)
    duration_proxy = float(min(1.0, trimmed.shape[0] / 72.0))
    split_ratio = float(split / max(1, trimmed.shape[0] - 1))

    vec = np.concatenate([
        part_a,
        part_b,
        keyframes,
        quarters_vec,
        sub.mean(axis=0),
        sub.std(axis=0),
        vel.mean(axis=0),
        vel.std(axis=0),
        acc.mean(axis=0),
        np.array([
            motion_energy.mean(),
            motion_energy.std(),
            motion_energy.max(),
            float(np.argmax(motion_energy)) / max(1, SEQ_LEN - 1),
            peaks / 4.0,
            duration_proxy,
            split_ratio,
            1.0 if peaks >= 2 else 0.0,
        ]),
    ]).astype(np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _dataset(signs: list[dict]):
    X, y = [], []
    labels = []
    for s in signs:
        samples = s.get('samples') or []
        if len(samples) < MIN_SAMPLES:
            continue
        labels.append({'id': int(s['id']), 'name': s['name']})
        for sample in samples:
            arr = np.asarray(sample, dtype=np.float64)
            X.append(embed_sequence(arr))
            y.append(int(s['id']))
            if arr.shape[0] >= 10:
                # Time jitter — keeps two-beat structure
                X.append(embed_sequence(arr[1:]))
                y.append(int(s['id']))
                X.append(embed_sequence(arr[:-1]))
                y.append(int(s['id']))
                # Mild stretch: drop every other frame once
                X.append(embed_sequence(arr[::2] if arr.shape[0] >= 16 else arr))
                y.append(int(s['id']))
    if not X:
        return None, None, []
    return np.vstack(X), np.asarray(y, dtype=np.int64), labels


def train_from_signs(signs: list[dict]) -> dict:
    global _bundle
    X, y, labels = _dataset(signs)
    if X is None or len(set(y.tolist())) < 1:
        with _lock:
            _bundle = None
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            meta = {'ready': False, 'classes': 0, 'samples': 0, 'engine': 'extratrees'}
            META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        return meta

    n_classes = len(set(y.tolist()))
    counts = np.bincount(y)
    positive = counts[counts > 0]
    min_per_class = int(positive.min()) if len(positive) else 0

    base = ExtraTreesClassifier(
        n_estimators=420,
        max_depth=24,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )

    if min_per_class >= 2 and len(y) >= max(6, n_classes * 2):
        cv = min(3, min_per_class)
        clf: object = CalibratedClassifierCV(base, method='sigmoid', cv=cv)
    else:
        clf = base

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])
    pipe.fit(X, y)

    meta = {
        'ready': True,
        'classes': n_classes,
        'samples': int(len(y)),
        'labels': labels,
        'dim': int(X.shape[1]),
        'engine': 'extratrees+joints',
        'seqLen': SEQ_LEN,
    }
    with _lock:
        joblib.dump({'model': pipe, 'meta': meta}, MODEL_PATH)
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        _bundle = {'model': pipe, 'meta': meta}
    return meta


def load_model() -> dict | None:
    global _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        if not MODEL_PATH.exists():
            return None
        try:
            _bundle = joblib.load(MODEL_PATH)
            return _bundle
        except Exception:
            _bundle = None
            return None


def status() -> dict:
    bundle = load_model()
    if bundle and bundle.get('meta'):
        return bundle['meta']
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'ready': False, 'classes': 0, 'samples': 0, 'engine': 'extratrees'}


def _classes_of(model) -> list[int]:
    if hasattr(model, 'classes_'):
        return [int(c) for c in model.classes_]
    clf = model.named_steps.get('clf') if hasattr(model, 'named_steps') else None
    if clf is not None and hasattr(clf, 'classes_'):
        return [int(c) for c in clf.classes_]
    return []


def is_intentional_gesture(frames, min_mean: float = 0.024, min_peaks: int = 1, min_max: float = 0.06) -> dict:
    info = analyze_gesture(frames)
    # Static held poses already marked ok by analyze_gesture — accept them
    if info.get('ok') and info.get('static'):
        return {**info, 'ok': True, 'reason': 'static'}
    ok = (
        info['ok']
        and info['mean'] >= min_mean
        and info['max'] >= min_max
        and info['peaks'] >= min_peaks
    )
    return {**info, 'ok': ok, 'reason': 'ok' if ok else 'idle'}


def _score_window(model, frames, ready: dict, classes: list[int]) -> list[dict]:
    emb = embed_sequence(frames).reshape(1, -1)
    try:
        proba = model.predict_proba(emb)[0]
    except Exception:
        return []
    ranked = []
    for i, cid in enumerate(classes):
        if cid not in ready:
            continue
        ranked.append({
            'id': cid,
            'name': ready[cid]['name'],
            'proba': float(proba[i]),
            'similarity': int(round(float(proba[i]) * 100)),
            'n': len(ready[cid]['samples']),
        })
    ranked.sort(key=lambda x: x['proba'], reverse=True)
    return ranked


def predict(
    frames,
    signs: list[dict],
    min_proba: float = 0.68,
    expected_id: int | None = None,
) -> dict | None:
    """Single-pass ExtraTrees — optimized for instant live recognition."""
    bundle = load_model()
    if not bundle or not bundle.get('model'):
        return None

    model = bundle['model']
    ready = {int(s['id']): s for s in signs if len(s.get('samples') or []) >= MIN_SAMPLES}
    if not ready:
        return None

    classes = _classes_of(model)
    if not classes or not hasattr(model, 'predict_proba'):
        return None

    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 5:
        return None

    gate = is_intentional_gesture(arr)
    if not gate['ok']:
        return {
            'match': None,
            'peek': None,
            'ranked': [],
            'engine': 'ml',
            'reason': 'idle',
            'motion': gate,
        }

    ranked = _score_window(model, arr, ready, classes)
    if not ranked:
        return None

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = best['proba'] - (second['proba'] if second else 0.0)
    peek = {
        'id': best['id'],
        'name': best['name'],
        'similarity': best['similarity'],
        'd': float(1.0 - best['proba']),
        'n': best['n'],
    }

    n_classes = len(ready)
    peaks = int(gate.get('peaks') or 0)
    if n_classes == 1:
        need = 0.78 if peaks >= 2 else 0.84
        need_margin = 0.0
    elif n_classes <= 3:
        need = 0.66 if peaks >= 2 else max(min_proba - 0.02, 0.70)
        need_margin = 0.10 if peaks >= 2 else 0.14
    else:
        need = 0.62 if peaks >= 2 else max(min_proba - 0.04, 0.66)
        need_margin = 0.08 if peaks >= 2 else 0.12

    # Verify mode: accept the expected word with a softer bar (user just trained it)
    if expected_id is not None and int(expected_id) in ready:
        for row in ranked:
            if int(row['id']) != int(expected_id):
                continue
            soft_need = max(0.48, need - 0.18)
            if row['proba'] >= soft_need:
                return {
                    'match': {
                        'id': row['id'],
                        'name': row['name'],
                        'similarity': max(int(row['similarity']), 70),
                        'd': float(1.0 - row['proba']),
                        'n': row['n'],
                        'margin': round(float(row['proba'] - (second['proba'] if second else 0.0)), 4),
                        'engine': 'ml-verify',
                        'window': 'full',
                        'peaks': peaks,
                    },
                    'peek': peek,
                    'ranked': ranked[:3],
                    'engine': 'ml-verify',
                    'motion': gate,
                }
            break

    if best['proba'] >= need and margin >= need_margin:
        return {
            'match': {
                'id': best['id'],
                'name': best['name'],
                'similarity': max(int(best['similarity']), 70),
                'd': float(1.0 - best['proba']),
                'n': best['n'],
                'margin': round(float(margin), 4),
                'engine': 'ml',
                'window': 'full',
                'peaks': peaks,
            },
            'peek': peek,
            'ranked': ranked[:3],
            'engine': 'ml',
            'motion': gate,
        }

    return {
        'match': None,
        'peek': peek,
        'ranked': ranked[:3],
        'engine': 'ml',
        'reason': 'low_conf',
        'motion': gate,
    }
