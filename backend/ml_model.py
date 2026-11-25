"""Trainable gesture classifier — multi-part sequence aware."""
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
    """Per-frame motion energy — captures multi-part peaks."""
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    motion = np.zeros(arr.shape[0], dtype=np.float64)
    for i in range(1, arr.shape[0]):
        motion[i] = frame_distance(arr[i], arr[i - 1])
    return motion


def count_motion_peaks(motion: np.ndarray, thresh: float = 0.045) -> int:
    if motion.size < 3:
        return 0
    peaks = 0
    active = False
    for v in motion:
        if v >= thresh and not active:
            peaks += 1
            active = True
        elif v < thresh * 0.55:
            active = False
    return peaks


def embed_sequence(frames) -> np.ndarray:
    """Temporal embedding: whole clip + 3 segments + motion peaks (multi-part)."""
    raw = np.asarray(frames, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raw = np.zeros((SEQ_LEN, FEAT_DIM))
    trimmed = trim_idle(raw)
    seq = resample(trimmed, SEQ_LEN)
    sub = seq[:, CHANNELS]
    vel = np.diff(sub, axis=0, prepend=sub[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])

    # 5 keyframes across the full multi-part sequence
    picks = [0, SEQ_LEN // 4, SEQ_LEN // 2, (3 * SEQ_LEN) // 4, SEQ_LEN - 1]
    keyframes = sub[picks].reshape(-1)

    # Three temporal thirds — distinguishes prefix-only vs full gesture
    thirds = []
    for t0, t1 in ((0, SEQ_LEN // 3), (SEQ_LEN // 3, 2 * SEQ_LEN // 3), (2 * SEQ_LEN // 3, SEQ_LEN)):
        part = sub[t0:t1]
        thirds.append(part.mean(axis=0))
        thirds.append(part.std(axis=0))
    thirds_vec = np.concatenate(thirds)

    motion = motion_profile(trimmed if trimmed.shape[0] >= 4 else seq)
    motion_energy = np.linalg.norm(vel, axis=1)
    peaks = float(count_motion_peaks(motion))
    duration_proxy = float(min(1.0, trimmed.shape[0] / 60.0))  # ~3s at 20fps → 1.0

    vec = np.concatenate([
        keyframes,
        thirds_vec,
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
            peaks / 5.0,
            duration_proxy,
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
            # Light augmentation: slight time jitter via trim variants
            if arr.shape[0] >= 8:
                X.append(embed_sequence(arr[1:]))
                y.append(int(s['id']))
                X.append(embed_sequence(arr[:-1]))
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
        n_estimators=360,
        max_depth=22,
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
        'engine': 'extratrees+multipart',
        'seqLen': SEQ_LEN,
    }
    with _lock:
        # Dimension changed — drop old incompatible model file first
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


def predict(frames, signs: list[dict], min_proba: float = 0.42) -> dict | None:
    """
    Multi-window predict: prefers the FULL gesture window so short prefixes
    (e.g. first move of «كيف حالك») don't steal the match from a shorter word.
    """
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
    if arr.ndim != 2 or arr.shape[0] < 4:
        return None

    n = arr.shape[0]
    # Build candidate windows: full multi-part first, then shorter tails
    windows: list[tuple[str, float, np.ndarray]] = [
        ('full', 1.20, arr),
    ]
    if n >= 16:
        windows.append(('body', 1.05, arr[max(0, n // 8):]))
    if n >= 24:
        windows.append(('tail', 0.82, arr[-max(10, n // 3):]))

    # Aggregate weighted votes across windows
    scores: dict[int, float] = {}
    best_by_id: dict[int, dict] = {}
    for kind, weight, win in windows:
        ranked = _score_window(model, win, ready, classes)
        for row in ranked[:4]:
            cid = row['id']
            val = row['proba'] * weight
            if val > scores.get(cid, -1):
                scores[cid] = val
                best_by_id[cid] = {**row, 'window': kind, 'weighted': val}

    if not scores:
        return None

    ordered = sorted(best_by_id.values(), key=lambda x: x['weighted'], reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    margin = best['weighted'] - (second['weighted'] if second else 0.0)

    # Raw full-window ranking for peek UI
    full_ranked = _score_window(model, arr, ready, classes) or ordered
    peek_src = full_ranked[0]
    peek = {
        'id': peek_src['id'],
        'name': peek_src['name'],
        'similarity': peek_src['similarity'],
        'd': float(1.0 - peek_src['proba']),
        'n': peek_src['n'],
    }

    n_classes = len(ready)
    need = 0.26 if n_classes == 1 else min_proba
    need_margin = 0.0 if n_classes == 1 else (0.08 if n_classes <= 3 else 0.07)

    # Reject short-tail-only wins when full window disagrees (anti prefix-confusion)
    if best.get('window') == 'tail' and full_ranked:
        full_best = full_ranked[0]
        if full_best['id'] != best['id'] and full_best['proba'] >= 0.28:
            best = {
                **full_best,
                'window': 'full',
                'weighted': full_best['proba'] * 1.20,
            }
            second = full_ranked[1] if len(full_ranked) > 1 else None
            margin = best['proba'] - (second['proba'] if second else 0.0)

    if best['proba'] >= need and margin >= need_margin:
        return {
            'match': {
                'id': best['id'],
                'name': best['name'],
                'similarity': max(int(best['similarity']), 55),
                'd': float(1.0 - best['proba']),
                'n': best['n'],
                'margin': round(float(margin), 4),
                'engine': 'ml',
                'window': best.get('window', 'full'),
            },
            'peek': peek,
            'ranked': full_ranked[:3],
            'engine': 'ml',
        }

    return {'match': None, 'peek': peek, 'ranked': full_ranked[:3], 'engine': 'ml'}
