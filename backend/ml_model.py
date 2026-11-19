"""Trainable gesture classifier — real ML model you teach with your words."""
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
from .recognition import resample, trim_idle

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


def embed_sequence(frames) -> np.ndarray:
    seq = np.asarray(frames, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] == 0:
        seq = np.zeros((SEQ_LEN, FEAT_DIM))
    seq = resample(trim_idle(seq), SEQ_LEN)
    sub = seq[:, CHANNELS]
    vel = np.diff(sub, axis=0, prepend=sub[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    picks = [0, SEQ_LEN // 4, SEQ_LEN // 2, (3 * SEQ_LEN) // 4, SEQ_LEN - 1]
    keyframes = sub[picks].reshape(-1)
    motion_energy = np.linalg.norm(vel, axis=1)
    vec = np.concatenate([
        keyframes,
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
            X.append(embed_sequence(sample))
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
        n_estimators=320,
        max_depth=20,
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
        'engine': 'extratrees+scaler',
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
        _bundle = joblib.load(MODEL_PATH)
        return _bundle


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


def predict(frames, signs: list[dict], min_proba: float = 0.48) -> dict | None:
    bundle = load_model()
    if not bundle or not bundle.get('model'):
        return None

    model = bundle['model']
    ready = {int(s['id']): s for s in signs if len(s.get('samples') or []) >= MIN_SAMPLES}
    if not ready:
        return None

    # One embedding on the trimmed full clip — much faster than multi-window
    classes = _classes_of(model)
    if not classes or not hasattr(model, 'predict_proba'):
        return None

    emb = embed_sequence(resample(trim_idle(np.asarray(frames, dtype=np.float64)))).reshape(1, -1)
    proba = model.predict_proba(emb)[0]

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

    # One class: sensitive. Multiple classes: require clearer margin for accuracy.
    n_classes = len(ranked)
    need = 0.28 if n_classes == 1 else min_proba
    need_margin = 0.0 if n_classes == 1 else (0.10 if n_classes <= 3 else 0.08)
    if best['proba'] >= need and margin >= need_margin:
        return {
            'match': {
                'id': best['id'],
                'name': best['name'],
                'similarity': max(best['similarity'], 55),
                'd': float(1.0 - best['proba']),
                'n': best['n'],
                'margin': round(margin, 4),
                'engine': 'ml',
            },
            'peek': peek,
            'ranked': ranked[:3],
            'engine': 'ml',
        }

    return {'match': None, 'peek': peek, 'ranked': ranked[:3], 'engine': 'ml'}
