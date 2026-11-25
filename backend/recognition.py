"""High-precision sequence matching (pose + velocity + acceleration DTW)."""
from __future__ import annotations

import numpy as np

from .config import FEAT_DIM, MIN_SAMPLES, SEQ_LEN

TIP = [4, 8, 12, 16, 20]


def _weights() -> np.ndarray:
    w = np.ones(FEAT_DIM, dtype=np.float64)
    for hand in (0, 1):
        base = hand * 76
        w[base] = 0.35
        w[base + 64] = 1.6
        w[base + 65] = 1.6
        for tip in TIP:
            o = base + 1 + tip * 3
            w[o] = 2.4
            w[o + 1] = 2.4
            w[o + 2] = 1.5
        w[base + 66 : base + 76] = 1.9
    return w


WEIGHTS = _weights()


def resample(frames: list | np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((n, FEAT_DIM), dtype=np.float64)
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    idx = np.linspace(0, arr.shape[0] - 1, n)
    out = np.empty((n, arr.shape[1]), dtype=np.float64)
    for i, t in enumerate(idx):
        a = int(np.floor(t))
        b = min(arr.shape[0] - 1, a + 1)
        u = t - a
        out[i] = arr[a] * (1 - u) + arr[b] * u
    return out


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.sum(WEIGHTS * d * d) / np.sum(WEIGHTS)))


def trim_idle(frames: np.ndarray, thresh: float = 0.035) -> np.ndarray:
    if frames.shape[0] < 5:
        return frames
    motion = np.zeros(frames.shape[0])
    for i in range(1, frames.shape[0]):
        motion[i] = frame_distance(frames[i], frames[i - 1])
    # Static held poses (e.g. مرحبًا) have almost no temporal motion — keep full clip
    if float(motion.mean()) < thresh * 0.55 and float(motion.max()) < thresh * 1.2:
        return frames
    start, end = 0, frames.shape[0] - 1
    while start < end - 2 and motion[start] < thresh:
        start += 1
    while end > start + 2 and motion[end] < thresh:
        end -= 1
    start = max(0, start - 1)
    end = min(frames.shape[0] - 1, end + 1)
    cut = frames[start : end + 1]
    return cut if cut.shape[0] >= 4 else frames


def velocities(seq: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seq)
    out[1:] = seq[1:] - seq[:-1]
    return out


def distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    band = max(2, int(np.ceil(max(n, m) * 0.4)))
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = frame_distance(a[i - 1], b[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / max(n, m))


def motion_distance(a: np.ndarray, b: np.ndarray) -> float:
    pose = distance(a, b)
    vel = distance(velocities(a), velocities(b))
    acc = distance(velocities(velocities(a)), velocities(velocities(b)))
    return pose * 0.55 + vel * 0.30 + acc * 0.15


def build_windows(raw_frames: list) -> list[np.ndarray]:
    """Multi-scale windows — include full multi-part clip, not only a short tail."""
    if not raw_frames:
        return []
    arr = np.asarray(raw_frames, dtype=np.float64)
    out: list[np.ndarray] = []
    n = arr.shape[0]
    spans = [n, max(16, (2 * n) // 3), max(12, n // 2), 16]
    seen = set()
    for span in spans:
        if n < 5:
            continue
        slice_ = arr[-span:] if span < n else arr
        key = (slice_.shape[0], int(slice_[0, 0] * 1000) if slice_.size else 0)
        if key in seen or slice_.shape[0] < 5:
            continue
        seen.add(key)
        out.append(resample(trim_idle(slice_)))
    if not out:
        out.append(resample(arr))
    return out[:5]


def sign_score(windows: list[np.ndarray], samples: list) -> float:
    best = float('inf')
    for sample in samples:
        s = np.asarray(sample, dtype=np.float64)
        variants = [s, resample(trim_idle(s))]
        for variant in variants:
            for w in windows:
                d = motion_distance(w, variant)
                if d < best:
                    best = d
    return best


def classify(raw_frames: list, signs: list, threshold: float = 0.18) -> dict | None:
    ready = [s for s in signs if len(s.get('samples') or []) >= MIN_SAMPLES]
    if not ready or not raw_frames:
        return None

    arr = np.asarray(raw_frames, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 5:
        motion = np.zeros(arr.shape[0])
        for i in range(1, arr.shape[0]):
            motion[i] = frame_distance(arr[i], arr[i - 1])
        if float(motion.mean()) < 0.025 or float(motion.max()) < 0.06:
            return {'match': None, 'peek': None, 'ranked': [], 'reason': 'idle'}

    windows = build_windows(raw_frames)
    ranked = []
    for s in ready:
        d = sign_score(windows, s['samples'])
        sim = int(max(0, min(99, round((1 - d / 0.55) * 100))))
        ranked.append({'id': s['id'], 'name': s['name'], 'd': d, 'n': len(s['samples']), 'similarity': sim})
    ranked.sort(key=lambda x: x['d'])

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = (second['d'] - best['d']) if second else float('inf')

    ready_count = len(ready)
    min_margin = 0.02
    if ready_count == 1:
        min_margin = 0.0
        threshold = min(threshold, 0.14)  # single class: very strict distance
    elif ready_count == 2:
        min_margin = 0.025
    elif ready_count <= 8:
        min_margin = 0.03
    else:
        min_margin = 0.035

    max_dist = threshold
    if best['similarity'] < 78:
        return {'match': None, 'peek': best, 'ranked': ranked[:3], 'reason': 'low_sim'}

    if best['d'] > max_dist or margin < min_margin:
        return {
            'match': None,
            'peek': best,
            'ranked': ranked[:3],
        }

    return {
        'match': {**best, 'margin': round(margin if margin != float('inf') else 1.0, 4)},
        'peek': best,
        'ranked': ranked[:3],
    }


def quality_score(frames: list) -> int:
    if not frames:
        return 0
    arr = np.asarray(frames, dtype=np.float64)
    present = arr[(arr[:, 0] > 0) | (arr[:, 76] > 0)]
    ratio = len(present) / max(1, len(arr))
    motion = 0.0
    for i in range(1, len(present)):
        motion += frame_distance(present[i], present[i - 1])
    motion /= max(1, len(present) - 1)
    return int(min(100, round(ratio * 75 + min(25, motion * 35))))
