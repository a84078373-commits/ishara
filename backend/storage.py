"""Persistent sign library on disk."""
from __future__ import annotations

import json
import threading

from .config import (
    DATA_VERSION,
    DEFAULT_WORDS,
    FEAT_DIM,
    MAX_SAMPLES,
    MAX_SIGNS,
    MIN_SAMPLES,
    SEQ_LEN,
    SIGNS_PATH,
)
from .recognition import quality_score, resample

_lock = threading.Lock()
_signs_cache: list[dict] | None = None


def _retrain(signs: list[dict] | None = None) -> dict:
    try:
        from . import ml_model
        data = signs if signs is not None else load_signs()
        return ml_model.train_from_signs(data)
    except Exception as e:
        return {'ready': False, 'error': str(e)}


def _default_signs() -> list[dict]:
    return [
        {'id': i, 'name': name, 'samples': [], 'verified': False, 'verifyHits': 0}
        for i, name in enumerate(DEFAULT_WORDS)
    ]


def _validate_sample(sample) -> list:
    arr = resample(sample)
    if arr.shape != (SEQ_LEN, FEAT_DIM):
        raise ValueError('أبعاد العينة غير صالحة')
    if not (abs(arr) <= 100).all():
        raise ValueError('قيم العينة خارج النطاق')
    return arr.tolist()


def _migrate_samples(signs: list[dict]) -> list[dict]:
    """Resample stored clips to current SEQ_LEN without wiping training."""
    out = []
    for i, s in enumerate(signs):
        samples = []
        for sample in (s.get('samples') or []):
            try:
                samples.append(_validate_sample(sample))
            except Exception:
                continue
        out.append({
            'id': i,
            'name': str(s.get('name') or f'كلمة {i + 1}')[:40],
            'samples': samples,
            'verified': bool(s.get('verified')),
            'verifyHits': int(s.get('verifyHits') or 0),
        })
    return out


def load_signs() -> list[dict]:
    global _signs_cache
    with _lock:
        if _signs_cache is not None:
            return _signs_cache
        if not SIGNS_PATH.exists():
            signs = _default_signs()
            SIGNS_PATH.write_text(
                json.dumps({'version': DATA_VERSION, 'signs': signs}, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            _signs_cache = signs
            return signs
        data = json.loads(SIGNS_PATH.read_text(encoding='utf-8'))
        if not isinstance(data.get('signs'), list):
            signs = _default_signs()
            _save_unlocked(signs)
            return signs
        signs = data['signs']
        # Soft migrate older versions (keep words + resample sequences)
        if data.get('version') != DATA_VERSION:
            signs = _migrate_samples(signs)
            _save_unlocked(signs)
            return signs
        _signs_cache = signs
        return signs


def _save_unlocked(signs: list[dict]) -> None:
    global _signs_cache
    SIGNS_PATH.write_text(
        json.dumps({'version': DATA_VERSION, 'signs': signs}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    _signs_cache = signs


def save_signs(signs: list[dict]) -> None:
    with _lock:
        _save_unlocked(signs)


def list_public(signs: list[dict] | None = None) -> list[dict]:
    signs = signs or load_signs()
    return [
        {
            'id': s['id'],
            'name': s['name'],
            'samples': len(s.get('samples') or []),
            'ready': len(s.get('samples') or []) >= MIN_SAMPLES,
            'verified': bool(s.get('verified')),
            'verifyHits': int(s.get('verifyHits') or 0),
        }
        for s in signs
    ]


def add_sign(name: str) -> dict:
    name = (name or '').strip()[:40]
    if not name:
        raise ValueError('الاسم فارغ')
    with _lock:
        if SIGNS_PATH.exists():
            signs = json.loads(SIGNS_PATH.read_text(encoding='utf-8')).get('signs') or _default_signs()
        else:
            signs = _default_signs()
        if len(signs) >= MAX_SIGNS:
            raise ValueError(f'الحد الأقصى {MAX_SIGNS}')
        sign = {'id': len(signs), 'name': name, 'samples': [], 'verified': False, 'verifyHits': 0}
        signs.append(sign)
        _save_unlocked(signs)
        return {'id': sign['id'], 'name': sign['name'], 'samples': 0, 'ready': False, 'verified': False, 'verifyHits': 0}


def rename_sign(sign_id: int, name: str) -> dict:
    name = (name or '').strip()[:40]
    if not name:
        raise ValueError('الاسم فارغ')
    from . import tts

    with _lock:
        if not SIGNS_PATH.exists():
            signs = _default_signs()
        else:
            signs = json.loads(SIGNS_PATH.read_text(encoding='utf-8')).get('signs') or _default_signs()
        if sign_id < 0 or sign_id >= len(signs):
            raise ValueError('إشارة غير موجودة')
        old_name = signs[sign_id]['name']
        signs[sign_id]['name'] = name
        signs[sign_id]['verified'] = False
        _save_unlocked(signs)
        public = list_public(signs)[sign_id]
    if old_name != name:
        try:
            tts.delete_clip(sign_id, old_name)
        except Exception:
            pass
    return public


def clear_samples(sign_id: int) -> dict:
    signs = load_signs()
    with _lock:
        if sign_id < 0 or sign_id >= len(signs):
            raise ValueError('إشارة غير موجودة')
        signs[sign_id]['samples'] = []
        signs[sign_id]['verified'] = False
        signs[sign_id]['verifyHits'] = 0
        snapshot = json.loads(json.dumps(signs))
        _save_unlocked(signs)
        public = list_public(signs)[sign_id]
    _retrain(snapshot)
    return public


def delete_sign(sign_id: int) -> list[dict]:
    """Remove a word entirely, reindex ids, and retrain."""
    from . import tts

    with _lock:
        if not SIGNS_PATH.exists():
            signs = _default_signs()
        else:
            signs = json.loads(SIGNS_PATH.read_text(encoding='utf-8')).get('signs') or _default_signs()
        if sign_id < 0 or sign_id >= len(signs):
            raise ValueError('إشارة غير موجودة')
        if len(signs) <= 1:
            raise ValueError('يجب الإبقاء على كلمة واحدة على الأقل')

        tts.delete_clips_for_sign(sign_id)
        signs.pop(sign_id)
        for i, s in enumerate(signs):
            old_id = int(s.get('id', i))
            if old_id != i:
                tts.rekey_clips(old_id, i)
            s['id'] = i
        snapshot = json.loads(json.dumps(signs))
        _save_unlocked(signs)
        public = list_public(signs)
    _retrain(snapshot)
    return public


def add_sample(sign_id: int, frames: list) -> dict:
    if not isinstance(frames, list) or len(frames) < 4:
        raise ValueError('عدد الإطارات غير كافٍ')
    from . import ml_model

    analysis = ml_model.analyze_gesture(frames)
    if not analysis.get('ok'):
        raise ValueError('اليد غير واضحة — أظهر اليد داخل الإطار وثبّت الإشارة أو حرّكها بوضوح')

    q = quality_score(frames)
    # Moving signs score higher; static held poses still get a fair baseline
    if analysis.get('static'):
        q = max(q, 62)
    if float(analysis.get('mean') or 0) >= 0.04:
        q = max(q, 72)
    if int(analysis.get('peaks') or 0) >= 2:
        q = max(q, 80)
    sample = _validate_sample(frames)
    signs = load_signs()
    with _lock:
        if sign_id < 0 or sign_id >= len(signs):
            raise ValueError('إشارة غير موجودة')
        if len(signs[sign_id]['samples']) >= MAX_SAMPLES:
            raise ValueError('وصلت للحد الأقصى من الأمثلة')
        signs[sign_id]['samples'].append(sample)
        signs[sign_id]['verified'] = False
        count = len(signs[sign_id]['samples'])
        snapshot = json.loads(json.dumps(signs))
        _save_unlocked(signs)
        public = list_public(signs)[sign_id]
    model_meta = _retrain(snapshot)
    return {
        'sign': public,
        'quality': max(q, 35),
        'count': count,
        'ready': count >= MIN_SAMPLES,
        'minSamples': MIN_SAMPLES,
        'model': model_meta,
        'peaks': int(analysis.get('peaks') or 0),
        'gesture': analysis,
    }


def mark_verified(sign_id: int) -> dict:
    signs = load_signs()
    with _lock:
        if sign_id < 0 or sign_id >= len(signs):
            raise ValueError('إشارة غير موجودة')
        signs[sign_id]['verified'] = True
        signs[sign_id]['verifyHits'] = int(signs[sign_id].get('verifyHits') or 0) + 1
        _save_unlocked(signs)
        return list_public(signs)[sign_id]


def export_data() -> dict:
    return {'version': DATA_VERSION, 'signs': load_signs()}


def import_data(payload: dict) -> list[dict]:
    if not isinstance(payload.get('signs'), list):
        raise ValueError('ملف التدريب غير صالح')
    ver = payload.get('version')
    if ver not in (DATA_VERSION, 2, 3):
        raise ValueError('ملف التدريب غير صالح')
    signs = payload['signs']
    if ver != DATA_VERSION:
        signs = _migrate_samples(signs)
    if not (1 <= len(signs) <= MAX_SIGNS):
        raise ValueError('عدد الإشارات غير صالح')
    cleaned = []
    for i, s in enumerate(signs):
        if s.get('id') != i or not isinstance(s.get('name'), str) or not s['name'].strip():
            raise ValueError('بيانات الإشارات غير صالحة')
        samples = s.get('samples') or []
        if len(samples) > MAX_SAMPLES:
            raise ValueError('بيانات الإشارات غير صالحة')
        ok_samples = [_validate_sample(sample) for sample in samples]
        cleaned.append({
            'id': i,
            'name': s['name'].strip()[:40],
            'samples': ok_samples,
            'verified': bool(s.get('verified')),
            'verifyHits': int(s.get('verifyHits') or 0),
        })
    save_signs(cleaned)
    _retrain()
    return list_public(cleaned)
