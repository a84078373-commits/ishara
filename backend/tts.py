"""ElevenLabs TTS — generate once, cache on disk."""
from __future__ import annotations

import re
from pathlib import Path

import requests

from .config import AUDIO_DIR, ELEVEN_API_KEY, ELEVEN_VOICE_ID


def _safe_name(sign_id: int, text: str) -> Path:
    slug = re.sub(r'[^\w\u0600-\u06FF]+', '_', text.strip())[:40] or 'clip'
    return AUDIO_DIR / f'{sign_id}_{slug}.mp3'


def has_clip(sign_id: int, text: str) -> bool:
    path = _safe_name(sign_id, text)
    return path.exists() and path.stat().st_size > 0


def clip_path(sign_id: int, text: str) -> Path | None:
    path = _safe_name(sign_id, text)
    return path if path.exists() and path.stat().st_size > 0 else None


def ensure_clip(sign_id: int, text: str, force: bool = False) -> tuple[Path, bool]:
    text = (text or '').strip()
    if not text:
        raise ValueError('لا توجد كلمة لتوليد الصوت')
    path = _safe_name(sign_id, text)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path, True

    if not ELEVEN_API_KEY:
        raise RuntimeError('مفتاح ElevenLabs غير مضبوط')

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}'
    res = requests.post(
        url,
        headers={
            'xi-api-key': ELEVEN_API_KEY,
            'Content-Type': 'application/json',
            'Accept': 'audio/mpeg',
        },
        json={
            'text': text,
            'model_id': 'eleven_multilingual_v2',
            'voice_settings': {'stability': 0.45, 'similarity_boost': 0.8},
        },
        timeout=60,
    )
    if res.status_code == 401:
        detail = ''
        try:
            detail = (res.json().get('detail') or {})
            if isinstance(detail, dict):
                detail = detail.get('message') or detail.get('code') or ''
        except Exception:
            detail = (res.text or '')[:160]
        if 'payment' in str(detail).lower():
            raise RuntimeError('ElevenLabs: مشكلة دفع في الاشتراك — أكمل الفاتورة من لوحة ElevenLabs')
        raise RuntimeError('مفتاح ElevenLabs غير صالح أو مرفوض')
    if res.status_code == 429:
        raise RuntimeError('تجاوزت حد ElevenLabs')
    if not res.ok:
        raise RuntimeError(f'فشل توليد الصوت ({res.status_code})')
    if not res.content:
        raise RuntimeError('ملف الصوت فارغ')
    path.write_bytes(res.content)
    return path, False


def delete_clip(sign_id: int, text: str) -> None:
    path = _safe_name(sign_id, text)
    if path.exists():
        path.unlink()


def delete_clips_for_sign(sign_id: int) -> None:
    if not AUDIO_DIR.exists():
        return
    for path in AUDIO_DIR.glob(f'{sign_id}_*.mp3'):
        try:
            path.unlink()
        except OSError:
            pass


def rekey_clips(old_id: int, new_id: int) -> None:
    """Rename cached clips when a sign id changes after delete/reindex."""
    if old_id == new_id or not AUDIO_DIR.exists():
        return
    for old in list(AUDIO_DIR.glob(f'{old_id}_*.mp3')):
        suffix = old.name.split('_', 1)[1]
        new = AUDIO_DIR / f'{new_id}_{suffix}'
        try:
            if new.exists():
                new.unlink()
            old.rename(new)
        except OSError:
            pass


def missing_for_signs(signs: list[dict]) -> list[dict]:
    missing = []
    for s in signs:
        name = (s.get('name') or '').strip()
        if name and not has_clip(int(s['id']), name):
            missing.append({'id': int(s['id']), 'name': name})
    return missing
