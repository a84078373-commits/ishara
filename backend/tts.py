"""TTS — generate once, cache on disk.

Providers (in order):
1) ElevenLabs (if key works)
2) Microsoft Edge neural voices via edge-tts (free, good Arabic)
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import requests

from .config import AUDIO_DIR, ELEVEN_API_KEY, ELEVEN_VOICE_ID

# High-quality Arabic neural voices (Microsoft)
EDGE_VOICE = os.getenv('EDGE_TTS_VOICE', 'ar-SA-ZariyahNeural')


def _safe_name(sign_id: int, text: str) -> Path:
    slug = re.sub(r'[^\w\u0600-\u06FF]+', '_', text.strip())[:40] or 'clip'
    return AUDIO_DIR / f'{sign_id}_{slug}.mp3'


def has_clip(sign_id: int, text: str) -> bool:
    path = _safe_name(sign_id, text)
    return path.exists() and path.stat().st_size > 0


def clip_path(sign_id: int, text: str) -> Path | None:
    path = _safe_name(sign_id, text)
    return path if path.exists() and path.stat().st_size > 0 else None


def _elevenlabs_generate(text: str, path: Path) -> None:
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
            raw = res.json().get('detail') or {}
            if isinstance(raw, dict):
                detail = raw.get('message') or raw.get('code') or ''
            else:
                detail = str(raw)
        except Exception:
            detail = (res.text or '')[:160]
        if 'payment' in str(detail).lower():
            raise RuntimeError('ElevenLabs: مشكلة دفع في الاشتراك')
        raise RuntimeError('مفتاح ElevenLabs غير صالح أو مرفوض')
    if res.status_code == 429:
        raise RuntimeError('تجاوزت حد ElevenLabs')
    if not res.ok:
        raise RuntimeError(f'فشل ElevenLabs ({res.status_code})')
    if not res.content:
        raise RuntimeError('ملف الصوت فارغ من ElevenLabs')
    path.write_bytes(res.content)


def _edge_generate(text: str, path: Path) -> None:
    try:
        import edge_tts
    except ImportError as e:
        raise RuntimeError('edge-tts غير مثبت — pip install edge-tts') from e

    async def _run():
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        await communicate.save(str(path))

    asyncio.run(_run())
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError('فشل توليد الصوت عبر Microsoft TTS')


def ensure_clip(sign_id: int, text: str, force: bool = False) -> tuple[Path, bool]:
    """Generate mp3 once and cache. Prefer ElevenLabs, else free Edge TTS."""
    text = (text or '').strip()
    if not text:
        raise ValueError('لا توجد كلمة لتوليد الصوت')
    path = _safe_name(sign_id, text)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path, True

    errors: list[str] = []

    # 1) ElevenLabs if key present
    if ELEVEN_API_KEY:
        try:
            _elevenlabs_generate(text, path)
            return path, False
        except Exception as e:
            errors.append(str(e))
            if path.exists() and path.stat().st_size <= 0:
                try:
                    path.unlink()
                except OSError:
                    pass

    # 2) Free Microsoft neural Arabic (edge-tts)
    try:
        _edge_generate(text, path)
        return path, False
    except Exception as e:
        errors.append(str(e))
        if path.exists() and path.stat().st_size <= 0:
            try:
                path.unlink()
            except OSError:
                pass

    raise RuntimeError('تعذّر توليد الصوت: ' + ' | '.join(errors))


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
