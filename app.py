"""Ishara — Flask server (UI + high-precision recognition API + TTS cache)."""
from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from backend import ml_model, recognition, storage, tts
from backend.config import DIST_DIR, MIN_SAMPLES

app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path='')
CORS(app)

# Train model on boot if data exists
try:
    ml_model.train_from_signs(storage.load_signs())
except Exception:
    pass


def _err(exc: Exception, code: int = 400):
    return jsonify({'ok': False, 'error': str(exc)}), code


def _cache_voice_now(sign_id: int, text: str) -> dict:
    """Generate+save mp3 at train time so live/test speak is instant."""
    try:
        path, from_cache = tts.ensure_clip(sign_id, text)
        return {
            'voiceCached': True,
            'fromCache': from_cache,
            'bytes': path.stat().st_size if path.exists() else 0,
        }
    except Exception as e:
        return {'voiceCached': False, 'voiceError': str(e)}


def _cache_voice_async(sign_id: int, text: str) -> None:
    """Background warmup only — never used on the live speak path."""
    def job():
        try:
            tts.ensure_clip(sign_id, text)
        except Exception:
            pass
    threading.Thread(target=job, daemon=True).start()


@app.get('/')
def index():
    return send_from_directory(DIST_DIR, 'index.html')


@app.get('/api/health')
def health():
    signs = storage.list_public()
    ready = sum(1 for s in signs if s['ready'])
    model = ml_model.status()
    missing = tts.missing_for_signs(storage.load_signs())
    return jsonify({
        'ok': True,
        'engine': 'flask+ml+dtw',
        'minSamples': MIN_SAMPLES,
        'signs': len(signs),
        'ready': ready,
        'model': model,
        'tts': True,
        'voiceMissing': len(missing),
    })


@app.post('/api/model/retrain')
def retrain_model():
    try:
        meta = ml_model.train_from_signs(storage.load_signs())
        return jsonify({'ok': True, 'model': meta})
    except Exception as e:
        return _err(e, 500)


@app.get('/api/signs')
def get_signs():
    signs = storage.list_public()
    full = storage.load_signs()
    for s in signs:
        s['voiceCached'] = tts.has_clip(s['id'], s['name'])
    return jsonify({'ok': True, 'signs': signs, 'minSamples': MIN_SAMPLES})


@app.post('/api/signs')
def create_sign():
    try:
        name = (request.json or {}).get('name', '')
        sign = storage.add_sign(name)
        voice = _cache_voice_now(sign['id'], sign['name'])
        sign['voiceCached'] = bool(voice.get('voiceCached'))
        return jsonify({'ok': True, 'sign': sign, **voice})
    except Exception as e:
        return _err(e)


@app.patch('/api/signs/<int:sign_id>')
def patch_sign(sign_id: int):
    try:
        name = (request.json or {}).get('name', '')
        sign = storage.rename_sign(sign_id, name)
        voice = _cache_voice_now(sign_id, sign['name'])
        sign['voiceCached'] = bool(voice.get('voiceCached'))
        return jsonify({'ok': True, 'sign': sign, **voice})
    except Exception as e:
        return _err(e)


@app.delete('/api/signs/<int:sign_id>')
def delete_sign(sign_id: int):
    try:
        signs = storage.delete_sign(sign_id)
        for s in signs:
            s['voiceCached'] = tts.has_clip(s['id'], s['name'])
        return jsonify({'ok': True, 'signs': signs})
    except Exception as e:
        return _err(e)


@app.delete('/api/signs/<int:sign_id>/samples')
def delete_samples(sign_id: int):
    try:
        return jsonify({'ok': True, 'sign': storage.clear_samples(sign_id)})
    except Exception as e:
        return _err(e)


@app.post('/api/signs/<int:sign_id>/samples')
def post_sample(sign_id: int):
    try:
        frames = (request.json or {}).get('frames') or []
        result = storage.add_sample(sign_id, frames)
        # Generate+save voice during training so test/live pulls a ready file
        voice = _cache_voice_now(sign_id, result['sign']['name'])
        result.update(voice)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return _err(e)


@app.post('/api/signs/<int:sign_id>/verify')
def verify_sign(sign_id: int):
    try:
        return jsonify({'ok': True, 'sign': storage.mark_verified(sign_id)})
    except Exception as e:
        return _err(e)


def _downsample_frames(frames: list, max_n: int = 72) -> list:
    """Keep multi-part timing but cap payload size for speed."""
    n = len(frames)
    if n <= max_n:
        return frames
    import numpy as np
    idx = np.linspace(0, n - 1, max_n)
    return [frames[int(round(i))] for i in idx]


@app.post('/api/recognize')
def recognize():
    """
    Instant path: ExtraTrees only when the model is ready (skip slow DTW).
    DTW is fallback only when no trained model exists.
    """
    try:
        body = request.json or {}
        frames = body.get('frames') or []
        expected_id = body.get('expectedId')
        if expected_id is not None:
            try:
                expected_id = int(expected_id)
            except (TypeError, ValueError):
                expected_id = None
        if len(frames) < 5:
            return jsonify({'ok': True, 'match': None, 'peek': None, 'reason': 'short'})
        frames = _downsample_frames(frames, 48)
        signs = storage.load_signs()
        model = ml_model.status()

        if model.get('ready'):
            ml = ml_model.predict(frames, signs, expected_id=expected_id) or {}
            if not ml:
                return jsonify({'ok': True, 'match': None, 'peek': None, 'reason': 'empty', 'engine': 'ml'})
            ml.setdefault('engine', 'ml')
            return jsonify({'ok': True, **ml})

        # No ML model yet — lightweight DTW fallback
        dtw = recognition.classify(frames, signs) or {}
        if dtw.get('match'):
            dtw['match']['engine'] = 'dtw'
            dtw['engine'] = 'dtw'
            return jsonify({'ok': True, **dtw})
        return jsonify({
            'ok': True,
            'match': None,
            'peek': dtw.get('peek'),
            'ranked': dtw.get('ranked'),
            'engine': 'dtw',
            'reason': dtw.get('reason') or 'low_conf',
        })
    except Exception as e:
        return _err(e, 500)


@app.post('/api/tts/<int:sign_id>')
def ensure_tts(sign_id: int):
    """Generate+cache clip once (training/settings only — not live speak)."""
    try:
        payload = request.json or {}
        text = payload.get('text') or ''
        force = bool(payload.get('force'))
        if not text:
            signs = storage.load_signs()
            text = signs[sign_id]['name']
        path, from_cache = tts.ensure_clip(sign_id, text, force=force)
        return jsonify({
            'ok': True,
            'fromCache': from_cache,
            'url': f'/api/audio/{sign_id}?t={text}',
            'bytes': path.stat().st_size,
        })
    except Exception as e:
        return _err(e, 500)


@app.post('/api/tts/warmup')
def warmup_tts():
    """Generate+cache mp3 for every word in the library (add-time policy)."""
    try:
        signs = storage.load_signs()
        cached = 0
        generated = 0
        errors = []
        for s in signs:
            name = (s.get('name') or '').strip()
            if not name:
                continue
            try:
                _, from_cache = tts.ensure_clip(int(s['id']), name)
                if from_cache:
                    cached += 1
                else:
                    generated += 1
            except Exception as e:
                errors.append({'id': s['id'], 'name': name, 'error': str(e)})
        return jsonify({
            'ok': True,
            'cached': cached,
            'generated': generated,
            'errors': errors,
        })
    except Exception as e:
        return _err(e, 500)


@app.get('/api/audio/<int:sign_id>')
def get_audio(sign_id: int):
    """Serve cached mp3 only — never call ElevenLabs here (keeps speak instant)."""
    try:
        text = request.args.get('t') or ''
        if not text:
            text = storage.load_signs()[sign_id]['name']
        path = tts.clip_path(sign_id, text)
        if not path:
            return jsonify({'ok': False, 'error': 'لا يوجد مقطع محفوظ لهذه الكلمة'}), 404
        return send_file(path, mimetype='audio/mpeg', conditional=True)
    except Exception as e:
        return _err(e, 404)


@app.get('/api/export')
def export_training():
    return jsonify(storage.export_data())


@app.post('/api/import')
def import_training():
    try:
        signs = storage.import_data(request.json or {})
        for s in signs:
            _cache_voice_async(s['id'], s['name'])
            s['voiceCached'] = tts.has_clip(s['id'], s['name'])
        return jsonify({'ok': True, 'signs': signs})
    except Exception as e:
        return _err(e)


# SPA / static assets
@app.route('/<path:path>')
def static_proxy(path: str):
    return send_from_directory(DIST_DIR, path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5173'))
    print(f'Ishara Flask → http://127.0.0.1:{port}')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
