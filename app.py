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


def _cache_voice_async(sign_id: int, text: str) -> None:
    """Generate ElevenLabs clip once in background — never during live speak."""
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
        # Save voice file as soon as the word is added (once)
        _cache_voice_async(sign['id'], sign['name'])
        sign['voiceCached'] = False
        return jsonify({'ok': True, 'sign': sign})
    except Exception as e:
        return _err(e)


@app.patch('/api/signs/<int:sign_id>')
def patch_sign(sign_id: int):
    try:
        name = (request.json or {}).get('name', '')
        sign = storage.rename_sign(sign_id, name)
        _cache_voice_async(sign_id, sign['name'])
        sign['voiceCached'] = tts.has_clip(sign_id, sign['name'])
        return jsonify({'ok': True, 'sign': sign})
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
        # Prefetch voice once when ready — never block recognition path
        if result['ready']:
            _cache_voice_async(sign_id, result['sign']['name'])
            result['voiceQueued'] = True
            result['voiceCached'] = tts.has_clip(sign_id, result['sign']['name'])
        return jsonify({'ok': True, **result})
    except Exception as e:
        return _err(e)


@app.post('/api/signs/<int:sign_id>/verify')
def verify_sign(sign_id: int):
    try:
        return jsonify({'ok': True, 'sign': storage.mark_verified(sign_id)})
    except Exception as e:
        return _err(e)


@app.post('/api/recognize')
def recognize():
    try:
        frames = (request.json or {}).get('frames') or []
        if len(frames) < 4:
            return jsonify({'ok': True, 'match': None, 'peek': None, 'reason': 'short'})
        signs = storage.load_signs()
        model = ml_model.status()

        # Fast path: trained ML only (skip slow DTW for snappy live UX)
        ml = ml_model.predict(frames, signs)
        if model.get('ready'):
            return jsonify({'ok': True, **(ml or {'match': None, 'peek': None, 'engine': 'ml'})})

        if ml and ml.get('match'):
            return jsonify({'ok': True, **ml})

        # Cold start / no model yet — DTW fallback
        dtw = recognition.classify(frames, signs) or {}
        if dtw.get('match'):
            dtw['match']['engine'] = 'dtw'
            dtw['engine'] = 'dtw'
            return jsonify({'ok': True, **dtw})
        if ml and ml.get('peek'):
            return jsonify({'ok': True, 'match': None, 'peek': ml['peek'], 'ranked': ml.get('ranked'), 'engine': 'ml'})
        return jsonify({
            'ok': True,
            'match': None,
            'peek': dtw.get('peek'),
            'ranked': dtw.get('ranked'),
            'engine': dtw.get('engine') or 'dtw',
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
    """Cache mp3 for trained words only (never during live speak)."""
    try:
        signs = storage.load_signs()
        cached = 0
        generated = 0
        errors = []
        for s in signs:
            name = (s.get('name') or '').strip()
            samples = s.get('samples') or []
            # Trained words + any already-cached stay warm; skip empty dictionary stubs
            if not name:
                continue
            if len(samples) < MIN_SAMPLES and not tts.has_clip(int(s['id']), name):
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
