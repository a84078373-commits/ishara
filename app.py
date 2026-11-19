"""Ishara — Flask server (UI + high-precision recognition API + TTS cache)."""
from __future__ import annotations

import os

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


@app.get('/')
def index():
    return send_from_directory(DIST_DIR, 'index.html')


@app.get('/api/health')
def health():
    signs = storage.list_public()
    ready = sum(1 for s in signs if s['ready'])
    model = ml_model.status()
    return jsonify({
        'ok': True,
        'engine': 'flask+ml+dtw',
        'minSamples': MIN_SAMPLES,
        'signs': len(signs),
        'ready': ready,
        'model': model,
        'tts': True,
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
    return jsonify({'ok': True, 'signs': storage.list_public(), 'minSamples': MIN_SAMPLES})


@app.post('/api/signs')
def create_sign():
    try:
        name = (request.json or {}).get('name', '')
        return jsonify({'ok': True, 'sign': storage.add_sign(name)})
    except Exception as e:
        return _err(e)


@app.patch('/api/signs/<int:sign_id>')
def patch_sign(sign_id: int):
    try:
        name = (request.json or {}).get('name', '')
        return jsonify({'ok': True, 'sign': storage.rename_sign(sign_id, name)})
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
        # Prefetch voice once when ready
        if result['ready']:
            try:
                tts.ensure_clip(sign_id, result['sign']['name'])
                result['voiceCached'] = True
            except Exception as voice_err:
                result['voiceCached'] = False
                result['voiceError'] = str(voice_err)
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

        # 1) ML first (fast). Skip heavy DTW when ML already confident.
        ml = ml_model.predict(frames, signs)
        if ml and ml.get('match'):
            return jsonify({'ok': True, **ml})
        if ml and ml.get('peek') and ml['peek'].get('similarity', 0) >= 70:
            # High peek — return as soft match path for the client without DTW delay
            return jsonify({'ok': True, 'match': None, 'peek': ml['peek'], 'ranked': ml.get('ranked'), 'engine': 'ml'})

        # 2) DTW only as fallback when ML is unsure
        dtw = recognition.classify(frames, signs) or {}
        if dtw.get('match'):
            dtw['match']['engine'] = 'dtw'
            dtw['engine'] = 'dtw'
            return jsonify({'ok': True, **dtw})

        if ml and ml.get('peek'):
            return jsonify({'ok': True, 'match': None, 'peek': ml['peek'], 'ranked': ml.get('ranked'), 'engine': 'ml'})
        return jsonify({'ok': True, 'match': None, 'peek': dtw.get('peek'), 'ranked': dtw.get('ranked'), 'engine': dtw.get('engine') or 'dtw'})
    except Exception as e:
        return _err(e, 500)


@app.post('/api/tts/<int:sign_id>')
def ensure_tts(sign_id: int):
    try:
        payload = request.json or {}
        text = payload.get('text') or ''
        force = bool(payload.get('force'))
        if not text:
            signs = storage.load_signs()
            text = signs[sign_id]['name']
        path, from_cache = tts.ensure_clip(sign_id, text, force=force)
        return jsonify({'ok': True, 'fromCache': from_cache, 'url': f'/api/audio/{sign_id}?t={text}'})
    except Exception as e:
        return _err(e, 500)


@app.get('/api/audio/<int:sign_id>')
def get_audio(sign_id: int):
    try:
        text = request.args.get('t') or storage.load_signs()[sign_id]['name']
        path = tts.clip_path(sign_id, text)
        if not path:
            path, _ = tts.ensure_clip(sign_id, text)
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
