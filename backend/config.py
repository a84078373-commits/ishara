import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / 'dist'

# On Render Free, filesystem is ephemeral unless a disk is mounted at /var/data
_data_env = os.getenv('DATA_DIR', '').strip()
if _data_env:
    DATA_DIR = Path(_data_env)
elif Path('/var/data').exists():
    DATA_DIR = Path('/var/data/ishara')
else:
    DATA_DIR = ROOT / 'data'

SIGNS_PATH = DATA_DIR / 'signs.json'
AUDIO_DIR = DATA_DIR / 'audio'

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 32 frames preserves two-beat gestures (part A → pause → part B)
SEQ_LEN = 32
FEAT_DIM = 152
MIN_SAMPLES = 2
MAX_SAMPLES = 12
MAX_SIGNS = 48
DATA_VERSION = 4

DEFAULT_WORDS = [
    'مرحبًا', 'شكرًا', 'نعم', 'لا', 'من فضلك', 'آسف', 'أنا', 'أنت', 'مساعدة', 'ماء',
    'طعام', 'بيت', 'مدرسة', 'أهلًا وسهلًا', 'كيف حالك', 'بخير', 'أحبك', 'انتظر', 'أين', 'مع السلامة',
]

ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY', '')
ELEVEN_VOICE_ID = os.getenv('ELEVEN_VOICE_ID', 'aMmeBf0lzDYlouyfqNjh')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', 'nova')
OPENAI_TTS_MODEL = os.getenv('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts')

# local-config.js overrides empty/env for local secrets (gitignored)
_local = DIST_DIR / 'local-config.js'
if _local.exists():
    text = _local.read_text(encoding='utf-8')
    for line in text.splitlines():
        if "'" not in line:
            continue
        val = line.split("'")[1].strip()
        if not val:
            continue
        if 'OPENAI_API_KEY' in line:
            OPENAI_API_KEY = val
        elif 'OPENAI_TTS_VOICE' in line:
            OPENAI_TTS_VOICE = val
        elif 'ELEVEN_API_KEY' in line:
            ELEVEN_API_KEY = val
        elif 'ELEVEN_VOICE_ID' in line:
            ELEVEN_VOICE_ID = val


