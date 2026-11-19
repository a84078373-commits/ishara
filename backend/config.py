import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
SIGNS_PATH = DATA_DIR / 'signs.json'
AUDIO_DIR = DATA_DIR / 'audio'
DIST_DIR = ROOT / 'dist'

DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

SEQ_LEN = 16
FEAT_DIM = 152
MIN_SAMPLES = 2
MAX_SAMPLES = 12
MAX_SIGNS = 48
DATA_VERSION = 2

DEFAULT_WORDS = [
    'مرحبًا', 'شكرًا', 'نعم', 'لا', 'من فضلك', 'آسف', 'أنا', 'أنت', 'مساعدة', 'ماء',
    'طعام', 'بيت', 'مدرسة', 'أهلًا وسهلًا', 'كيف حالك', 'بخير', 'أحبك', 'انتظر', 'أين', 'مع السلامة',
]

ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY', '')
ELEVEN_VOICE_ID = os.getenv('ELEVEN_VOICE_ID', 'aMmeBf0lzDYlouyfqNjh')

# Load from dist/local-config.js if env empty
_local = DIST_DIR / 'local-config.js'
if (not ELEVEN_API_KEY) and _local.exists():
    text = _local.read_text(encoding='utf-8')
    for line in text.splitlines():
        if 'ELEVEN_API_KEY' in line and "'" in line:
            ELEVEN_API_KEY = line.split("'")[1]
        if 'ELEVEN_VOICE_ID' in line and "'" in line:
            ELEVEN_VOICE_ID = line.split("'")[1]
