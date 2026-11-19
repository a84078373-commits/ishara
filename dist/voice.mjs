/** ElevenLabs: generate once, cache in IndexedDB, replay locally. */
export const DEFAULT_VOICE_ID = 'aMmeBf0lzDYlouyfqNjh';
const CFG_KEY = 'ishara-elevenlabs-cfg';
const DB_NAME = 'ishara-voice-clips-v1';
const STORE = 'clips';

export function getVoiceConfig() {
  try {
    const raw = localStorage.getItem(CFG_KEY);
    if (!raw) return { apiKey: '', voiceId: DEFAULT_VOICE_ID };
    const parsed = JSON.parse(raw);
    return {
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey.trim() : '',
      voiceId: (typeof parsed.voiceId === 'string' && parsed.voiceId.trim()) || DEFAULT_VOICE_ID,
    };
  } catch {
    return { apiKey: '', voiceId: DEFAULT_VOICE_ID };
  }
}

export function setVoiceConfig({ apiKey, voiceId }) {
  const next = {
    apiKey: (apiKey ?? getVoiceConfig().apiKey).trim(),
    voiceId: (voiceId ?? getVoiceConfig().voiceId).trim() || DEFAULT_VOICE_ID,
  };
  localStorage.setItem(CFG_KEY, JSON.stringify(next));
  return next;
}

function clipKey(signId, text) {
  return `${signId}::${String(text).trim()}`;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || Error('IndexedDB unavailable'));
  });
}

export async function getCachedClip(signId, text) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(clipKey(signId, text));
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function hasCachedClip(signId, text) {
  const blob = await getCachedClip(signId, text);
  return !!(blob && blob.size);
}

export async function putCachedClip(signId, text, blob) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(blob, clipKey(signId, text));
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

export async function deleteCachedClip(signId, text) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(clipKey(signId, text));
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllClips() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

const inflight = new Map();

/** Fetch from ElevenLabs only if not cached. Dedupes parallel requests for the same word. */
export async function ensureClip(signId, text, { force = false } = {}) {
  const clean = String(text || '').trim();
  if (!clean) throw Error('لا توجد كلمة لتوليد الصوت');
  const key = clipKey(signId, clean);

  if (!force) {
    const cached = await getCachedClip(signId, clean);
    if (cached?.size) return { blob: cached, fromCache: true };
    if (inflight.has(key)) return inflight.get(key);
  }

  const job = (async () => {
    const { apiKey, voiceId } = getVoiceConfig();
    if (!apiKey) throw Error('أضف مفتاح ElevenLabs من الإعدادات أولًا');

    const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
        Accept: 'audio/mpeg',
      },
      body: JSON.stringify({
        text: clean,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.45, similarity_boost: 0.8 },
      }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      if (res.status === 401) throw Error('مفتاح ElevenLabs غير صالح');
      if (res.status === 429) throw Error('تجاوزت حد ElevenLabs؛ انتظر قليلًا');
      throw Error(`فشل توليد الصوت (${res.status})${detail ? ': ' + detail.slice(0, 120) : ''}`);
    }

    const blob = await res.blob();
    if (!blob.size) throw Error('ملف الصوت فارغ من ElevenLabs');
    await putCachedClip(signId, clean, blob);
    return { blob, fromCache: false };
  })();

  inflight.set(key, job);
  try {
    return await job;
  } finally {
    inflight.delete(key);
  }
}

let currentAudio = null;

export async function playClip(blob) {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = '';
    currentAudio = null;
  }
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  currentAudio = audio;
  audio.onended = () => {
    URL.revokeObjectURL(url);
    if (currentAudio === audio) currentAudio = null;
  };
  audio.onerror = () => {
    URL.revokeObjectURL(url);
    if (currentAudio === audio) currentAudio = null;
  };
  await audio.play();
  return audio;
}

export async function prefetchSignVoice(signId, text) {
  return ensureClip(signId, text, { force: false });
}
