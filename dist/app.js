import {
  features, resample, classify, peekMatch, validateData, createDefaultSigns, buildWindows,
  MAX_SAMPLES, MIN_SAMPLES, MAX_SIGNS, DATA_VERSION, qualityScore, selfConsistency,
} from './core.mjs';
import {
  getVoiceConfig, setVoiceConfig, ensureClip, playClip, hasCachedClip,
  getCachedClip, deleteCachedClip, DEFAULT_VOICE_ID,
} from './voice.mjs';
import * as api from './api.mjs';

const $ = (id) => document.getElementById(id);
const KEY = 'ishara-saudi-training-v2';

/** Map API public sign → UI sign (samples length only; full samples live on Flask). */
function fromApiSign(s) {
  return {
    id: s.id,
    name: s.name,
    samples: Array(s.samples || 0).fill(null),
    verified: !!s.verified,
    verifyHits: s.verifyHits || 0,
  };
}

let signs = createDefaultSigns();
let flaskMode = false;
let engineLabel = 'محلي';

let model, stream, running = false, starting = false, raf, lastVideo = -1, lastDetect = 0;
let lastFeature = 0, buffer = [], current = null, selected = 0, recording = false, capture = [];
let mirror = true, pending = null, pendingAt = 0, latched = null, latchedAt = 0, noHandAt = 0;
let frameCount = 0, fpsAt = 0, toastTimer, session = 0;
let smoothHands = [], trails = [[], []];
let verifyTarget = null;
let verifyFailAt = 0;
let recognizing = false;
let lastRecognizeAt = 0;
const SAME_SIGN_COOLDOWN_MS = 1800;
/** Live two-beat gestures: wait for parts, then settle → fast speak. */
const LIVE_BUFFER_MS = 4800;
const GESTURE_SETTLE_MS = 420;       // after 2nd beat, confirm quickly
const GESTURE_SETTLE_1BEAT_MS = 700; // single-beat signs need longer settle
const MIN_GESTURE_MS = 900;
const MAX_GESTURE_MS = 5000;
const MOTION_THRESH = 0.11;
const MIN_MOTION_FRAMES = 6;
const MIN_MOTION_SUM = 1.1;
const MIN_MATCH_SCORE = 74;
let gestureArmed = false;
let gestureStartAt = 0;
let lastMotionAt = 0;
let lastFeatFrame = null;
let softPeekAt = 0;
let motionFrames = 0;
let motionSum = 0;
let livePeaks = 0;
let peakActive = false;

const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d', { alpha: true });

function toast(text) {
  $('status').textContent = text;
  $('status').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('status').hidden = true; }, 6500);
}

function persist() {
  // In Flask mode training is on the server; keep a tiny local mirror for offline fallback UI.
  if (flaskMode) return true;
  try {
    localStorage.setItem(KEY, JSON.stringify({ version: DATA_VERSION, signs }));
    return true;
  } catch {
    toast('لم يتم حفظ التدريب؛ مساحة المتصفح غير متاحة.');
    return false;
  }
}

async function bootBackend() {
  try {
    const h = await api.health();
    const data = await api.fetchSigns();
    signs = data.signs.map(fromApiSign);
    flaskMode = true;
    engineLabel = h.engine || 'flask';
    const model = h.model || {};
    const modelMsg = model.ready
      ? `نموذج مدرّب · ${model.classes} كلمات · ${model.samples} عينة`
      : 'نموذج جاهز للتدريب — أضف كلمات وسجّل أمثلة';
    toast(`✓ Flask + ML جاهز — ${modelMsg}`);
    const st = $('voice-status');
    if (st) st.textContent = `Flask ML · ${modelMsg}`;
    // Warm every word's mp3 into memory (generate missing once, then instant play)
    warmAllVoices().catch(() => {});
  } catch {
    flaskMode = false;
    engineLabel = 'محلي';
    try {
      const saved = localStorage.getItem(KEY);
      if (saved) signs = validateData(JSON.parse(saved));
    } catch {
      toast('تعذّر قراءة التدريب المحلي.');
    }
    toast('Flask غير متصل — يعمل الوضع المحلي. شغّل: python app.py');
    Promise.allSettled(signs.map(async (s) => {
      try {
        const blob = await getCachedClip(s.id, s.name);
        if (!blob?.size) return;
        const prev = readyAudio.get(s.id);
        if (prev?.src?.startsWith('blob:')) URL.revokeObjectURL(prev.src);
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.preload = 'auto';
        audio._isharaText = s.name;
        readyAudio.set(s.id, audio);
      } catch { /* ignore */ }
    })).then(() => render());
  }
  render();
}

function view(name) {
  $('live-view').hidden = name !== 'live';
  $('library-view').hidden = name !== 'library';
  document.querySelectorAll('.nav').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
}

document.querySelectorAll('.nav').forEach((b) => { b.onclick = () => view(b.dataset.view); });
$('go-library').onclick = () => view('library');
$('help').onclick = () => $('help-dialog').showModal();
$('close-help').onclick = () => $('help-dialog').close();

function updateLiveTrainHint() {
  const ready = signs.filter((s) => s.samples.length >= MIN_SAMPLES);
  const partial = signs.filter((s) => s.samples.length > 0 && s.samples.length < MIN_SAMPLES);
  if (ready.length) {
    $('result-note').textContent = `جاهز: ${ready.length} إشارة مدرّبة — نفّذ الإشارة أمام الكاميرا`;
  } else if (partial.length) {
    const s = partial[0];
    $('result-note').textContent = `«${s.name}»: ${s.samples.length}/${MIN_SAMPLES} — سجّل مثالًا آخر من المكتبة`;
  } else {
    $('result-note').textContent = 'من المكتبة: اختر كلمة ← سجّل الإشارة كاملة (~4 ث) ← مثالان على الأقل';
  }
}

function render() {
  const ready = signs.filter((s) => s.samples.length >= MIN_SAMPLES).length;
  $('trained-count').replaceChildren(
    document.createTextNode(String(ready).padStart(2, '0') + ' '),
    Object.assign(document.createElement('small'), { textContent: `/ ${signs.length}` }),
  );
  const count = String(signs.length);
  document.querySelectorAll('#lib-count-desktop, #lib-count-mobile, .nav[data-view="library"] span').forEach((el) => {
    el.textContent = count;
  });
  if (!current && !verifyTarget && !recording) updateLiveTrainHint();

  $('signs').replaceChildren(...signs.map((s) => {
    const card = document.createElement('article');
    card.className = 'sign-card'
      + (s.samples.length >= MIN_SAMPLES ? ' ready' : '')
      + (s.verified ? ' verified' : '');
    const n = document.createElement('span');
    n.className = 'number';
    n.textContent = String(s.id + 1).padStart(2, '0') + ' / SIGN';
    const title = document.createElement('h3');
    title.textContent = s.name;
    const p = document.createElement('p');
    const voiceBit = readyAudio.has(s.id) ? ' · صوت جاهز' : '';
    p.textContent = s.samples.length >= MIN_SAMPLES
      ? `${s.samples.length} أمثلة · جاهزة للتعرّف${voiceBit}`
      : `${s.samples.length} / ${MIN_SAMPLES} أمثلة · تحتاج تدريبًا`;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.min(100, (s.samples.length / MIN_SAMPLES) * 100)}%`;
    bar.append(fill);
    if (s.verified) {
      const badge = document.createElement('div');
      badge.className = 'badge-ok';
      badge.textContent = `✓ تم التحقق من التعرّف${s.verifyHits > 1 ? ` · ${s.verifyHits}×` : ''}`;
      card.append(n, title, p, badge, bar);
    } else {
      card.append(n, title, p, bar);
    }
    const actions = document.createElement('div');
    actions.className = 'sign-actions';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = s.samples.length ? 'تعديل التدريب ↗' : '＋ تعليم هذه الإشارة';
    btn.onclick = () => openTrainer(s.id);
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'danger-btn';
    del.textContent = 'حذف';
    del.title = 'حذف الكلمة نهائيًا';
    del.onclick = (e) => {
      e.stopPropagation();
      deleteWord(s.id);
    };
    actions.append(btn, del);
    card.append(actions);
    return card;
  }));
}

function openTrainer(id) {
  selected = id;
  $('sign-name').value = signs[id].name;
  updateTakes();
  $('trainer').showModal();
}

function updateTakes() {
  const s = signs[selected];
  const n = s.samples.length;
  $('take-count').textContent =
    `الأمثلة المسجّلة: ${n} / ${MAX_SAMPLES} — يكفي ${MIN_SAMPLES} للتعرّف`;
  $('record').disabled = n >= MAX_SAMPLES;

  const testBtn = $('test-sign');
  const status = $('train-status');
  if (n >= MIN_SAMPLES) {
    testBtn.hidden = false;
    status.hidden = false;
    if (s.verified) {
      status.className = 'train-status ok';
      status.textContent = `✓ تم التحقق من تعرّف «${s.name}». سجّل أمثلة أكثر لدقة أعلى أو اختبر مجددًا.`;
    } else {
      status.className = 'train-status warn';
      status.textContent = `جاهز: اضغط «اختبر التعرّف الآن» ونفّذ «${s.name}» أمام الكاميرا.`;
    }
  } else {
    testBtn.hidden = true;
    status.hidden = false;
    status.className = 'train-status warn';
    status.textContent = `بقي ${MIN_SAMPLES - n} مثال للتعرّف. اضغط «ابدأ التسجيل الآن».`;
  }
}

function setCoach({ title, step, hand, handOk, hint, progress }) {
  const el = $('coach');
  el.hidden = false;
  $('coach-title').textContent = title || 'التدريب';
  $('coach-step').textContent = step || '—';
  const handEl = $('coach-hand');
  handEl.textContent = hand || '○ بانتظار اليد';
  handEl.className = handOk === true ? 'ok' : handOk === false ? 'bad' : '';
  $('coach-hint').textContent = hint || '';
  $('coach-fill').style.width = `${Math.max(0, Math.min(100, progress || 0))}%`;
}
function hideCoach() {
  $('coach').hidden = true;
  $('coach-fill').style.width = '0%';
}

function setVerifyBanner(mode, text) {
  const el = $('verify-banner');
  if (!text) {
    el.hidden = true;
    el.textContent = '';
    el.className = 'verify-banner';
    return;
  }
  el.hidden = false;
  el.className = `verify-banner ${mode}`;
  el.textContent = text;
}

function startVerify(id) {
  verifyTarget = id;
  verifyFailAt = 0;
  buffer = [];
  pending = null;
  latched = null;
  gestureArmed = false;
  motionFrames = 0;
  motionSum = 0;
  livePeaks = 0;
  peakActive = false;
  lastFeatFrame = null;
  $('trainer').close();
  view('live');
  const name = signs[id].name;
  setVerifyBanner('waiting', `⏳ اختبار: «${name}» — حركة 1 ثم حركة 2 ثم توقف`);
  $('result').textContent = 'اختبار التعرّف…';
  $('result-note').textContent = `نفّذ «${name}» بجزئين واضحين — بعده نطق سريع ودقيق`;
  $('result-ring').textContent = '?';
  $('score').textContent = '—';
  $('meter-fill').style.width = '0%';
  toast(`اختبار «${name}»: أكمل كل أجزاء الإشارة ثم توقف قليلًا`);
}

function showSuccess(match) {
  const flash = $('success-flash');
  $('success-title').textContent = '✓ نجح التعرّف';
  $('success-word').textContent = match.name;
  $('success-sub').textContent = `تعرّف النظام على الإشارة التي علّمتها · تشابه ${match.similarity}%`;
  flash.hidden = false;
}

$('success-ok').onclick = () => { $('success-flash').hidden = true; };
$('test-sign').onclick = () => {
  if (!running) {
    $('trainer').close();
    view('live');
    toast('شغّل الكاميرا أولًا ثم اضغط اختبار التعرّف.');
    return;
  }
  startVerify(selected);
};

$('close-trainer').onclick = () => { if (!recording) $('trainer').close(); };
$('trainer').addEventListener('cancel', (e) => { if (recording) e.preventDefault(); });

$('sign-name').onchange = async () => {
  const name = $('sign-name').value.trim();
  if (name) {
    const prev = signs[selected].name;
    try {
      if (flaskMode) {
        const res = await api.renameSign(selected, name);
        signs[selected] = fromApiSign(res.sign);
        if (prev !== name) {
          readyAudio.delete(selected);
          cacheVoiceForSign(signs[selected]).catch(() => {});
        }
      } else {
        if (prev !== name) {
          try { await deleteCachedClip(selected, prev); } catch { /* ignore */ }
          signs[selected].verified = false;
          cacheVoiceForSign(signs[selected]).catch(() => {});
        }
        signs[selected].name = name;
        persist();
      }
      render();
      updateTakes();
    } catch (e) {
      toast(e.message);
      $('sign-name').value = signs[selected].name;
    }
  } else {
    $('sign-name').value = signs[selected].name;
  }
};

$('delete-word')?.addEventListener('click', () => {
  if (recording) return;
  deleteWord(selected);
});

$('delete-sign').onclick = async () => {
  if (recording) return;
  if (!confirm('حذف جميع أمثلة هذه الإشارة؟ (الكلمة تبقى)')) return;
  try {
    if (flaskMode) {
      const res = await api.clearSamples(selected);
      signs[selected] = fromApiSign(res.sign);
    } else {
      signs[selected].samples = [];
      signs[selected].verified = false;
      signs[selected].verifyHits = 0;
      persist();
    }
    if (verifyTarget === selected) {
      verifyTarget = null;
      setVerifyBanner('', '');
    }
    updateTakes();
    render();
  } catch (e) {
    toast(e.message);
  }
};

async function deleteWord(id) {
  if (recording) return;
  const word = signs[id]?.name || '';
  if (!confirm(`حذف الكلمة «${word}» نهائيًا مع أمثلتها والصوت؟`)) return;
  try {
    if (flaskMode) {
      const res = await api.deleteSign(id);
      signs = res.signs.map(fromApiSign);
      clearAudioCache();
      warmAllVoices().catch(() => {});
    } else {
      if (signs.length <= 1) throw Error('يجب الإبقاء على كلمة واحدة على الأقل');
      try { await deleteCachedClip(id, word); } catch { /* ignore */ }
      signs.splice(id, 1);
      signs.forEach((s, i) => { s.id = i; });
      persist();
      clearAudioCache();
    }
    if (selected >= signs.length) selected = Math.max(0, signs.length - 1);
    if (verifyTarget === id) {
      verifyTarget = null;
      setVerifyBanner('', '');
    } else if (verifyTarget != null && verifyTarget > id) {
      verifyTarget -= 1;
    }
    if ($('trainer').open) $('trainer').close();
    latched = null;
    current = null;
    render();
    toast(`تم حذف «${word}»`);
  } catch (e) {
    toast(e.message || 'تعذّر الحذف');
  }
}

$('add-sign')?.addEventListener('click', async () => {
  if (signs.length >= MAX_SIGNS) return toast(`الحد الأقصى ${MAX_SIGNS} إشارة.`);
  const name = prompt('اسم الإشارة / الكلمة الجديدة:', 'كلمة جديدة');
  if (!name?.trim()) return;
  try {
    if (flaskMode) {
      const res = await api.createSign(name.trim());
      signs.push(fromApiSign(res.sign));
      openTrainer(res.sign.id);
      // Generate+cache voice file once in background
      cacheVoiceForSign(signs[res.sign.id]).then((ok) => {
        if (ok) toast(`صوت «${res.sign.name}» محفوظ — النطق سيكون فوريًا`);
      });
    } else {
      const id = signs.length;
      signs.push({ id, name: name.trim().slice(0, 40), samples: [], verified: false, verifyHits: 0 });
      persist();
      openTrainer(id);
      cacheVoiceForSign(signs[id]).catch(() => {});
    }
    render();
    toast('أُضيفت الإشارة. سجّل مثالين على الأقل.');
  } catch (e) {
    toast(e.message);
  }
});

function browserVoice() {
  return window.speechSynthesis?.getVoices().find((v) => v.lang.toLowerCase().startsWith('ar'));
}

async function updateVoice() {
  const el = $('voice-status');
  if (flaskMode) {
    const n = readyAudio.size;
    if (current && readyAudio.has(current.id)) {
      el.textContent = `نطق فوري · «${current.name}» محفوظ في الذاكرة`;
    } else if (n) {
      el.textContent = `${n} مقطع جاهز · النطق من الذاكرة بدون ElevenLabs`;
    } else {
      el.textContent = 'ElevenLabs يُستدعى مرة عند إضافة الكلمة فقط · ثم نطق فوري';
    }
    return;
  }
  const cfg = getVoiceConfig();
  if (!cfg.apiKey) {
    el.textContent = 'ElevenLabs: أضف المفتاح من الإعدادات (أو شغّل Flask)';
    return;
  }
  el.textContent = 'وضع محلي · ElevenLabs عبر المتصفح';
}

function speakBrowser(text) {
  if (!window.speechSynthesis) return toast('النطق غير مدعوم في هذا المتصفح');
  const v = browserVoice();
  if (!v) return toast('أضف صوتًا عربيًا للجهاز أو اضبط ElevenLabs.');
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = v.lang;
  u.voice = v;
  u.rate = 0.95;
  u.onerror = () => toast('تعذّر تشغيل الصوت الاحتياطي.');
  speechSynthesis.speak(u);
}

let serverAudio = null;
/** In-memory Audio elements with blob: URLs — play is instant, no network/ElevenLabs. */
const readyAudio = new Map(); // signId -> HTMLAudioElement

function clearAudioCache() {
  for (const a of readyAudio.values()) {
    try {
      a.pause();
      if (a.src.startsWith('blob:')) URL.revokeObjectURL(a.src);
    } catch { /* ignore */ }
  }
  readyAudio.clear();
  serverAudio = null;
}

function stopSpeaking() {
  if (serverAudio) {
    try { serverAudio.pause(); serverAudio.currentTime = 0; } catch { /* ignore */ }
  }
  try { speechSynthesis?.cancel(); } catch { /* ignore */ }
}

/** Load a cached mp3 into memory. Never generates TTS. Returns true if ready. */
async function loadClipIntoMemory(signId, text) {
  const name = String(text || '').trim();
  if (!name) return false;
  try {
    const res = await fetch(api.audioUrl(signId, name));
    if (!res.ok) return false;
    const blob = await res.blob();
    if (!blob.size) return false;
    const prev = readyAudio.get(signId);
    if (prev?.src?.startsWith('blob:')) {
      try { URL.revokeObjectURL(prev.src); } catch { /* ignore */ }
    }
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.preload = 'auto';
    audio._isharaText = name;
    readyAudio.set(signId, audio);
    // Decode ahead so first play has zero buffer delay
    try { audio.load(); } catch { /* ignore */ }
    return true;
  } catch {
    return false;
  }
}

/**
 * Live speak: ONLY from in-memory Audio. Zero ElevenLabs / network wait.
 * Missing clip → browser TTS now; generate+warm in background for next time.
 */
function speak(signId, text) {
  const name = String(text || '').trim();
  stopSpeaking();

  const cached = readyAudio.get(signId);
  if (cached && cached._isharaText === name) {
    try {
      cached.currentTime = 0;
      serverAudio = cached;
      const p = cached.play();
      if (p && typeof p.catch === 'function') p.catch(() => speakBrowser(name));
      updateVoice();
      return Promise.resolve(true);
    } catch {
      /* fall through */
    }
  }

  // Do not await fetch/ensure here — keeps gesture→speech instant
  speakBrowser(name);
  (async () => {
    if (flaskMode) {
      const loaded = await loadClipIntoMemory(signId, name);
      if (!loaded) await cacheVoiceForSign({ id: signId, name });
    } else {
      await cacheVoiceForSign({ id: signId, name });
    }
  })().catch(() => {});
  return Promise.resolve(false);
}

/** Generate+cache once (add/train/rename/boot) then load into memory. */
async function cacheVoiceForSign(sign) {
  if (!sign?.name) return false;
  try {
    if (flaskMode) {
      await api.ensureVoice(sign.id, sign.name);
      const ok = await loadClipIntoMemory(sign.id, sign.name);
      updateVoice();
      return ok;
    }
    if (getVoiceConfig().apiKey) {
      const already = await hasCachedClip(sign.id, sign.name);
      if (!already) await ensureClip(sign.id, sign.name);
      updateVoice();
      return true;
    }
  } catch (e) {
    console.warn('voice cache', e);
  }
  return false;
}

async function warmAllVoices() {
  if (!flaskMode || !signs.length) return;
  try {
    // Fill any missing mp3s on the server once
    await api.warmupVoices();
  } catch (e) {
    console.warn('warmup', e);
  }
  await Promise.allSettled(
    signs.map((s) => loadClipIntoMemory(s.id, s.name)),
  );
  const n = readyAudio.size;
  const st = $('voice-status');
  if (st) st.textContent = n
    ? `أصوات جاهزة في الذاكرة: ${n} · النطق فوري بدون ElevenLabs`
    : 'لا توجد مقاطع صوت بعد — تُحفظ عند إضافة/تدريب الكلمة';
  render();
}

$('speak').onclick = () => current && speak(current.id, current.name);
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = updateVoice;

async function bootVoiceConfig() {
  try {
    const mod = await import('./local-config.js');
    const cfg = getVoiceConfig();
    if (mod.ELEVEN_API_KEY && !cfg.apiKey) {
      setVoiceConfig({
        apiKey: mod.ELEVEN_API_KEY,
        voiceId: mod.ELEVEN_VOICE_ID || DEFAULT_VOICE_ID,
      });
    } else if (mod.ELEVEN_VOICE_ID && !cfg.voiceId) {
      setVoiceConfig({ voiceId: mod.ELEVEN_VOICE_ID });
    }
  } catch {
    /* local-config.js اختياري */
  }
  const cfg = getVoiceConfig();
  if ($('eleven-key')) $('eleven-key').value = cfg.apiKey ? '••••••••' + cfg.apiKey.slice(-4) : '';
  if ($('eleven-voice')) $('eleven-voice').value = cfg.voiceId || DEFAULT_VOICE_ID;
  updateVoice();
}

$('voice-settings')?.addEventListener('click', () => {
  const cfg = getVoiceConfig();
  $('eleven-key').value = cfg.apiKey || '';
  $('eleven-voice').value = cfg.voiceId || DEFAULT_VOICE_ID;
  $('voice-dialog').showModal();
});
$('close-voice')?.addEventListener('click', () => $('voice-dialog').close());
$('save-voice')?.addEventListener('click', () => {
  const apiKey = $('eleven-key').value.trim();
  const voiceId = $('eleven-voice').value.trim() || DEFAULT_VOICE_ID;
  const prev = getVoiceConfig().apiKey;
  const masked = prev ? `••••••••${prev.slice(-4)}` : '';
  if (!apiKey) setVoiceConfig({ apiKey: '', voiceId });
  else if (apiKey === masked) setVoiceConfig({ voiceId });
  else setVoiceConfig({ apiKey, voiceId });
  $('voice-dialog').close();
  toast('تم حفظ إعدادات الصوت');
  updateVoice();
});
bootVoiceConfig();

function recognized(match) {
  current = match;
  $('result').textContent = match.name;
  $('score').textContent = match.similarity + '%';
  $('meter-fill').style.width = match.similarity + '%';
  $('speak').disabled = false;
  $('result-ring').textContent = '✓';

  const panel = $('recognition-panel');
  panel.classList.remove('hit', 'success-hit');
  void panel.offsetWidth;

  const testing = verifyTarget != null;
  const expectedOk = testing && match.id === verifyTarget;

  if (expectedOk) {
    panel.classList.add('hit', 'success-hit');
    $('result-note').textContent = 'نجح التعرّف على الإشارة التي علّمتها';
    setVerifyBanner('ok', `✓ نجح! تعرّف على «${match.name}» بنسبة ${match.similarity}%`);
    showSuccess(match);
    verifyTarget = null;
    toast(`✓ نجح التعرّف على «${match.name}»`);
    (async () => {
      try {
        if (flaskMode) {
          const res = await api.markVerified(match.id);
          signs[match.id] = fromApiSign(res.sign);
        } else {
          signs[match.id].verified = true;
          signs[match.id].verifyHits = (signs[match.id].verifyHits || 0) + 1;
          persist();
        }
        render();
      } catch { /* ignore */ }
    })();
  } else if (testing && match.id !== verifyTarget) {
    panel.classList.add('hit');
    const expected = signs[verifyTarget]?.name || '';
    $('result-note').textContent = `تعرّف على «${match.name}» بدل «${expected}» — أعد المحاولة أو سجّل أمثلة أوضح`;
    setVerifyBanner('fail', `✗ لم يطابق الاختبار. المتوقع «${expected}» · ظهر «${match.name}»`);
    toast(`لم ينجح بعد. المتوقع «${expected}». أضف مثالًا أوضح وحاول مجددًا.`);
  } else {
    panel.classList.add('hit');
    $('result-note').textContent = flaskMode
      ? '✓ تعرّف Flask بدقة عالية'
      : '✓ تم التعرّف بنجاح من أمثلتك المحفوظة';
    setVerifyBanner('ok', `✓ نجح التعرّف: «${match.name}» · ${match.similarity}%`);
    (async () => {
      try {
        if (flaskMode) {
          const res = await api.markVerified(match.id);
          signs[match.id] = fromApiSign(res.sign);
        } else {
          signs[match.id].verified = true;
          signs[match.id].verifyHits = (signs[match.id].verifyHits || 0) + 1;
          persist();
        }
        render();
      } catch { /* ignore */ }
    })();
  }

  if ($('history').querySelector('.history-empty')) $('history').replaceChildren();
  const row = document.createElement('div');
  row.className = 'history-item';
  const word = document.createElement('span');
  word.textContent = (expectedOk || !testing ? '✓ ' : '') + match.name;
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' });
  row.append(word, time);
  $('history').prepend(row);
  while ($('history').children.length > 30) $('history').lastChild.remove();

  // Instant speak from memory — never wait for ElevenLabs on the live path
  if ($('auto-speak').checked) {
    speak(match.id, match.name).catch(() => speakBrowser(match.name));
  } else {
    updateVoice();
  }
}
$('clear').onclick = () => { $('history').replaceChildren(); };

const MODEL_JS = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs';
const MODEL_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';
const MODEL_TASK =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let modelPromise = null;
let modelReady = false;

function setModelStatus(text, kind = 'loading') {
  const el = $('model-status');
  if (!el) return;
  el.textContent = text;
  el.className = `model-status ${kind}`;
}

/** Load MediaPipe in the background as soon as the page opens — don't wait for the button. */
function preloadHandModel() {
  if (model) {
    modelReady = true;
    setModelStatus('✓ متتبّع اليد جاهز — شغّل الكاميرا', 'ready');
    return Promise.resolve(model);
  }
  if (modelPromise) return modelPromise;

  setModelStatus('⏳ تحميل متتبّع اليد (أول مرة فقط)…', 'loading');
  const startedAt = performance.now();

  modelPromise = (async () => {
    try {
      setModelStatus('① تحميل مكتبة الرؤية…', 'loading');
      const { HandLandmarker, FilesetResolver } = await import(MODEL_JS);

      setModelStatus('② تجهيز محرّك WASM…', 'loading');
      const files = await FilesetResolver.forVisionTasks(MODEL_WASM);

      setModelStatus('③ تحميل نموذج المفاصل…', 'loading');
      let created;
      try {
        created = await HandLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MODEL_TASK, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      } catch {
        // GPU unavailable — fall back to CPU
        created = await HandLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MODEL_TASK, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      }

      model = created;
      modelReady = true;
      const sec = Math.max(1, Math.round((performance.now() - startedAt) / 1000));
      setModelStatus(`✓ جاهز خلال ${sec} ث — اضغط تشغيل الكاميرا`, 'ready');
      const hint = $('model-hint');
      if (hint) hint.textContent = 'النموذج محمّل في المتصفح. التشغيل التالي سيكون أسرع.';
      return model;
    } catch (e) {
      modelPromise = null;
      modelReady = false;
      setModelStatus('✗ فشل التحميل — تحقق من الإنترنت ثم حدّث الصفحة', 'error');
      throw e;
    }
  })();

  return modelPromise;
}

async function start() {
  if (running || starting) return;
  starting = true;
  const token = ++session;
  $('start').disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw Error('افتح الموقع من رابط آمن أو localhost لاستخدام الكاميرا.');
    }

    // Start camera and model in parallel when possible
    $('start').textContent = modelReady ? 'فتح الكاميرا…' : 'فتح الكاميرا وتحميل المتتبّع…';
    if (!modelReady) setModelStatus('⏳ يكمل تحميل المتتبّع…', 'loading');

    const camPromise = navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user', frameRate: { ideal: 30 } },
      audio: false,
    });
    const [acquired] = await Promise.all([camPromise, preloadHandModel()]);

    if (token !== session) {
      acquired.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = acquired;
    video.srcObject = stream;
    await video.play();
    if (token !== session) return;

    running = true;
    lastVideo = -1;
    lastDetect = 0;
    buffer = [];
    smoothHands = [];
    trails = [[], []];
    fpsAt = performance.now();
    frameCount = 0;
    $('camera-empty').hidden = true;
    $('stop').disabled = false;
    $('camera-dot').classList.add('on');
    $('camera-state').textContent = 'الكاميرا مباشرة';
    video.classList.toggle('mirror', mirror);
    canvas.classList.toggle('mirror', mirror);
    raf = requestAnimationFrame(loop);
  } catch (e) {
    stop();
    toast(
      e.name === 'NotAllowedError'
        ? 'لم يُسمح باستخدام الكاميرا. فعّل الإذن من إعدادات الموقع وحاول مجددًا.'
        : e.name === 'NotFoundError'
          ? 'لم نجد كاميرا متصلة بالجهاز.'
          : e.name === 'NotReadableError'
            ? 'الكاميرا مشغولة أو غير متاحة. أغلق التطبيقات الأخرى وحاول مجددًا.'
            : e.message || 'تعذّر بدء التتبع. تحقق من اتصال الإنترنت.',
    );
  } finally {
    starting = false;
    $('start').disabled = false;
    $('start').textContent = '◉ تشغيل الكاميرا';
  }
}

// Begin downloading the hand model immediately (don't wait for the button)
preloadHandModel().catch(() => {});

function stop() {
  session++;
  running = false;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  buffer = [];
  capture = [];
  pending = null;
  latched = null;
  current = null;
  smoothHands = [];
  trails = [[], []];
  $('result').textContent = 'بانتظار إشارتك';
  $('score').textContent = '—';
  $('meter-fill').style.width = '0%';
  $('speak').disabled = true;
  $('camera-empty').hidden = false;
  $('stop').disabled = true;
  $('camera-state').textContent = 'الكاميرا غير مفعّلة';
  $('camera-dot').classList.remove('on');
  $('hands').textContent = '○ بانتظار اليد';
  $('fps').textContent = '— FPS';
  window.speechSynthesis?.cancel();
}

$('start').onclick = start;
$('stop').onclick = stop;
$('mirror').onclick = () => {
  mirror = !mirror;
  video.classList.toggle('mirror', mirror);
  canvas.classList.toggle('mirror', mirror);
  $('mirror').setAttribute('aria-pressed', String(mirror));
};

/* ── MediaPipe hand topology ── */
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17], // palm cross braces
];
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const PALM = [0, 5, 9, 13, 17];

const HAND_STYLE = [
  { bone: '#ff5ec8', glow: '#ff5ec855', joint: '#ff9adf', tip: '#ffffff', trail: '#ff5ec8' }, // Left
  { bone: '#d4f542', glow: '#d4f54255', joint: '#f0ff9a', tip: '#ffffff', trail: '#d4f542' }, // Right
];

function smoothLandmarks(result) {
  const next = [undefined, undefined];
  if (!result.landmarks.length) {
    smoothHands = [];
    return next;
  }
  result.landmarks.forEach((hand, i) => {
    const side = result.handedness?.[i]?.[0]?.categoryName === 'Left' ? 0 : 1;
    const prev = smoothHands[side];
    const alpha = 0.45;
    next[side] = hand.map((p, j) => {
      if (!prev?.[j]) return { x: p.x, y: p.y, z: p.z };
      return {
        x: prev[j].x * (1 - alpha) + p.x * alpha,
        y: prev[j].y * (1 - alpha) + p.y * alpha,
        z: prev[j].z * (1 - alpha) + p.z * alpha,
      };
    });
  });
  smoothHands = next;
  return next;
}

function px(p, w, h) {
  return [p.x * w, p.y * h];
}

function drawHand(hand, style, trailIdx, w, h) {
  if (!hand) return;

  // Motion trail (wrist)
  const [wx, wy] = px(hand[0], w, h);
  trails[trailIdx].push({ x: wx, y: wy });
  if (trails[trailIdx].length > 18) trails[trailIdx].shift();
  if (trails[trailIdx].length > 1) {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let i = 1; i < trails[trailIdx].length; i++) {
      const a = trails[trailIdx][i - 1];
      const b = trails[trailIdx][i];
      const t = i / trails[trailIdx].length;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = style.trail;
      ctx.globalAlpha = t * 0.45;
      ctx.lineWidth = 2 + t * 4;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // Soft palm fill
  ctx.beginPath();
  PALM.forEach((idx, i) => {
    const [x, y] = px(hand[idx], w, h);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = style.glow;
  ctx.fill();

  // Glow bones (underlay)
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = style.glow;
  ctx.lineWidth = 10;
  for (const [a, b] of EDGES) {
    const [ax, ay] = px(hand[a], w, h);
    const [bx, by] = px(hand[b], w, h);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Sharp bones
  ctx.strokeStyle = style.bone;
  ctx.lineWidth = 3.2;
  for (const [a, b] of EDGES) {
    const [ax, ay] = px(hand[a], w, h);
    const [bx, by] = px(hand[b], w, h);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Joints
  hand.forEach((p, i) => {
    const [x, y] = px(p, w, h);
    const tip = FINGERTIPS.has(i);
    const r = tip ? 6.5 : i === 0 ? 5.5 : 3.8;
    // outer glow
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, Math.PI * 2);
    ctx.fillStyle = style.glow;
    ctx.fill();
    // core
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = tip ? style.tip : style.joint;
    ctx.fill();
    if (tip) {
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = style.bone;
      ctx.fill();
    }
  });

  // Bounding box
  const xs = hand.map((p) => p.x * w);
  const ys = hand.map((p) => p.y * h);
  const pad = 22;
  const x0 = Math.min(...xs) - pad;
  const y0 = Math.min(...ys) - pad;
  const bw = Math.max(...xs) - Math.min(...xs) + pad * 2;
  const bh = Math.max(...ys) - Math.min(...ys) + pad * 2;
  ctx.strokeStyle = style.bone + '99';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 5]);
  ctx.strokeRect(x0, y0, bw, bh);
  ctx.setLineDash([]);
}

function draw(result) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  const smoothed = smoothLandmarks(result);
  // Map by handedness for consistent colors
  const left = smoothed[0];
  const right = smoothed[1];

  if (!left && !right && result.landmarks.length) {
    // Fallback raw draw if smoothing missed
    result.landmarks.forEach((hand, i) => {
      const isLeft = result.handedness?.[i]?.[0]?.categoryName === 'Left';
      drawHand(hand, HAND_STYLE[isLeft ? 0 : 1], isLeft ? 0 : 1, w, h);
    });
    return;
  }

  if (left) drawHand(left, HAND_STYLE[0], 0, w, h);
  if (right) drawHand(right, HAND_STYLE[1], 1, w, h);

  if (!left) trails[0] = [];
  if (!right) trails[1] = [];
}

function loop(now) {
  if (!running) return;
  try {
    // During recording sample by clock (not video.currentTime) — some cams barely advance currentTime.
    const gap = recording ? 20 : 33;
    if (video.readyState >= 2 && now - lastDetect >= gap) {
      lastDetect = now;
      lastVideo = video.currentTime;
      const result = model.detectForVideo(video, now);
      draw(result);
      frameCount++;
      if (now - fpsAt > 1000) {
        $('fps').textContent = Math.round((frameCount * 1000) / (now - fpsAt)) + ' FPS';
        fpsAt = now;
        frameCount = 0;
      }

      const visible = result.landmarks.length > 0;
      $('hands').textContent = visible
        ? `● ${result.landmarks.length === 2 ? 'اليدان ظاهرتان' : 'يد واحدة ظاهرة'} · تتبع دقيق`
        : '○ بانتظار اليد';

      if (recording && !$('coach').hidden) {
        const handEl = $('coach-hand');
        handEl.textContent = visible
          ? `✓ اليد ظاهرة · إطارات ${capture.length}`
          : '⚠ لا تظهر اليد — ارفعها داخل الإطار';
        handEl.className = visible ? 'ok' : 'bad';
      }

      if (!visible) {
        if (!noHandAt) noHandAt = now;
        if (now - noHandAt > 400) {
          buffer = [];
          pending = null;
          latched = null;
          gestureArmed = false;
          motionFrames = 0;
          motionSum = 0;
          livePeaks = 0;
          peakActive = false;
          lastFeatFrame = null;
          smoothHands = [];
          trails = [[], []];
        }
      } else {
        noHandAt = 0;
      }

      // While recording: keep every detect. Live mode: ~20 fps features.
      const sampleEvery = recording ? 0 : 40;
      if (now - lastFeature >= sampleEvery) {
        lastFeature = now;
        const f = features(result);
        if (recording) capture.push({ f, visible });
        if (visible) {
          let motion = 0;
          if (lastFeatFrame) {
            let sum = 0;
            const step = 3;
            for (let i = 0; i < f.length; i += step) {
              const d = f[i] - lastFeatFrame[i];
              sum += d * d;
            }
            motion = Math.sqrt(sum / Math.ceil(f.length / step));
          }
          lastFeatFrame = f;

          buffer.push({ t: now, f, motion });
          buffer = buffer.filter((x) => now - x.t < LIVE_BUFFER_MS);

          if (!recording) {
            const moving = motion >= MOTION_THRESH;
            if (moving) {
              if (!gestureArmed) {
                gestureArmed = true;
                gestureStartAt = now;
                motionFrames = 0;
                motionSum = 0;
                livePeaks = 0;
                peakActive = false;
                if (!verifyTarget) {
                  $('result-note').textContent = 'التقاط… الجزء 1 ثم الجزء 2';
                  $('result-ring').textContent = '…';
                }
              }
              if (!peakActive) {
                livePeaks += 1;
                peakActive = true;
              }
              motionFrames += 1;
              motionSum += motion;
              lastMotionAt = now;
            } else if (peakActive && motion < MOTION_THRESH * 0.45) {
              peakActive = false; // valley between beats
            }

            const gestureMs = gestureArmed ? (now - gestureStartAt) : 0;

            if (gestureArmed && gestureMs >= 900 && motionFrames < 3 && (now - lastMotionAt) > 400) {
              gestureArmed = false;
              motionFrames = 0;
              motionSum = 0;
              livePeaks = 0;
              peakActive = false;
              if (!verifyTarget && !current) {
                $('result-note').textContent = 'بانتظار إشارة واضحة — حركتان ثم توقف';
                $('result-ring').textContent = '○';
              }
            }

            if (gestureArmed && gestureMs > MAX_GESTURE_MS) {
              gestureArmed = false;
              motionFrames = 0;
              motionSum = 0;
              livePeaks = 0;
              peakActive = false;
            }

            const strongEnough = motionFrames >= MIN_MOTION_FRAMES && motionSum >= MIN_MOTION_SUM;
            const settleNeed = livePeaks >= 2 ? GESTURE_SETTLE_MS : GESTURE_SETTLE_1BEAT_MS;
            const settled = gestureArmed
              && strongEnough
              && livePeaks >= 1
              && (now - lastMotionAt) >= settleNeed
              && gestureMs >= MIN_GESTURE_MS
              && buffer.length >= 12;

            if (settled && !recognizing && now - lastRecognizeAt > 350) {
              lastRecognizeAt = now;
              const clip = buffer.filter((x) => x.t >= gestureStartAt - 60);
              const frames = (clip.length >= 12 ? clip : buffer).map((x) => x.f);
              const step = Math.max(1, Math.ceil(frames.length / 72));
              const payload = frames.filter((_, i) => i % step === 0);

              const usedMotionFrames = motionFrames;
              const usedMotionSum = motionSum;
              const usedPeaks = livePeaks;
              gestureArmed = false;
              motionFrames = 0;
              motionSum = 0;
              livePeaks = 0;
              peakActive = false;

              const applyResult = (match, peek, reason) => {
                // Never promote peeks — idle standing often gets a soft "peek"
                if (!match) {
                  if (reason === 'idle') {
                    if (!verifyTarget) {
                      $('result-note').textContent = 'ما في إشارة واضحة — أنت واقف أو الحركة ضعيفة';
                      $('result-ring').textContent = '○';
                      $('score').textContent = '—';
                      $('meter-fill').style.width = '0%';
                    }
                    return;
                  }
                  if (peek && peek.similarity >= 55) {
                    $('result').textContent = peek.name;
                    $('score').textContent = peek.similarity + '%';
                    $('meter-fill').style.width = Math.max(8, peek.similarity) + '%';
                    $('result-note').textContent = `غير مؤكد («${peek.name}» ${peek.similarity}%) — أعد الإشارة أوضح`;
                    $('result-ring').textContent = '…';
                  } else if (!verifyTarget) {
                    $('result-note').textContent = 'لم يُعتمد — نفّذ الإشارة بوضوح ثم توقف لحظة';
                    $('result-ring').textContent = '○';
                  }
                  return;
                }

                if ((match.similarity || 0) < MIN_MATCH_SCORE) {
                  $('result-note').textContent = `ثقة منخفضة (${match.similarity}%) — لم يُنطق. أعد الحركة أوضح`;
                  $('result-ring').textContent = '…';
                  $('score').textContent = match.similarity + '%';
                  return;
                }

                const nowMs = performance.now();
                if (latched === match.id && (nowMs - latchedAt) < SAME_SIGN_COOLDOWN_MS) return;

                $('result').textContent = match.name;
                $('score').textContent = match.similarity + '%';
                $('meter-fill').style.width = match.similarity + '%';
                recognized(match);
                latched = match.id;
                latchedAt = nowMs;
                buffer = buffer.filter((x) => nowMs - x.t < 500);
                lastFeatFrame = null;
              };

              if (flaskMode) {
                recognizing = true;
                $('result-note').textContent = `تحليل سريع… ${usedPeaks} جزء · ثقة عالية`;
                api.recognize(payload).then((res) => {
                  recognizing = false;
                  applyResult(res.match, res.peek, res.reason);
                }).catch((err) => {
                  recognizing = false;
                  console.warn('recognize failed', err);
                });
              } else {
                const windows = buildWindows(buffer);
                const match = classify(windows, signs);
                const peek = peekMatch(windows, signs);
                applyResult(match, peek);
              }
            } else if (gestureArmed && strongEnough && gestureMs >= 500 && now - softPeekAt > 280) {
              softPeekAt = now;
              if (!verifyTarget) {
                const partLabel = livePeaks >= 2 ? '✓ جزآن' : (livePeaks === 1 ? 'الجزء 1… أنهِ الجزء 2' : 'ابدأ الحركة');
                $('result-note').textContent = `${partLabel} · ${(gestureMs / 1000).toFixed(1)} ث`;
              }
            } else if (!gestureArmed && !recognizing && !verifyTarget && now - softPeekAt > 2000) {
              softPeekAt = now;
              if (!current) {
                $('result-note').textContent = 'بانتظار إشارة — حركتان باليد ثم توقف للنطق';
              }
            }
          }
        }
      }
    }
    raf = requestAnimationFrame(loop);
  } catch {
    stop();
    toast('توقف التتبع بسبب خطأ. أعد تشغيل الكاميرا أو جرّب متصفح Chrome أو Edge.');
  }
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function ensureCamera() {
  if (running) return true;
  toast('جارٍ تشغيل الكاميرا للتدريب…');
  await start();
  return running;
}

function showTrainResult(ok, title, note) {
  $('result').textContent = title;
  $('result-note').textContent = note;
  $('result-ring').textContent = ok ? '✓' : '!';
  const panel = $('recognition-panel');
  panel.classList.remove('hit', 'success-hit');
  void panel.offsetWidth;
  if (ok) panel.classList.add('hit', 'success-hit');
  else panel.classList.add('hit');
}

$('record').onclick = async () => {
  const id = selected;
  const word = signs[id].name;
  $('trainer').close();
  view('live');

  if (!(await ensureCamera())) {
    showTrainResult(false, 'فشل الكاميرا', 'اسمح بالكاميرا ثم أعد التسجيل من المكتبة');
    toast('تعذّر تشغيل الكاميرا. اسمح بالوصول ثم حاول مجددًا.');
    openTrainer(id);
    return;
  }

  // Give the detect loop a moment after camera start
  await wait(350);

  const token = session;
  recording = true;
  capture = [];
  current = null;
  latched = null;
  $('record').disabled = true;
  $('delete-sign').disabled = true;
  $('close-trainer').disabled = true;
  $('sign-name').disabled = true;
  $('recording').hidden = false;

  const takeNum = signs[id].samples.length + 1;
  try {
    setCoach({
      title: `تدريب «${word}» · مثال ${takeNum}/${MIN_SAMPLES}`,
      step: 'ارفع يدك الآن',
      hand: '○ نبحث عن اليد…',
      handOk: false,
      hint: 'لازم يظهر الهيكل على يدك قبل العدّ',
      progress: 5,
    });
    $('recording').textContent = '✋';
    showTrainResult(false, `تسجيل «${word}»`, 'ارفع يدك حتى يبدأ العد التنازلي');

    let handReady = false;
    const readyDeadline = performance.now() + 20000;
    while (performance.now() < readyDeadline) {
      if (!running || session !== token) throw Error('تم إيقاف الكاميرا أثناء التدريب.');
      const liveVisible = $('hands').textContent.includes('ظاهرة');
      if (liveVisible) {
        handReady = true;
        break;
      }
      setCoach({
        title: `تدريب «${word}» · مثال ${takeNum}`,
        step: 'ارفع يدك داخل الإطار',
        hand: '⚠ اليد غير ظاهرة بعد',
        handOk: false,
        hint: 'قرّب اليد + إضاءة أفضل — بانتظار الهيكل',
        progress: 8,
      });
      await wait(100);
    }
    if (!handReady) throw Error('ما ظهرت اليد خلال 20 ثانية. فعّل الإضاءة وقرّب اليد ثم أعد المحاولة.');

    for (let i = 3; i > 0; i--) {
      $('recording').textContent = String(i);
      setCoach({
        title: `تدريب «${word}» · مثال ${takeNum}`,
        step: `استعد… ${i}`,
        hand: '✓ اليد ظاهرة',
        handOk: true,
        hint: 'بعد العد: حركة 1 → توقف قصير → حركة 2',
        progress: 15 + (3 - i) * 10,
      });
      showTrainResult(false, `${i}`, `حركتان واضحتان لـ «${word}»`);
      await wait(650);
      if (!running || session !== token) throw Error('تم إلغاء التسجيل بعد إيقاف الكاميرا.');
    }

    capture = [];
    const captureMs = 4800;
    const minCaptureMs = 2200;
    const started = performance.now();
    let trainPeaks = 0;
    let trainPeakActive = false;
    let lastTrainFeat = null;
    let lastTrainMotionAt = started;
    $('recording').textContent = '①';

    while (true) {
      if (!running || session !== token) throw Error('تم إلغاء التسجيل.');
      const elapsed = performance.now() - started;
      if (elapsed >= captureMs) break;

      // Count motion beats live from captured frames
      if (capture.length >= 2) {
        const a = capture[capture.length - 2].f;
        const b = capture[capture.length - 1].f;
        let sum = 0;
        for (let i = 0; i < b.length; i += 3) {
          const d = b[i] - a[i];
          sum += d * d;
        }
        const m = Math.sqrt(sum / Math.ceil(b.length / 3));
        if (m >= 0.1) {
          if (!trainPeakActive) {
            trainPeaks += 1;
            trainPeakActive = true;
          }
          lastTrainMotionAt = performance.now();
        } else if (trainPeakActive && m < 0.045) {
          trainPeakActive = false;
        }
        lastTrainFeat = b;
      }

      // Smart early stop: both beats captured + short settle
      if (
        elapsed >= minCaptureMs
        && trainPeaks >= 2
        && (performance.now() - lastTrainMotionAt) >= 480
      ) {
        break;
      }

      const liveVisible = $('hands').textContent.includes('ظاهرة');
      const secLeft = Math.max(0, ((captureMs - elapsed) / 1000)).toFixed(1);
      let stepMsg = '① نفّذ الحركة الأولى';
      let hintMsg = 'بعدها توقف لحظة ثم الحركة الثانية';
      $('recording').textContent = '①';
      if (trainPeaks === 1) {
        stepMsg = '①✓ → ② نفّذ الحركة الثانية الآن';
        hintMsg = 'أكمل الجزء الثاني بوضوح';
        $('recording').textContent = '②';
      } else if (trainPeaks >= 2) {
        stepMsg = '①✓ ②✓ — ثبّت لحظة';
        hintMsg = 'تم التقاط الجزئين';
        $('recording').textContent = '✓';
      }
      setCoach({
        title: `تدريب «${word}» · مثال ${takeNum}`,
        step: stepMsg,
        hand: liveVisible ? `✓ يد · ${capture.length} إطار · أجزاء ${Math.min(2, trainPeaks)}/2` : '⚠ أرجع اليد',
        handOk: liveVisible,
        hint: liveVisible ? `${hintMsg} (${secLeft} ث)` : 'اليد اختفت — أرجعها',
        progress: 45 + Math.min(50, (elapsed / captureMs) * 50),
      });
      showTrainResult(false, trainPeaks >= 2 ? 'جزآن ✓' : `أجزاء ${trainPeaks}/2`, `تسجيل ذكي لـ «${word}»`);
      await wait(35);
    }

    const frames = capture.filter((x) => x.visible).map((x) => x.f);
    const ratio = capture.length ? frames.length / capture.length : 0;

    if (capture.length < 3 && frames.length < 3) {
      throw Error(`ما انحفظت إطارات كافية (${capture.length}). انتظر ✓ جاهز للمتتبّع، وخلّ اليد ظاهرة.`);
    }
    if (frames.length < 8) {
      throw Error(`الإشارة قصيرة جدًا (${frames.length} إطار). نفّذ الحركتين كاملتين.`);
    }
    if (capture.length >= 8 && ratio < 0.2) {
      throw Error('اليد اختفت معظم الوقت. أعد التسجيل وخلّها داخل الإطار.');
    }
    if (trainPeaks < 1) {
      throw Error('ما انرصدت حركة واضحة. أعد التسجيل: حركة 1 ثم حركة 2 بوضوح.');
    }

    let q = Math.max(qualityScore(frames), 35);
    let count;
    let ready;
    let peaksSaved = trainPeaks;

    if (flaskMode) {
      const res = await api.uploadSample(id, frames);
      signs[id] = fromApiSign(res.sign);
      q = res.quality || q;
      count = res.count;
      ready = res.ready;
      peaksSaved = res.peaks || trainPeaks;
      if (res.voiceError) toast('حُفظت الإشارة؛ الصوت لاحقًا: ' + res.voiceError);
    } else {
      signs[id].samples.push(resample(frames));
      signs[id].verified = false;
      persist();
      count = signs[id].samples.length;
      ready = count >= MIN_SAMPLES;
    }

    buffer = [];
    pending = null;
    latched = null;
    current = null;
    render();

    const peaksNote = peaksSaved >= 2 ? 'جزآن ✓' : 'جزء واحد — إن كانت حركتين أعد التسجيل';
    $('recording').textContent = '✓';
    setCoach({
      title: `تدريب «${word}»`,
      step: `✓ تم حفظ المثال ${count}`,
      hand: `جودة ${q}% · ${peaksNote} · ${frames.length} إطار`,
      handOk: true,
      hint: ready ? 'الآن نفّذ الإشارة (جزآن) للتعرّف السريع' : `سجّل مثال ${count + 1} بنفس الأسلوب`,
      progress: 100,
    });
    showTrainResult(
      true,
      `✓ حُفظ: ${word}`,
      ready
        ? `أمثلة ${count} · ${peaksNote} — نفّذ الإشارة الآن`
        : `مثال ${count}/${MIN_SAMPLES} · ${peaksNote}`,
    );
    $('score').textContent = q + '%';
    $('meter-fill').style.width = q + '%';

    if (peaksSaved < 2) {
      toast(`حُفظ «${word}» لكن رُصد جزء واحد فقط — للإشارة من حركتين أعد التسجيل.`);
    } else {
      toast(
        ready
          ? `✓ «${word}» جاهزة (جزآن). نفّذها الآن للنطق السريع.`
          : `✓ مثال ${count} بجزئين. بقي ${MIN_SAMPLES - count}.`,
      );
    }

    await wait(1100);

    if (ready) {
      // Ensure mp3 exists + in memory before live testing
      cacheVoiceForSign(signs[id]).then((ok) => {
        if (ok) toast(`صوت «${word}» جاهز في الذاكرة`);
      });
      setVerifyBanner('ok', `✓ «${word}» جاهزة — نفّذ الإشارة الآن`);
      startVerify(id);
    } else {
      openTrainer(id);
    }
  } catch (e) {
    $('recording').textContent = '✗';
    const msg = e.message || 'فشل التسجيل';
    setCoach({
      title: `تدريب «${word}»`,
      step: '✗ لم يُحفظ',
      hand: 'حاول مرة ثانية',
      handOk: false,
      hint: msg,
      progress: 0,
    });
    showTrainResult(false, 'لم يُحفظ المثال', msg);
    toast(msg);
    await wait(1600);
    openTrainer(id);
  } finally {
    recording = false;
    capture = [];
    $('recording').hidden = true;
    hideCoach();
    $('delete-sign').disabled = false;
    $('close-trainer').disabled = false;
    $('sign-name').disabled = false;
    updateTakes();
  }
};

$('export').onclick = async () => {
  try {
    const data = flaskMode ? await api.exportTraining() : { version: DATA_VERSION, signs };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ishara-training.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    toast(e.message);
  }
};

$('import').onclick = () => $('import-file').click();
$('import-file').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    if (file.size > 8e6) throw Error('الملف أكبر من الحجم المسموح');
    const payload = JSON.parse(await file.text());
    if (!confirm('استبدال التدريب الموجود ببيانات هذا الملف؟')) return;
    if (flaskMode) {
      const res = await api.importTraining(payload);
      signs = res.signs.map(fromApiSign);
    } else {
      signs = validateData(payload);
      persist();
    }
    render();
    buffer = [];
    latched = null;
    toast('تم استيراد التدريب.');
  } catch (err) {
    toast('تعذّر الاستيراد: ' + err.message);
  } finally {
    e.target.value = '';
  }
};

window.addEventListener('pagehide', stop);
document.addEventListener('visibilitychange', () => {
  if (document.hidden && (running || starting)) stop();
});

bootBackend();
