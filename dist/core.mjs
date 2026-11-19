/** Ishara core: hand features + high-precision gesture matching. */
export const WORDS = [
  'مرحبًا', 'شكرًا', 'نعم', 'لا', 'من فضلك', 'آسف', 'أنا', 'أنت', 'مساعدة', 'ماء',
  'طعام', 'بيت', 'مدرسة', 'أهلًا وسهلًا', 'كيف حالك', 'بخير', 'أحبك', 'انتظر', 'أين', 'مع السلامة',
];
export const SEQ_LEN = 16;
export const FEAT_DIM = 152;
export const MAX_SAMPLES = 12;
export const MIN_SAMPLES = 2;
export const MAX_SIGNS = 48;
export const DATA_VERSION = 2;

const TIP = [4, 8, 12, 16, 20];
const PIP = [3, 6, 10, 14, 18];
const MCP = [2, 5, 9, 13, 17];

function angle(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y, abz = a.z - b.z;
  const cbx = c.x - b.x, cby = c.y - b.y, cbz = c.z - b.z;
  const den = Math.hypot(abx, aby, abz) * Math.hypot(cbx, cby, cbz);
  if (den < 1e-6) return 0;
  return Math.acos(Math.min(1, Math.max(-1, (abx * cbx + aby * cby + abz * cbz) / den))) / Math.PI;
}

/** Compact, scale/translation-invariant hand descriptor (76 dims × 2 hands = 152). */
export function features(result) {
  const out = [];
  for (const side of ['Left', 'Right']) {
    const i = (result.handedness ?? []).findIndex((h) => h[0]?.categoryName === side);
    const hand = result.landmarks[i];
    if (!hand) {
      out.push(...Array(76).fill(0));
      continue;
    }
    const wrist = hand[0];
    const mid = hand[9];
    const scale = Math.max(0.035, Math.hypot(mid.x - wrist.x, mid.y - wrist.y, mid.z - wrist.z));
    out.push(2);
    for (const p of hand) {
      out.push((p.x - wrist.x) / scale, (p.y - wrist.y) / scale, (p.z - wrist.z) / scale);
    }
    out.push(wrist.x * 2, wrist.y * 2);
    for (let f = 0; f < 5; f++) {
      out.push(angle(hand[MCP[f]], hand[PIP[f]], hand[TIP[f]]));
      out.push(Math.hypot(hand[TIP[f]].x - wrist.x, hand[TIP[f]].y - wrist.y, hand[TIP[f]].z - wrist.z) / scale);
    }
  }
  return out;
}

export function resample(frames, n = SEQ_LEN) {
  if (!frames.length) return Array.from({ length: n }, () => Array(FEAT_DIM).fill(0));
  if (frames.length === 1) return Array.from({ length: n }, () => frames[0].slice());
  return Array.from({ length: n }, (_, i) => {
    const t = (i * (frames.length - 1)) / (n - 1);
    const a = Math.floor(t);
    const b = Math.min(frames.length - 1, a + 1);
    const u = t - a;
    return frames[a].map((v, j) => v * (1 - u) + frames[b][j] * u);
  });
}

function buildWeights() {
  const w = Array(FEAT_DIM).fill(1);
  for (const hand of [0, 1]) {
    const base = hand * 76;
    w[base] = 0.35;
    w[base + 64] = 1.6;
    w[base + 65] = 1.6;
    for (const tip of TIP) {
      const o = base + 1 + tip * 3;
      w[o] = 2.4;
      w[o + 1] = 2.4;
      w[o + 2] = 1.5;
    }
    for (let k = 0; k < 10; k++) w[base + 66 + k] = 1.9;
  }
  return w;
}
const WEIGHTS = buildWeights();

function frameDistance(a, b) {
  let sum = 0;
  let wsum = 0;
  for (let i = 0; i < a.length; i++) {
    const w = WEIGHTS[i] ?? 1;
    const d = a[i] - b[i];
    sum += w * d * d;
    wsum += w;
  }
  return Math.sqrt(sum / wsum);
}

export function velocities(seq) {
  const out = [];
  for (let i = 0; i < seq.length; i++) {
    if (i === 0) out.push(Array(seq[0].length).fill(0));
    else out.push(seq[i].map((v, j) => v - seq[i - 1][j]));
  }
  return out;
}

export function accelerations(seq) {
  return velocities(velocities(seq));
}

/**
 * Trim idle frames at the ends so live recognition matches training clips
 * (training is mostly the gesture; live buffer often includes stillness).
 */
export function trimIdle(frames, thresh = 0.035) {
  if (!frames || frames.length < 5) return frames || [];
  const motion = [0];
  for (let i = 1; i < frames.length; i++) motion.push(frameDistance(frames[i], frames[i - 1]));
  let start = 0;
  let end = frames.length - 1;
  while (start < end - 2 && motion[start] < thresh) start++;
  while (end > start + 2 && motion[end] < thresh) end--;
  start = Math.max(0, start - 1);
  end = Math.min(frames.length - 1, end + 1);
  const cut = frames.slice(start, end + 1);
  return cut.length >= 4 ? cut : frames;
}

/** Build several candidate windows from a live buffer for robust matching. */
export function buildWindows(bufferEntries) {
  if (!bufferEntries?.length) return [];
  const now = bufferEntries[bufferEntries.length - 1].t;
  const spans = [800, 1200, 1800, 2600];
  const out = [];
  for (const ms of spans) {
    const slice = bufferEntries.filter((x) => now - x.t <= ms).map((x) => x.f);
    if (slice.length < 8) continue;
    out.push(resample(trimIdle(slice)));
    out.push(resample(slice));
  }
  // Full buffer trimmed once
  const all = bufferEntries.map((x) => x.f);
  if (all.length >= 8) out.push(resample(trimIdle(all)));
  return out;
}

export function distance(a, b) {
  const n = a.length;
  const m = b.length;
  const band = Math.max(2, Math.ceil(Math.max(n, m) * 0.4));
  const d = Array.from({ length: n + 1 }, () => Array(m + 1).fill(Infinity));
  d[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    const j0 = Math.max(1, i - band);
    const j1 = Math.min(m, i + band);
    for (let j = j0; j <= j1; j++) {
      d[i][j] = frameDistance(a[i - 1], b[j - 1]) + Math.min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]);
    }
  }
  return d[n][m] / Math.max(n, m);
}

export function motionDistance(a, b) {
  const pose = distance(a, b);
  const vel = distance(velocities(a), velocities(b));
  const acc = distance(accelerations(a), accelerations(b));
  return pose * 0.58 + vel * 0.28 + acc * 0.14;
}

/** Best (minimum) distance across samples × windows — sharper recognition. */
function signScore(windows, samples) {
  let best = Infinity;
  for (const sample of samples) {
    const trimmedSample = trimIdle(sample);
    const sampleVariants = [sample, resample(trimmedSample)];
    for (const variant of sampleVariants) {
      for (const w of windows) {
        const d = motionDistance(w, variant);
        if (d < best) best = d;
      }
    }
  }
  return best;
}

function rankSigns(windows, signs) {
  const ready = signs.filter((s) => s.samples.length >= MIN_SAMPLES);
  if (!ready.length || !windows?.length) return [];
  return ready
    .map((s) => {
      const d = signScore(windows, s.samples);
      const similarity = Math.round(Math.max(0, Math.min(99, (1 - d / 0.55) * 100)));
      return { id: s.id, name: s.name, d, n: s.samples.length, similarity };
    })
    .sort((a, b) => a.d - b.d);
}

/** Always returns the nearest trained sign (for live preview), even if weak. */
export function peekMatch(windows, signs) {
  const ranked = rankSigns(windows, signs);
  return ranked[0] || null;
}

/**
 * Classify with adaptive thresholds:
 * - few trained signs → easier accept (so training feels like it works)
 * - many signs → require clearer margin to avoid mix-ups
 */
export function classify(windows, signs, threshold = 0.22) {
  const ranked = rankSigns(windows, signs);
  if (!ranked.length) return null;

  const best = ranked[0];
  const second = ranked[1];
  const margin = second ? second.d - best.d : Infinity;
  const readyCount = ranked.length;

  let minMargin = 0;
  if (readyCount === 2) minMargin = 0.006;
  else if (readyCount > 2 && readyCount <= 8) minMargin = 0.01;
  else if (readyCount > 8) minMargin = 0.015;

  const maxDist = threshold + (readyCount === 1 ? 0.04 : 0) - Math.min(0.02, Math.max(0, best.n - MIN_SAMPLES) * 0.002);
  if (best.d > maxDist || margin < minMargin) return null;

  return { ...best, margin: +margin.toFixed(4) };
}

export function selfConsistency(sign) {
  if (!sign?.samples || sign.samples.length < MIN_SAMPLES) return null;
  const query = sign.samples[sign.samples.length - 1];
  const prior = sign.samples.slice(0, -1);
  const d = signScore([query], prior);
  const ok = d <= 0.2;
  const similarity = Math.round(Math.max(0, Math.min(99, (1 - d / 0.55) * 100)));
  return { ok, d, similarity };
}

export function validateData(data) {
  if (data?.version !== DATA_VERSION || !Array.isArray(data.signs)) throw Error('ملف التدريب غير صالح');
  if (data.signs.length < 1 || data.signs.length > MAX_SIGNS) throw Error('عدد الإشارات غير صالح');
  return data.signs.map((s, i) => {
    if (s.id !== i || typeof s.name !== 'string' || !s.name.trim() || s.name.length > 40) {
      throw Error('بيانات الإشارات غير صالحة');
    }
    if (!Array.isArray(s.samples) || s.samples.length > MAX_SAMPLES) throw Error('بيانات الإشارات غير صالحة');
    for (const sample of s.samples) {
      if (
        !Array.isArray(sample) ||
        sample.length !== SEQ_LEN ||
        sample.some(
          (f) =>
            !Array.isArray(f) ||
            f.length !== FEAT_DIM ||
            f.some((v) => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 100),
        )
      ) {
        throw Error('بيانات الحركة غير صالحة');
      }
    }
    return {
      id: i,
      name: s.name.trim(),
      samples: s.samples,
      verified: !!s.verified,
      verifyHits: Math.max(0, Math.min(99, Number(s.verifyHits) || 0)),
    };
  });
}

export function createDefaultSigns() {
  return WORDS.map((name, id) => ({ id, name, samples: [], verified: false, verifyHits: 0 }));
}

export function qualityScore(frames) {
  if (!frames.length) return 0;
  const present = frames.filter((f) => f[0] > 0 || f[76] > 0);
  const ratio = present.length / frames.length;
  let motion = 0;
  for (let i = 1; i < present.length; i++) {
    motion += frameDistance(present[i], present[i - 1]);
  }
  motion /= Math.max(1, present.length - 1);
  return Math.round(Math.min(100, ratio * 75 + Math.min(25, motion * 35)));
}
