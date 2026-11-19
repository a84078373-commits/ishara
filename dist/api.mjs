/** Flask API client for Ishara backend. */
const BASE = '';

async function req(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok || data?.ok === false) {
    throw Error(data?.error || `HTTP ${res.status}`);
  }
  return data;
}

export async function health() {
  return req('/api/health');
}

export async function fetchSigns() {
  return req('/api/signs');
}

export async function createSign(name) {
  return req('/api/signs', { method: 'POST', body: JSON.stringify({ name }) });
}

export async function renameSign(id, name) {
  return req(`/api/signs/${id}`, { method: 'PATCH', body: JSON.stringify({ name }) });
}

export async function clearSamples(id) {
  return req(`/api/signs/${id}/samples`, { method: 'DELETE' });
}

export async function uploadSample(id, frames) {
  return req(`/api/signs/${id}/samples`, {
    method: 'POST',
    body: JSON.stringify({ frames }),
  });
}

export async function markVerified(id) {
  return req(`/api/signs/${id}/verify`, { method: 'POST', body: '{}' });
}

export async function recognize(frames) {
  return req('/api/recognize', {
    method: 'POST',
    body: JSON.stringify({ frames }),
  });
}

export async function ensureVoice(id, text) {
  return req(`/api/tts/${id}`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

export function audioUrl(id, text) {
  return `/api/audio/${id}?t=${encodeURIComponent(text)}&v=1`;
}

export async function exportTraining() {
  return req('/api/export');
}

export async function importTraining(payload) {
  return req('/api/import', { method: 'POST', body: JSON.stringify(payload) });
}
