import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  WORDS, features, resample, classify, validateData, distance, motionDistance,
  SEQ_LEN, FEAT_DIM, DATA_VERSION, createDefaultSigns, qualityScore, selfConsistency,
  trimIdle, buildWindows, MIN_SAMPLES,
} from '../dist/core.mjs';

const frame = (x) => Array(FEAT_DIM).fill(x);
const sequence = (x) => Array.from({ length: SEQ_LEN }, (_, i) => frame(x + i * 0.02));

test('unknown movements and insufficient training are rejected', () => {
  assert.equal(classify([sequence(4)], [{ id: 0, name: 'ماء', samples: [sequence(0), sequence(0), sequence(0)] }]), null);
  assert.equal(classify([sequence(0)], [{ id: 0, name: 'ماء', samples: [sequence(0)] }]), null);
});

test('trained sequence matches; ambiguous words are rejected', () => {
  const a = { id: 0, name: 'ماء', samples: Array(MIN_SAMPLES).fill(sequence(0)) };
  assert.equal(classify([sequence(0)], [a]).name, 'ماء');
  assert.equal(classify([sequence(0)], [a, { ...a, id: 1, name: 'طعام' }]), null);
});

test('temporal direction matters for pose and motion distance', () => {
  const a = sequence(0);
  assert.equal(distance(a, a), 0);
  assert.ok(distance(a, [...a].reverse()) > 0.05);
  assert.ok(motionDistance(a, [...a].reverse()) > motionDistance(a, a));
});

test('handedness ordering and missing hands have fixed dimensions', () => {
  const hand = Array.from({ length: 21 }, (_, i) => ({ x: 0.1 + i * 0.01, y: 0.2 + i * 0.01, z: 0 }));
  const f = features({ landmarks: [hand], handedness: [[{ categoryName: 'Right' }]] });
  assert.equal(f.length, FEAT_DIM);
  assert.equal(f[0], 0);
  assert.equal(f[76], 2);
  assert.ok(f.every(Number.isFinite));
});

test('imports validate schema, labels, sample lengths and finite coordinates', () => {
  const signs = createDefaultSigns();
  assert.equal(validateData({ version: DATA_VERSION, signs }).length, WORDS.length);
  assert.throws(() => validateData({ version: 1, signs }));
  signs[0].samples = [Array(SEQ_LEN).fill(frame(NaN))];
  assert.throws(() => validateData({ version: DATA_VERSION, signs }));
  assert.equal(resample([frame(0), frame(1)]).length, SEQ_LEN);
  assert.equal(resample([frame(0), frame(1)])[0].length, FEAT_DIM);
});

test('quality score rewards visible motion frames', () => {
  const good = sequence(0).map((f) => {
    const g = f.slice();
    g[0] = 2;
    return g;
  });
  assert.ok(qualityScore(good) > qualityScore(sequence(0)));
});

test('selfConsistency accepts matching takes of the same sign', () => {
  const sign = { id: 0, name: 'لكمة', samples: [sequence(0), sequence(0.01), sequence(0.02)] };
  const r = selfConsistency(sign);
  assert.equal(r.ok, true);
  assert.ok(r.similarity > 50);
});

test('trimIdle removes flat ends and buildWindows returns candidates', () => {
  const idle = Array(6).fill(frame(0));
  const move = Array.from({ length: 8 }, (_, i) => frame(i * 0.08));
  const seq = [...idle, ...move, ...idle];
  const trimmed = trimIdle(seq);
  assert.ok(trimmed.length < seq.length);
  assert.ok(trimmed.length >= 4);
  const now = 1000;
  const buf = seq.map((f, i) => ({ t: now - (seq.length - i) * 40, f }));
  assert.ok(buildWindows(buf).length >= 2);
});
