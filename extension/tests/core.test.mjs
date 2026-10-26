// Unit tests for the DOM-free core logic:  node --test tests/
import assert from "node:assert/strict";
import test from "node:test";
import { BackgroundModel, SceneChangeMonitor } from "../src/core/background.js";
import {
  MaskRefiner, addHand, backgroundMask, blur, differenceMask, differenceMaskHalf, dilate, exposureGain, upsampleGain, erode, fillDown, pullPushFill, softKnown,
} from "../src/core/masks.js";
import { PalmToggle } from "../src/core/toggle.js";

const DT = 1 / 20;
function feed(tg, visible, start, duration) {
  let n = 0, t = start;
  while (t < start + duration - 1e-9) { n += tg.update(visible, t) ? 1 : 0; t += DT; }
  return [n, t];
}

// ---- PalmToggle (same cases as the Python tests) ---------------------------------
test("toggle: short flash does not toggle", () => {
  const tg = new PalmToggle();
  const [n, t] = feed(tg, true, 0, 0.3);
  const [n2] = feed(tg, false, t, 1);
  assert.equal(n + n2, 0);
});
test("toggle: hold toggles once, release required before next", () => {
  const tg = new PalmToggle();
  let [n1, t] = feed(tg, true, 0, 5);
  assert.equal(n1, 1);
  let n2; [n2, t] = feed(tg, false, t, 2);
  let n3; [n3, t] = feed(tg, true, t, 1);
  assert.deepEqual([n2, n3], [0, 1]);
  assert.equal(tg.active, false);
});
test("toggle: cooldown blocks quick retoggle", () => {
  const tg = new PalmToggle();
  let [a, t] = feed(tg, true, 0, 0.6);
  let b; [b, t] = feed(tg, false, t, 0.4);
  let c; [c, t] = feed(tg, true, t, 0.6);
  assert.deepEqual([a, b, c], [1, 0, 0]);
});
test("toggle: brief dropouts don't reset, long gap does", () => {
  const tg = new PalmToggle();
  let n = 0, t = 0;
  for (let i = 0; i < 15; i++) { n += tg.update(i % 3 !== 2, t) ? 1 : 0; t += DT; }
  assert.equal(n, 1);
  const tg2 = new PalmToggle();
  let [x, t2] = feed(tg2, true, 0, 0.35);
  let y; [y, t2] = feed(tg2, false, t2, 0.4);
  let z; [z] = feed(tg2, true, t2, 0.35);
  assert.equal(x + y + z, 0);
});
test("toggle: manual toggle and set()", () => {
  const tg = new PalmToggle();
  tg.toggle(0);
  assert.equal(tg.active, true);
  tg.set(true, 5);
  assert.equal(tg.active, true);
  tg.set(false, 5);
  assert.equal(tg.active, false);
});

// ---- mask ops -----------------------------------------------------------------------
test("dilate/erode/blur basics", () => {
  const w = 9, h = 9, m = new Uint8Array(w * h);
  m[4 * w + 4] = 255;
  const d = dilate(m, w, h, 2);
  assert.equal(d.filter((v) => v).length, 25);
  assert.equal(erode(d, w, h, 2).filter((v) => v).length, 1);
  const b = blur(Float32Array.from(m, (v) => v / 255), w, h, 1);
  assert.ok(b[4 * w + 4] < 1 && b[4 * w + 4] > 0 && b[0] === 0);
});
test("fillDown extends to the bottom only", () => {
  const w = 3, h = 5, m = new Uint8Array(w * h);
  m[2 * w + 1] = 255;
  const f = fillDown(m, w, h);
  assert.deepEqual([0, 1, 2, 3, 4].map((y) => f[y * w + 1]), [0, 0, 255, 255, 255]);
  assert.equal(f[4 * w + 0], 0);
});
test("backgroundMask excludes person, area below, and margin", () => {
  const w = 40, h = 30, p = new Float32Array(w * h);
  for (let y = 5; y < 10; y++) for (let x = 15; x < 25; x++) p[y * w + x] = 1;
  const bg = backgroundMask(p, w, h, { margin: 2 });
  assert.equal(bg[7 * w + 20], 0);   // person
  assert.equal(bg[25 * w + 20], 0);  // below the person
  assert.equal(bg[7 * w + 26], 0);   // margin
  assert.equal(bg[2 * w + 2], 255);  // far away
  assert.equal(backgroundMask(p, w, h, { margin: 2, down: false })[25 * w + 20], 255);
  // Body cone: well below the person, the excluded area is wider than the person.
  assert.equal(bg[28 * w + 33], 0);
  assert.equal(backgroundMask(p, w, h, { margin: 2, slope: 0 })[28 * w + 33], 255);
});
test("addHand covers the hand region, leaves the rest", () => {
  const w = 320, h = 180, hand = Array.from({ length: 21 }, () => [0.5, 0.5]);
  hand[0] = [0.5, 0.6];
  hand[9] = [0.5, 0.45];
  const out = addHand(new Float32Array(w * h), w, h, hand);
  assert.equal(out[90 * w + 160], 1);
  assert.equal(out[5 * w + 5], 0);
  const off = Array.from({ length: 21 }, () => [0.99, 0.99]);
  off[0] = [1.1, 1.2];
  assert.equal(addHand(new Float32Array(w * h), w, h, off).length, w * h);
});
test("MaskRefiner: grows instantly, shrinks slowly, soft threshold", () => {
  const r = new MaskRefiner({ dilatePx: 0, featherPx: 0, decay: 0.8 });
  const full = new Float32Array(4).fill(1);
  assert.equal(r.refine(full, 2, 2)[0], 1);
  assert.ok(Math.abs(r.refine(new Float32Array(4), 2, 2)[0] - 0.8) < 1e-6);
  const r2 = new MaskRefiner({ dilatePx: 0, featherPx: 0, decay: 0 });
  const o = r2.refine(Float32Array.from([0.1, 0.4, 0.9]), 3, 1);
  assert.deepEqual(Array.from(o, (v) => +v.toFixed(2)), [0, 0.5, 1]);
});

function room(w, h) {
  const img = new Uint8ClampedArray(w * h * 4), plate = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, v = [40 + ((x >> 3) * 37) % 120, 40 + ((y >> 3) * 53) % 120, 40 + (((x + y) >> 3) * 29) % 120];
    for (let c = 0; c < 3; c++) { img[i * 4 + c] = v[c]; plate[i * 3 + c] = v[c]; }
    img[i * 4 + 3] = 255;
  }
  return { img, plate };
}
test("differenceMask: clear when unchanged, detects new object, ignores exposure", () => {
  const w = 80, h = 45, { img, plate } = room(w, h), known = new Uint8Array(w * h).fill(255);
  let r = differenceMask(img, plate, known, w, h);
  assert.equal(r.changed, 0);
  const obj = img.slice();
  for (let y = 15; y < 35; y++) for (let x = 30; x < 50; x++) for (let c = 0; c < 3; c++) obj[(y * w + x) * 4 + c] = 230;
  r = differenceMask(obj, plate, known, w, h);
  assert.ok(r.mask[25 * w + 40] > 0.9 && r.mask[3 * w + 3] < 0.1);
  assert.ok(r.changed > 0.05 && r.changed < 0.3, String(r.changed));
  const bright = img.map((v, i) => (i % 4 === 3 ? 255 : Math.min(255, v * 1.3)));
  assert.ok(differenceMask(bright, plate, known, w, h).changed < 0.02);
  const half = known.slice();
  for (let i = 0; i < half.length; i++) if (i % w >= 40) half[i] = 0;
  const m = differenceMask(obj, plate, half, w, h).mask;
  for (let y = 0; y < h; y++) for (let x = 41; x < w; x++) assert.equal(m[y * w + x], 0);
});
test("pullPushFill fills holes from surroundings", () => {
  const w = 40, h = 20, rgb = new Float32Array(w * h * 3), known = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, v = x < 20 ? 30 : 200;
    if (x < 15 || x >= 25) { known[i] = 255; rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v; }
  }
  const out = pullPushFill(rgb, known, w, h);
  assert.equal(out[(10 * w + 5) * 3], 30);           // known kept exactly
  assert.ok(out[(10 * w + 16) * 3] < 115);            // hole near dark side is darker
  assert.ok(out[(10 * w + 24) * 3] > 115);            // hole near light side is lighter
});
test("softKnown is exactly 0 in holes and ramps up inside known", () => {
  const w = 60, h = 20, known = new Uint8Array(w * h);
  for (let i = 0; i < known.length; i++) if (i % w < 30) known[i] = 255;
  const s = softKnown(known, w, h, 4);
  for (let y = 0; y < h; y++) for (let x = 30; x < w; x++) assert.equal(s[y * w + x], 0);
  assert.ok(s[10 * w + 2] > 0.99);
  const row = Array.from(s.subarray(10 * w + 15, 10 * w + 30));
  assert.ok(row.some((v) => v > 0.05 && v < 0.95));
});

// ---- background model -----------------------------------------------------------------
function frameWithPerson(w, h, x0) {
  const { img } = room(w, h), bg = new Uint8Array(w * h).fill(255);
  for (let y = 5; y < h - 5; y++) for (let x = x0; x < x0 + 20; x++) {
    for (let c = 0; c < 3; c++) img[(y * w + x) * 4 + c] = 255;
  }
  for (let y = 0; y < h; y++) for (let x = Math.max(0, x0 - 3); x < Math.min(w, x0 + 23); x++) bg[y * w + x] = 0;
  return { img, bg };
}
test("background: confirmation, never leaks a still person, moving reveals", () => {
  const w = 80, h = 45, bm = new BackgroundModel(w, h, { confirmFrames: 4, learnRate: 0.5 });
  const a = frameWithPerson(w, h, 30);
  for (let i = 0; i < 3; i++) bm.learn(a.img, a.bg);
  assert.equal(bm.coverage, 0);                     // not confirmed yet
  for (let i = 0; i < 5; i++) bm.learn(a.img, a.bg);
  assert.ok(bm.coverage > 0.5 && bm.coverage < 0.8);
  for (let i = 0; i < w * h; i++) if (bm.known[i]) assert.notEqual(bm.plate[i * 3], 255);
  const b = frameWithPerson(w, h, 0);
  for (let i = 0; i < 5; i++) bm.learn(b.img, b.bg);
  assert.ok(bm.coverage > 0.65);
});
test("background: flickering pixels are never learned", () => {
  const w = 80, h = 45, bm = new BackgroundModel(w, h, { confirmFrames: 4 });
  const a = frameWithPerson(w, h, 30), flick = new Uint8Array(w * h).fill(255);
  for (let i = 0; i < 20; i++) bm.learn(a.img, i % 2 ? flick : a.bg);
  for (let y = 5; y < h - 5; y++) for (let x = 30; x < 50; x++) assert.equal(bm.known[y * w + x], 0);
});
test("background: no new pixels while no person is detected; fresh confirmation after", () => {
  const w = 40, h = 20, bm = new BackgroundModel(w, h, { confirmFrames: 2 });
  const { img } = room(w, h), half = new Uint8Array(w * h), all = new Uint8Array(w * h).fill(255);
  for (let i = 0; i < half.length; i++) half[i] = i % w < 20 ? 255 : 0;
  for (let i = 0; i < 3; i++) bm.learn(img, half);
  const c0 = bm.coverage;
  for (let i = 0; i < 10; i++) bm.learn(img, all, false);
  assert.equal(bm.coverage, c0);
  bm.learn(img, all, true);
  assert.equal(bm.coverage, c0);
  bm.learn(img, all, true);
  assert.ok(bm.coverage > c0);
});
test("background: known pixels follow lighting; GPU masks match", () => {
  const w = 20, h = 10, bm = new BackgroundModel(w, h, { confirmFrames: 1, learnRate: 0.5 });
  const { img } = room(w, h), all = new Uint8Array(w * h).fill(255);
  const first = bm.learn(img, all);
  assert.equal(first.newMask.filter((v) => v).length, w * h);
  const bright = img.map((v, i) => (i % 4 === 3 ? 255 : Math.min(255, v + 30)));
  let last;
  for (let i = 0; i < 12; i++) last = bm.learn(bright, all, false);
  assert.equal(last.updateMask.filter((v) => v).length, w * h);
  assert.ok(Math.abs(bm.plate[0] - bright[0]) < 1);
});
test("background: settles only after coverage stops growing", () => {
  const w = 20, h = 10, bm = new BackgroundModel(w, h, { confirmFrames: 1, settleSeconds: 2, minCoverageForFill: 0.2 });
  const { img } = room(w, h), all = new Uint8Array(w * h).fill(255);
  bm.learn(img, all);
  assert.equal(bm.settled(0), false);
  assert.equal(bm.settled(1), false);
  assert.equal(bm.settled(2.5), true);
});
test("scene change monitor needs a sustained change", () => {
  const m = new SceneChangeMonitor({ threshold: 0.5, hold: 2 });
  assert.equal(m.update(0.9, 0), false);
  assert.equal(m.update(0.9, 2.1), true);
  m.update(0.9, 3);
  m.update(0.1, 4);
  assert.equal(m.update(0.9, 5.5), false);
});

test("background: empty-room capture = mean of frames, fully known", () => {
  const w = 10, h = 5, bm = new BackgroundModel(w, h);
  const a = new Uint8ClampedArray(w * h * 4).fill(100), b = new Uint8ClampedArray(w * h * 4).fill(200);
  assert.equal(bm.captureFrame(a, 0), 1);
  assert.equal(bm.captureFrame(b, 1), 0.5);
  assert.equal(bm.coverage, 1);
  assert.equal(bm.plate[0], 150);
});

test("differenceMaskHalf matches the full version on a clear object", () => {
  const w = 80, h = 44, { img, plate } = room(w, h), known = new Uint8Array(w * h).fill(255);
  const obj = img.slice();
  for (let y = 14; y < 34; y++) for (let x = 30; x < 50; x++) for (let c = 0; c < 3; c++) obj[(y * w + x) * 4 + c] = 230;
  const r = differenceMaskHalf(obj, plate, known, w, h);
  assert.equal(r.mask.length, w * h);
  assert.ok(r.mask[24 * w + 40] > 0.9 && r.mask[3 * w + 3] < 0.1);
  assert.ok(r.changed > 0.05 && r.changed < 0.3);
  assert.equal(differenceMaskHalf(img, plate, known, w, h).changed, 0);
});

test("exposureGain recovers a global brightening, incl. behind the person", () => {
  const w = 80, h = 44, { img, plate } = room(w, h), known = new Uint8Array(w * h).fill(255);
  const person = new Float32Array(w * h);
  for (let y = 10; y < 44; y++) for (let x = 25; x < 55; x++) person[y * w + x] = 1; // big person, plate behind is old
  const bright = img.map((v, i) => (i % 4 === 3 ? 255 : Math.min(255, Math.round(v * 1.3))));
  const g = exposureGain(bright, plate, known, person, w, h);
  const gw = 20, gh = 11;
  for (const [gx, gy] of [[2, 2], [10, 8], [18, 5]]) {   // in the room and behind the person
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(g[(gy * gw + gx) * 3 + c] - 1.3) < 0.08, `${gx},${gy}: ${g[(gy * gw + gx) * 3 + c]}`);
  }
  // Unchanged exposure -> ~1 everywhere.
  const g1 = exposureGain(img, plate, known, person, w, h);
  assert.ok(g1.every((v) => Math.abs(v - 1) < 0.02));
  // Nothing visible -> neutral.
  assert.ok(exposureGain(bright, plate, new Uint8Array(w * h), person, w, h).every((v) => v === 1));
});

test("exposureGain follows a left-to-right lighting gradient", () => {
  const w = 80, h = 44, { img, plate } = room(w, h), known = new Uint8Array(w * h).fill(255), person = new Float32Array(w * h);
  const lit = img.map((v, i) => { if (i % 4 === 3) return 255; const x = (i / 4) % w; return Math.min(255, Math.round(v * (1 + 0.4 * x / w))); });
  const g = exposureGain(lit, plate, known, person, w, h);
  assert.ok(g[(5 * 20 + 1) * 3] < g[(5 * 20 + 18) * 3] - 0.2);
});

test("learning with the exposure gain keeps the plate at the reference exposure", () => {
  const w = 40, h = 20, bm = new BackgroundModel(w, h, { confirmFrames: 1, learnRate: 0.5 });
  const { img } = room(w, h), all = new Uint8Array(w * h).fill(255);
  bm.learn(img, all);                                   // reference exposure
  const ref = bm.plate.slice();
  const bright = img.map((v, i) => (i % 4 === 3 ? 255 : Math.round(v * 1.25)));
  const gain = new Float32Array(w * h * 3).fill(1.25);
  for (let i = 0; i < 10; i++) bm.learn(bright, all, true, gain);
  let maxErr = 0;
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(bm.plate[i] - ref[i]));
  assert.ok(maxErr < 1.5, String(maxErr));             // unchanged, not brightened
});

test("upsampleGain interpolates and preserves constants", () => {
  const g = upsampleGain(new Float32Array([1, 1, 1, 2, 2, 2]), 2, 1, 8, 2);
  assert.equal(g[0], 1);
  assert.equal(g[(7) * 3], 2);
  assert.ok(g[4 * 3] > 1 && g[4 * 3] < 2);
});
