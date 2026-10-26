// GhostFrame processor (extension-origin frame, may use WebAssembly).
//
// Receives camera frames from the page hook as ImageBitmaps and returns the
// processed frame. Port of the Python app's pipeline (ghostframe/__main__.py):
//
//   segmentation (+ hand hull from the gesture worker)
//   + difference from the learned room          -> "person" (pose-independent)
//   background learning (confirmation, fill-down, no learning when no person seen)
//   camera-moved / lighting-changed detection    -> automatic relearn
//   never-seen areas always replaced, basic fill  -> GPU composite with fade
//
// Settings come from chrome.storage; commands (keyboard shortcut, popup) and
// status go through a port to the service worker.

import { BackgroundModel, SceneChangeMonitor } from "./core/background.js";
import {
  MaskRefiner, addHand, backgroundMask, differenceMaskHalf, exposureGain, pullPushFill, softKnown, upsampleGain,
} from "./core/masks.js";
import { PalmToggle } from "./core/toggle.js";
import { Compositor } from "./gpu.js";
import { deleteBackground, loadBackground, saveBackground } from "./store.js";

const LOW_W = 320;            // processing resolution width
const FADE_SECONDS = 0.3;
const GESTURE_MAX_AGE = 0.3;  // s; older gesture results are ignored
const PALM_MIN_SCORE = 0.6;
const MIN_PERSON = 0.02;      // segmenter must see >=2% person for new background to be learned
const LEARN_WARMUP = 1.5;     // s after the camera starts before anything new is learned
const SCENE_CHANGED = 0.5;    // >50% of the visible room differs: camera moved / lighting changed
const MIN_COVERAGE = 0.25;    // fraction of room known before invisibility works
const FILL_INTERVAL = 0.5;    // s between basic-fill refreshes
const SAVE_INTERVAL = 30;     // s between background saves
const CAPTURE_COUNTDOWN = 3;  // s to step out of view for an empty-room capture
const CAPTURE_FRAMES = 20;
// Realistic fills for never-seen background, best first. LaMa is much better
// on large holes but takes ~30 s (WASM only); MI-GAN is fast (~0.3-2 s) and
// good on small holes but invents objects in large ones, so it's only used
// for small holes and never replaces a LaMa result. Timing: [min interval,
// refresh interval] in s; shrink: rerun when the hole shrank below this ratio.
const FILLERS = {
  lama: { script: "./lama-worker.js", model: "inpainting_lama_2025jan.onnx", rank: 2, maxHole: 1, shrink: 0.85, timing: { wasm: [120, 600] } },
  migan: { script: "./fill-worker.js", model: "migan_pipeline_v2.onnx", rank: 1, maxHole: 0.15, shrink: 0.95, timing: { webgpu: [3, 30], wasm: [10, 60] } },
};

const DEFAULTS = { enabled: true, gestures: true, holdTime: 0.5 };

const parentWin = window.parent;
const parentOrigin = location.ancestorOrigins && location.ancestorOrigins[0];
const now = () => performance.now() / 1000;

let settings = { ...DEFAULTS };
let ready = false;
let initError = "";
const toggle = new PalmToggle({ holdTime: DEFAULTS.holdTime });

// ---- gesture worker ------------------------------------------------------------
const gesture = { label: "", score: 0, hand: null, t: -1e9 };
let worker = null;
let workerBusy = false;
let workerReady = false;

function startWorker() {
  worker = new Worker(new URL("./gesture-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const d = e.data;
    if (d.type === "ready") workerReady = true;
    else if (d.type === "result") {
      workerBusy = false;
      Object.assign(gesture, { label: d.label, score: d.score, hand: d.hand, t: d.ts / 1000 });
    } else if (d.type === "error") {
      workerBusy = false;
      console.warn("[GhostFrame] gesture:", d.message);
    }
  };
  worker.onerror = (e) => {
    workerBusy = false;
    console.warn("[GhostFrame] gesture worker failed:", e.message);
  };
}

function sendToWorker(bitmap, w, h) {
  if (!worker || workerBusy || !workerReady || !settings.gestures) return;
  workerBusy = true;
  const gw = 640, gh = Math.round((640 * h) / w);
  createImageBitmap(bitmap, { resizeWidth: gw, resizeHeight: gh, resizeQuality: "low" })
    .then((small) => worker.postMessage({ bitmap: small, ts: performance.now() }, [small]))
    .catch(() => { workerBusy = false; });
}

// ---- segmentation worker -------------------------------------------------------
const seg = { mask: null, w: 0, h: 0, busy: false, ready: false, worker: null };

function startSegmenter() {
  return new Promise((resolve, reject) => {
    seg.worker = new Worker(new URL("./segment-worker.js", import.meta.url), { type: "module" });
    seg.worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "ready") { seg.ready = true; resolve(); }
      else if (d.type === "result") {
        seg.busy = false;
        if (d.mask) Object.assign(seg, { mask: d.mask, w: d.w, h: d.h });
      } else if (d.type === "error") {
        seg.busy = false;
        if (!seg.ready) reject(new Error(d.message));
        else console.warn("[GhostFrame] segmentation:", d.message);
      }
    };
    seg.worker.onerror = (e) => { seg.busy = false; if (!seg.ready) reject(new Error(e.message)); };
  });
}

/** Hand the analysis frame to the segmentation worker (transfers ownership). */
function sendToSegmenter(small) {
  if (!seg.ready || seg.busy) return small.close();
  seg.busy = true;
  seg.worker.postMessage({ bitmap: small, ts: performance.now() }, [small]);
}

// ---- realistic fill workers ------------------------------------------------------------
const fillers = Object.fromEntries(Object.keys(FILLERS).map((name) =>
  [name, { worker: null, available: null, ready: false, backend: "", busy: false, job: 0, lastMs: 0 }]));

async function checkFillers() {
  await Promise.all(Object.entries(FILLERS).map(async ([name, cfg]) => {
    try {
      const r = await fetch(new URL(`../models/${cfg.model}`, import.meta.url), { method: "HEAD" });
      fillers[name].available = r.ok;
    } catch {
      fillers[name].available = false; // model not packaged
    }
  }));
}

function startFiller(name) {
  const f = fillers[name];
  f.worker = new Worker(new URL(FILLERS[name].script, import.meta.url), { type: "module" });
  f.worker.onmessage = (e) => {
    const d = e.data;
    if (d.type === "ready") Object.assign(f, { ready: true, backend: d.backend });
    else if (d.type === "result") {
      f.busy = false;
      f.lastMs = d.ms;
      const p = P, st = p && p.fill[name];
      if (p && d.job === st.job && FILLERS[name].rank >= p.fillRank) {
        if (p.realFill) p.realFill.close();
        p.realFill = d.bitmap;
        p.comp.setFill(d.bitmap);
        p.fillRank = FILLERS[name].rank;
        p.fillSource = name;
        st.done = true;
      } else d.bitmap.close();
    } else if (d.type === "error") {
      f.busy = false;
      console.warn(`[GhostFrame] ${name} fill:`, d.message);
      if (!f.ready) f.available = false;
    }
  };
}

function maybeStartFills(p, t) {
  if (p.capture || p.bm.coverage >= 0.995) return;
  const holePx = p.bm.w * p.bm.h - p.bm.knownCount;
  for (const [name, cfg] of Object.entries(FILLERS)) {
    const f = fillers[name], st = p.fill[name];
    if (!f.available || f.busy) continue;
    if (p.fillRank > cfg.rank) continue;                    // a better fill exists
    if (holePx / (p.bm.w * p.bm.h) > cfg.maxHole) continue;  // too big for this model
    if (!f.worker) startFiller(name);
    if (!f.ready || !p.bm.settled(t)) continue;
    if (st.done) { // this model already produced a fill: refresh only when worthwhile
      const [minInterval, refresh] = cfg.timing[f.backend] || Object.values(cfg.timing)[0];
      const since = t - st.start;
      if (since < minInterval || !(holePx < st.holePx * cfg.shrink || since >= refresh)) continue;
    }
    f.busy = true;
    st.start = t;
    st.holePx = holePx;
    st.job = ++f.job;
    const job = st.job;
    const rgba = p.comp.readPlate();
    createImageBitmap(new ImageData(new Uint8ClampedArray(rgba.buffer), p.w, p.h)).then((plate) => {
      f.worker.postMessage({ plate, known: p.bm.known.slice(), lw: p.lw, lh: p.lh, job }, [plate]);
    }).catch(() => { f.busy = false; });
  }
}

const newFillState = () => Object.fromEntries(Object.keys(FILLERS).map((n) => [n, { job: 0, start: -1e9, holePx: 0, done: false }]));

// ---- per-resolution pipeline ------------------------------------------------------
let P = null;

function createPipeline(w, h, cam) {
  const lw = LOW_W, lh = Math.round((LOW_W * h) / w);
  const small = new OffscreenCanvas(lw, lh);
  const p = {
    w, h, lw, lh, key: `${cam || "default"}:${w}x${h}`,
    comp: new Compositor(w, h),
    bm: new BackgroundModel(lw, lh),
    // Slightly wider edge than the desktop app: the browser's mask can be a frame old.
    refiner: new MaskRefiner({ dilatePx: 4, featherPx: 4 }),
    gain: null, // camera exposure vs the plate's reference exposure (low-res RGB map)
    scene: new SceneChangeMonitor({ threshold: SCENE_CHANGED }),
    small, ctx: small.getContext("2d", { willReadFrequently: true }),
    softVersion: -1, fillAt: -1e9, fillVersion: -1, savedAt: now(), savedVersion: 0,
    capture: null, sceneChanged: false, strength: 0, lastT: now(), startT: now(),
    realFill: null, fillRank: 0, fillSource: "", fill: newFillState(),
  };
  restore(p);
  return p;
}

function disposePipeline() {
  if (P) {
    P.comp.dispose();
    if (P.realFill) P.realFill.close();
  }
  P = null;
}

async function restore(p) {
  try {
    const rec = await loadBackground(p.key);
    if (!rec || rec.width !== p.w || rec.height !== p.h || rec.lowW !== p.lw || rec.lowH !== p.lh || P !== p) return;
    const bmp = await createImageBitmap(rec.plate);
    if (P !== p) return bmp.close();
    p.comp.loadPlate(bmp);
    bmp.close();
    p.bm.load(rec.lowPlate, rec.known);
    p.savedVersion = p.bm.version;
    if (rec.fill && P === p) {
      // Last session's realistic fill: better than the basic fill until a fresh one is ready.
      p.realFill = await createImageBitmap(rec.fill);
      p.comp.setFill(p.realFill);
      // Use it now; a fresh fill of the same quality still runs once (state not "done").
      p.fillSource = rec.fillSource || "migan";
      p.fillRank = FILLERS[p.fillSource] ? FILLERS[p.fillSource].rank : 1;
    }
  } catch (err) {
    console.warn("[GhostFrame] could not restore background:", err);
  }
}

async function save(p) {
  if (!p || p.bm.coverage === 0 || p.capture) return;
  p.savedAt = now();
  p.savedVersion = p.bm.version;
  try {
    const rgba = p.comp.readPlate();
    const c = new OffscreenCanvas(p.w, p.h);
    c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), p.w, p.h), 0, 0);
    const plate = await c.convertToBlob({ type: "image/png" });
    let fill;
    if (p.fillRank > 0 && p.realFill) {
      const fc = new OffscreenCanvas(p.realFill.width, p.realFill.height);
      fc.getContext("2d").drawImage(p.realFill, 0, 0);
      fill = await fc.convertToBlob({ type: "image/png" });
    }
    await saveBackground(p.key, {
      width: p.w, height: p.h, lowW: p.lw, lowH: p.lh, plate, known: p.bm.known.slice(), lowPlate: p.bm.plate.slice(), fill, fillSource: p.fillSource,
    });
  } catch (err) {
    console.warn("[GhostFrame] could not save background:", err);
  }
}

function relearn(p, reason) {
  if (!p) return;
  p.bm.reset();
  p.refiner.reset();
  p.softVersion = p.fillVersion = -1;
  p.fillAt = -1e9;
  p.fillRank = 0;
  p.fillSource = "";
  p.gain = null;
  p.fill = newFillState(); // results of running jobs are discarded
  if (p.realFill) p.realFill.close();
  p.realFill = null;
  deleteBackground(p.key).catch(() => {});
  console.info("[GhostFrame] relearning background:", reason);
}

// ---- per-frame processing ------------------------------------------------------------
const stats = { fps: 0, ms: 0, frames: 0, lastFrameT: 0, steps: {} };
let mark = 0;
function step(name) { // per-step timing (EMA, ms) for profiling
  const t = performance.now(), d = t - mark;
  mark = t;
  const s = stats.steps;
  s[name] = s[name] === undefined ? d : 0.9 * s[name] + 0.1 * d;
}

function processFrame(bitmap, small, cam) {
  const t0 = performance.now();
  mark = t0;
  const t = now();
  if (!P || P.w !== bitmap.width || P.h !== bitmap.height) {
    disposePipeline();
    P = createPipeline(bitmap.width, bitmap.height, cam);
  }
  const p = P;
  const { lw, lh, bm } = p;
  const dt = Math.min(0.2, t - p.lastT);
  p.lastT = t;

  p.ctx.drawImage(small || bitmap, 0, 0, lw, lh);
  if (small) sendToSegmenter(small);
  else createImageBitmap(p.small).then(sendToSegmenter);
  const img = p.ctx.getImageData(0, 0, lw, lh).data;
  step("downscale");
  p.comp.setFrame(bitmap);
  step("upload");

  // Gestures (asynchronous, newest result).
  sendToWorker(bitmap, p.w, p.h);
  const gestureFresh = t - gesture.t <= GESTURE_MAX_AGE;
  const palm = gestureFresh && gesture.label === "Open_Palm" && gesture.score >= PALM_MIN_SCORE;
  if (settings.gestures && toggle.update(palm, t)) announce();
  const hand = gestureFresh ? gesture.hand : null;

  // Segmentation: newest mask from the worker (at most a frame old).
  const segMask = seg.mask && seg.w === lw && seg.h === lh ? seg.mask : new Float32Array(lw * lh);
  step("segment");
  let person = hand ? addHand(segMask, lw, lh, hand) : segMask;
  let segCount = 0;
  for (let i = 0; i < segMask.length; i++) if (segMask[i] > 0.5) segCount++;
  // Only the segmenter's own evidence unlocks learning new background: a hand
  // alone must not, or everything else (the rest of the person) gets learned.
  // Nothing new is learned during warm-up either: the first frames have no
  // hand detection yet (raised fingers would be learned as background).
  const warm = t - p.startT >= LEARN_WARMUP && (workerReady || !settings.gestures);
  const personSeen = warm && segCount / segMask.length >= MIN_PERSON;

  // Anything that differs from the known room is treated as the person.
  p.sceneChanged = false;
  if (bm.coverage >= 0.2 && !p.capture) {
    const room = new Uint8Array(segMask.length);
    for (let i = 0; i < segMask.length; i++) room[i] = bm.known[i] && segMask[i] < 0.1 ? 1 : 0;
    const { mask: diff, changed } = differenceMaskHalf(img, bm.plate, bm.known, lw, lh, { region: room });
    p.sceneChanged = changed > SCENE_CHANGED;
    if (!p.sceneChanged) {
      const merged = new Float32Array(person.length);
      for (let i = 0; i < merged.length; i++) merged[i] = Math.max(person[i], diff[i]);
      person = merged;
    }
    if (p.scene.update(changed, t)) relearn(p, "camera moved or lighting changed");
  }
  step("difference");

  // Camera exposure / white balance relative to the plate's reference
  // exposure, from the room visible right now. The plate is learned at the
  // reference exposure (frames divided by this) and shown at the current one
  // (multiplied by it), so parts learned at different times all match.
  const gw = Math.ceil(lw / 4), gh = Math.ceil(lh / 4);
  let gainPx = null;
  if (!p.capture && bm.coverage > 0) {
    const g = exposureGain(img, bm.plate, bm.known, person, lw, lh);
    if (!p.gain || p.gain.length !== g.length) p.gain = g;
    else for (let i = 0; i < g.length; i++) p.gain[i] = 0.6 * p.gain[i] + 0.4 * g[i];
    p.comp.setGain(p.gain, gw, gh);
    gainPx = upsampleGain(p.gain, gw, gh, lw, lh);
  } else {
    p.gain = null;
    p.comp.setGain(new Float32Array([1, 1, 1]), 1, 1);
  }
  step("exposure");

  // Learning (or manual empty-room capture).
  if (p.capture) {
    if (t - p.capture.start >= CAPTURE_COUNTDOWN) {
      const k = p.capture.frames++;
      if (k === 0) bm.reset();
      const rate = bm.captureFrame(img, k);
      const all = new Uint8Array(lw * lh).fill(255);
      p.comp.learn(new Uint8Array(lw * lh), all, lw, lh, rate);
      if (p.capture.frames >= CAPTURE_FRAMES) {
        p.capture = null;
        p.refiner.reset();
        save(p);
      }
    }
  } else {
    const { newMask, updateMask } = bm.learn(img, backgroundMask(person, lw, lh), personSeen, gainPx);
    p.comp.learn(newMask, updateMask, lw, lh, bm.learnRate);
  }
  maybeStartFills(p, t);
  step("learn");

  // Fade in/out.
  const target = toggle.active && settings.enabled && bm.coverage >= MIN_COVERAGE && !p.capture ? 1 : 0;
  const fade = dt / FADE_SECONDS;
  p.strength = target > p.strength ? Math.min(target, p.strength + fade) : Math.max(target, p.strength - fade);

  let out = bitmap;
  if (p.strength > 0) {
    if (bm.version !== p.softVersion) {
      p.comp.setSoft(softKnown(bm.known, lw, lh), lw, lh);
      p.softVersion = bm.version;
    }
    if (p.fillRank === 0 && (p.fillVersion === -1 || (bm.version !== p.fillVersion && t - p.fillAt >= FILL_INTERVAL))) {
      p.comp.setFill(pullPushFill(bm.plate, bm.known, lw, lh), lw, lh);
      p.fillVersion = bm.version;
      p.fillAt = t;
    }
    // Never-seen areas have always been covered by the person: always replace them.
    const comp = new Float32Array(person.length);
    for (let i = 0; i < comp.length; i++) comp[i] = Math.max(person[i], bm.known[i] ? 0 : 1);
    const refined = p.refiner.refine(comp, lw, lh);
    step("refine");
    out = p.comp.composite(refined, lw, lh, p.strength);
    step("composite");
    bitmap.close();
  } else {
    p.refiner.reset();
  }

  if (t - p.savedAt >= SAVE_INTERVAL && !p.capture) save(p);

  const ms = performance.now() - t0;
  stats.ms = stats.ms ? 0.9 * stats.ms + 0.1 * ms : ms;
  if (stats.lastFrameT) {
    const inst = 1 / Math.max(1e-3, t - stats.lastFrameT);
    stats.fps = stats.fps ? 0.9 * stats.fps + 0.1 * inst : inst;
  }
  stats.lastFrameT = t;
  stats.frames++;
  return out;
}

// ---- status, settings, commands ----------------------------------------------------------
function status() {
  const p = P;
  let background = "none";
  if (p) {
    if (p.capture) background = now() - p.capture.start < CAPTURE_COUNTDOWN ? "capture-countdown" : "capturing";
    else if (p.sceneChanged) background = "scene-changed";
    else if (p.bm.coverage >= 0.98) background = "ready";
    else if (p.bm.coverage >= MIN_COVERAGE) background = "partial";
    else if (p.bm.coverage > 0) background = "learning";
  }
  return {
    ready, initError, enabled: settings.enabled, gestures: settings.gestures, gesturesReady: workerReady,
    invisible: toggle.active, coverage: p ? +p.bm.coverage.toFixed(3) : 0, background,
    captureLeft: p && p.capture ? Math.max(0, CAPTURE_COUNTDOWN - (now() - p.capture.start)) : 0,
    gesture: now() - gesture.t <= GESTURE_MAX_AGE ? gesture.label : "", holdProgress: +toggle.progress(now()).toFixed(2),
    fill: Object.values(fillers).some((f) => f.busy) ? (p && p.fillRank ? "realistic-improving" : "improving")
      : p && p.fillRank ? "realistic" : "basic",
    fillSource: p ? p.fillSource : "",
    fillBackend: Object.entries(fillers).filter(([, f]) => f.ready).map(([n, f]) => `${n}:${f.backend}`).join(" "),
    fillMs: Object.fromEntries(Object.entries(fillers).map(([n, f]) => [n, f.lastMs])),
    fps: +stats.fps.toFixed(1), ms: +stats.ms.toFixed(1),
    steps: Object.fromEntries(Object.entries(stats.steps).map(([k, v]) => [k, +v.toFixed(1)])), frames: stats.frames, resolution: p ? `${p.w}x${p.h}` : "",
  };
}

let port = null;
function connectPort() {
  try {
    port = chrome.runtime.connect({ name: "processor" });
  } catch {
    port = null;
    return;
  }
  port.onMessage.addListener((m) => handleCommand(m));
  port.onDisconnect.addListener(() => {
    port = null;
    setTimeout(connectPort, 1000); // service worker restarted: reconnect
  });
  announce();
}

function announce() {
  const s = status();
  if (port) try { port.postMessage({ type: "status", status: s }); } catch { /* reconnecting */ }
  if (parentOrigin) parentWin.postMessage({ gf: "state", state: s }, parentOrigin);
}

function handleCommand(m) {
  const t = now();
  switch (m && m.type) {
    case "toggle": toggle.toggle(t); break;
    case "set-invisible": toggle.set(!!m.value, t); break;
    case "relearn": relearn(P, "requested"); break;
    case "capture": if (P) P.capture = P.capture ? null : { start: t, frames: 0 }; break;
    case "status": break;
    default: return;
  }
  announce();
}

async function loadSettings() {
  settings = { ...DEFAULTS, ...(await chrome.storage.local.get(DEFAULTS)) };
  toggle.holdTime = settings.holdTime;
  if (!settings.enabled && toggle.active) toggle.set(false, now());
}
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local") loadSettings().then(announce);
});

// ---- frames from the page -----------------------------------------------------------------
window.addEventListener("message", (e) => {
  if (e.source !== parentWin || e.origin !== parentOrigin) return;
  const d = e.data;
  if (!d || typeof d.gf !== "string") return;
  if (d.gf === "hello") {
    if (ready) parentWin.postMessage({ gf: "ready" }, parentOrigin);
  } else if (d.gf === "frame" && d.bitmap instanceof ImageBitmap) {
    let out = d.bitmap;
    if (ready && settings.enabled) {
      try {
        out = processFrame(d.bitmap, d.small, d.cam);
      } catch (err) {
        console.warn("[GhostFrame] frame failed, passing through:", err);
        out = d.bitmap;
      }
    }
    if (d.small && out === d.bitmap) try { d.small.close(); } catch { /* already closed */ }
    parentWin.postMessage({ gf: "result", id: d.id, bitmap: out }, parentOrigin, [out]);
  }
});

// Save what was learned when the meeting tab closes.
addEventListener("pagehide", () => save(P));
setInterval(announce, 500);

// ---- startup --------------------------------------------------------------------------------
(async () => {
  try {
    await loadSettings();
    await startSegmenter();
    checkFillers();
    startWorker();
    ready = true;
    parentWin.postMessage({ gf: "ready" }, parentOrigin);
  } catch (err) {
    initError = String(err);
    console.warn("[GhostFrame] processor init failed; camera passes through unchanged:", err);
  }
  connectPort();
  announce();
})();
