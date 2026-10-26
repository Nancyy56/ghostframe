// Realistic fill of never-seen background with MI-GAN (Picsart AI Research,
// MIT licence) via ONNX Runtime Web, in its own thread.
//
// Why MI-GAN and not LaMa (used by the Python app): LaMa's Fourier
// convolutions compute wrong results on ONNX Runtime Web's WebGPU backend and
// take ~30 s on single-threaded WASM. MI-GAN is plain convolutions: correct on
// both, ~0.25 s on WebGPU and ~1.4 s on WASM at 1280x720 (measured), and on
// real rooms its quality matched LaMa in our comparison.
//
// The "pipeline" export takes uint8 RGB + a uint8 mask (255 = keep, 0 = fill)
// at any size, crops around the hole, inpaints and pastes back.

import * as ort from "../vendor/ort/ort.webgpu.min.mjs";

let session = null;
let backend = "";

async function init() {
  ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = 1; // extension frames aren't cross-origin isolated: no WASM threads
  ort.env.logLevel = "error";
  const model = new URL("../models/migan_pipeline_v2.onnx", import.meta.url).href;
  let hasGpu = false;
  try {
    hasGpu = !!(self.navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch { /* no WebGPU */ }
  for (const ep of hasGpu ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: [ep] });
      backend = ep;
      break;
    } catch (err) {
      console.warn(`[GhostFrame] fill on ${ep} unavailable:`, err);
    }
  }
  if (!session) throw new Error("no backend for the fill model");
  postMessage({ type: "ready", backend });
}

/**
 * plate: ImageBitmap (full-res learned background), known: Uint8Array lw*lh
 * (255 = really seen). Returns a full-res ImageBitmap with the holes filled.
 */
async function fill(plate, known, lw, lh) {
  const W = plate.width, H = plate.height, n = W * H;
  const c = new OffscreenCanvas(W, H);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(plate, 0, 0);
  plate.close();
  const px = g.getImageData(0, 0, W, H).data;

  // Mask at full res from the low-res known map (nearest), with the hole
  // grown by a few pixels so the resize fringe gets filled too.
  const hole = new Uint8Array(lw * lh);
  for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
    if (known[y * lw + x]) continue;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < lh && xx >= 0 && xx < lw) hole[yy * lw + xx] = 1;
    }
  }
  const img = new Uint8Array(3 * n), mask = new Uint8Array(n);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(lh - 1, Math.floor((y * lh) / H));
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const h = hole[sy * lw + Math.min(lw - 1, Math.floor((x * lw) / W))];
      mask[i] = h ? 0 : 255;
      img[i] = px[i * 4];
      img[n + i] = px[i * 4 + 1];
      img[2 * n + i] = px[i * 4 + 2];
    }
  }
  const out = await session.run({
    image: new ort.Tensor("uint8", img, [1, 3, H, W]),
    mask: new ort.Tensor("uint8", mask, [1, 1, H, W]),
  });
  const o = out.result.data;
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    rgba[i * 4] = o[i];
    rgba[i * 4 + 1] = o[n + i];
    rgba[i * 4 + 2] = o[2 * n + i];
    rgba[i * 4 + 3] = 255;
  }
  for (const t of Object.values(out)) t.dispose?.();
  return createImageBitmap(new ImageData(rgba, W, H));
}

self.onmessage = async (e) => {
  const { plate, known, lw, lh, job } = e.data;
  try {
    const t = performance.now();
    const bitmap = await fill(plate, known, lw, lh);
    postMessage({ type: "result", job, bitmap, ms: Math.round(performance.now() - t) }, [bitmap]);
  } catch (err) {
    postMessage({ type: "error", job, message: String(err) });
  }
};

init().catch((err) => postMessage({ type: "error", message: "init: " + String(err) }));
