// High-quality fill for large never-seen areas: LaMa (OpenCV's export,
// Apache-2.0) on ONNX Runtime Web, WASM backend, in its own thread.
//
// LaMa is clearly better than MI-GAN on large holes (on learned backgrounds:
// mean error 8.5 vs 17.8 and 29.6 vs 52.2; MI-GAN invents furniture/text, LaMa
// continues walls and floors). But its Fourier convolutions compute wrong
// results on ORT's WebGPU backend, so it runs on WASM: ~30 s per fill on one
// core. The processor therefore runs it rarely and caches the result.
// Preprocessing as in the Python app (ghostframe/fill.py): 512x512 BGR 0..1,
// hole content zeroed, mask = 1 in the hole.

import * as ort from "../vendor/ort/ort.webgpu.min.mjs";

const SIZE = 512;
let session = null;

async function init() {
  ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = 1; // extension frames aren't cross-origin isolated: no WASM threads
  ort.env.logLevel = "error";
  session = await ort.InferenceSession.create(new URL("../models/inpainting_lama_2025jan.onnx", import.meta.url).href,
    { executionProviders: ["wasm"] });
  postMessage({ type: "ready", backend: "wasm" });
}

async function fill(plate, known, lw, lh) {
  const c = new OffscreenCanvas(SIZE, SIZE);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(plate, 0, 0, SIZE, SIZE);
  plate.close();
  const px = g.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, hole = new Uint8Array(n);
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(lh - 1, Math.floor((y * lh) / SIZE));
    for (let x = 0; x < SIZE; x++) hole[y * SIZE + x] = known[sy * lw + Math.min(lw - 1, Math.floor((x * lw) / SIZE))] ? 0 : 1;
  }
  const img = new Float32Array(3 * n), mask = new Float32Array(n);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    // Grow the hole by 2 px to cover the resize fringe.
    let m = 0;
    for (let dy = -2; dy <= 2 && !m; dy++) for (let dx = -2; dx <= 2 && !m; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < SIZE && xx >= 0 && xx < SIZE && hole[yy * SIZE + xx]) m = 1;
    }
    const i = y * SIZE + x;
    mask[i] = m;
    img[i] = m ? 0 : px[i * 4 + 2] / 255;          // B
    img[n + i] = m ? 0 : px[i * 4 + 1] / 255;      // G
    img[2 * n + i] = m ? 0 : px[i * 4] / 255;      // R
  }
  const out = await session.run({
    image: new ort.Tensor("float32", img, [1, 3, SIZE, SIZE]),
    mask: new ort.Tensor("float32", mask, [1, 1, SIZE, SIZE]),
  });
  const o = out.output.data; // BGR 0..255
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    rgba[i * 4] = o[2 * n + i];
    rgba[i * 4 + 1] = o[n + i];
    rgba[i * 4 + 2] = o[i];
    rgba[i * 4 + 3] = 255;
  }
  return createImageBitmap(new ImageData(rgba, SIZE, SIZE));
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
