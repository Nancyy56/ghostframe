// Copies the MediaPipe runtime into vendor/ and puts the models into models/.
// Chrome extensions may not load remote code, so everything is packaged.
// Models are copied from ../models (the Python app's downloads) when present,
// otherwise downloaded from their official locations.
//
//   npm run setup

import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const MP = path.join(ROOT, "node_modules", "@mediapipe", "tasks-vision");

const MODELS = {
  "gesture_recognizer.task":
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
  "selfie_segmenter_landscape.tflite":
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
};
// Realistic fill model (MI-GAN, MIT licence) and its licence text.
const FILL = {
  "migan_pipeline_v2.onnx": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
  "MIGAN_LICENSE": "https://huggingface.co/andraniksargsyan/migan/raw/main/LICENSE",
};

function copy(src, dst) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
}

// MediaPipe runtime: ES-module bundle + ES-module WASM loader. The module
// loader works both in the processor frame and in the gesture worker (module
// workers can't use importScripts, which the classic loader needs).
const vendor = path.join(ROOT, "vendor", "mediapipe");
fs.rmSync(vendor, { recursive: true, force: true });
copy(path.join(MP, "vision_bundle.mjs"), path.join(vendor, "vision_bundle.mjs"));
for (const f of ["vision_wasm_module_internal.js", "vision_wasm_module_internal.wasm"]) {
  copy(path.join(MP, "wasm", f), path.join(vendor, "wasm", f));
}
console.log("MediaPipe runtime -> vendor/mediapipe");

// ONNX Runtime Web (WebGPU build + its WASM) for the realistic fill.
const ORT = path.join(ROOT, "node_modules", "onnxruntime-web", "dist");
const ortDir = path.join(ROOT, "vendor", "ort");
fs.rmSync(ortDir, { recursive: true, force: true });
for (const f of ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) {
  copy(path.join(ORT, f), path.join(ortDir, f));
}
console.log("ONNX Runtime Web -> vendor/ort");

// LaMa (OpenCV's export, Apache-2.0): best fill for large never-seen areas.
FILL["inpainting_lama_2025jan.onnx"] = "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx";

const wanted = { ...MODELS, ...FILL };
for (const [name, url] of Object.entries(wanted)) {
  const dst = path.join(ROOT, "models", name);
  if (fs.existsSync(dst)) continue;
  const local = path.join(ROOT, "..", "models", name);
  if (fs.existsSync(local)) {
    copy(local, dst);
    console.log(`${name}: copied from ../models`);
    continue;
  }
  process.stdout.write(`${name}: downloading... `);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.writeFileSync(dst, Buffer.from(await res.arrayBuffer()));
  console.log("done");
}
