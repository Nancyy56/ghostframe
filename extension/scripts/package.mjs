// Builds the Chrome Web Store upload: dist/ghostframe-<version>.zip containing
// only what the extension needs at runtime. Checks that every file the
// manifest and code reference is present.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const mf = JSON.parse(fs.readFileSync(path.join(ROOT, "manifest.json"), "utf8"));
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8"));
if (mf.version !== pkg.version) throw new Error(`version mismatch: manifest ${mf.version}, package.json ${pkg.version}`);

const INCLUDE = ["manifest.json", "icons", "popup", "src", "vendor", "models", "PRIVACY.md", "THIRD_PARTY_NOTICES.md"];
const REQUIRED = [
  "src/hook.js", "src/bridge.js", "src/processor.html", "src/processor.js", "src/background.js",
  "src/gesture-worker.js", "src/segment-worker.js", "src/fill-worker.js", "src/lama-worker.js",
  "vendor/mediapipe/vision_bundle.mjs", "vendor/mediapipe/wasm/vision_wasm_module_internal.wasm",
  "vendor/ort/ort.webgpu.min.mjs", "vendor/ort/ort-wasm-simd-threaded.asyncify.wasm",
  "models/gesture_recognizer.task", "models/selfie_segmenter_landscape.tflite", "models/migan_pipeline_v2.onnx", "models/inpainting_lama_2025jan.onnx",
  ...Object.values(mf.icons),
];
const missing = [...INCLUDE, ...REQUIRED].filter((f) => !fs.existsSync(path.join(ROOT, f)));
if (missing.length) throw new Error(`missing (run npm run setup?): ${missing.join(", ")}`);

const dist = path.join(ROOT, "dist");
fs.mkdirSync(dist, { recursive: true });
const zip = path.join(dist, `ghostframe-${mf.version}.zip`);
fs.rmSync(zip, { force: true });
execFileSync("zip", ["-r", "-q", "-X", zip, ...INCLUDE, "-x", "*.map", "*.DS_Store"], { cwd: ROOT });
const mb = (fs.statSync(zip).size / 1048576).toFixed(1);
console.log(`${path.relative(ROOT, zip)}  (${mb} MB)`);
