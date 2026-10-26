// Person segmentation in its own thread. The processor sends the small
// analysis frame and uses the newest mask (at most one frame old), so the
// ~15-20 ms per segmentation runs in parallel with compositing instead of
// adding to every frame's latency. The one-frame lag is covered by the
// difference-from-room mask, dilation and the grow-fast/shrink-slow filter.

import { ImageSegmenter } from "../vendor/mediapipe/vision_bundle.mjs";
import { modelUrl, visionFileset } from "./mediapipe.js";

let segmenter = null;
let lastTs = 0;

async function init() {
  segmenter = await ImageSegmenter.createFromOptions(await visionFileset(), {
    baseOptions: { modelAssetPath: modelUrl("selfie_segmenter_landscape.tflite"), delegate: "CPU" },
    runningMode: "VIDEO",
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  });
  postMessage({ type: "ready" });
}

self.onmessage = (e) => {
  const { bitmap, ts, id } = e.data;
  if (!segmenter) {
    bitmap.close();
    postMessage({ type: "result", id, mask: null });
    return;
  }
  const t = Math.max(Math.round(ts), lastTs + 1);
  lastTs = t;
  const { width: w, height: h } = bitmap; // read before close(): a closed bitmap reports 0x0
  let mask = null;
  try {
    const r = segmenter.segmentForVideo(bitmap, t);
    mask = r.confidenceMasks[0].getAsFloat32Array().slice();
    r.close();
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  } finally {
    bitmap.close();
  }
  postMessage({ type: "result", id, mask, w, h }, mask ? [mask.buffer] : []);
};

init().catch((err) => postMessage({ type: "error", message: "init: " + String(err) }));
