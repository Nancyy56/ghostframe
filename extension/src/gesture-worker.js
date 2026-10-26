// Gesture recognition in its own thread, so the ~20-30 ms per detection never
// delays the video. The processor sends a downscaled frame when this worker is
// idle and uses the newest result (same idea as MediaPipe LIVE_STREAM mode in
// the Python app).

import { GestureRecognizer } from "../vendor/mediapipe/vision_bundle.mjs";
import { modelUrl, visionFileset } from "./mediapipe.js";

let recognizer = null;
let lastTs = 0;

async function init() {
  const fileset = await visionFileset();
  recognizer = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: modelUrl("gesture_recognizer.task"), delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 1,
  });
  postMessage({ type: "ready" });
}

self.onmessage = (e) => {
  const { bitmap, ts } = e.data;
  if (!recognizer) {
    bitmap.close();
    postMessage({ type: "result", ts, label: "", score: 0, hand: null });
    return;
  }
  const t = Math.max(Math.round(ts), lastTs + 1); // must strictly increase
  lastTs = t;
  let label = "", score = 0, hand = null;
  try {
    const r = recognizer.recognizeForVideo(bitmap, t);
    if (r.gestures.length && r.gestures[0].length) ({ categoryName: label, score } = r.gestures[0][0]);
    if (r.landmarks.length) hand = r.landmarks[0].map((p) => [p.x, p.y]);
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  } finally {
    bitmap.close();
  }
  postMessage({ type: "result", ts, label, score, hand });
};

init().catch((err) => postMessage({ type: "error", message: "init: " + String(err) }));
