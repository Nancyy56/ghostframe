"""High-quality hole filling for never-seen background (behind a still person)
using the LaMa inpainting model, run in a background thread.

LaMa takes ~8-10 s per fill on the i5-12450H CPU, so it never runs per frame:
the plate uses a cheap OpenCV fill instantly and swaps in the LaMa result when
ready. Because the known area only grows, the hole only shrinks, so an older
LaMa fill stays valid; it is recomputed only when the hole has shrunk
noticeably (new context to use) or periodically to follow lighting.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "inpainting_lama_2025jan.onnx")
LAMA_SIZE = 512


class LamaFiller:
    def __init__(self, model_path: str = MODEL_PATH, threads: int = 2,
                 min_interval: float = 10.0, refresh_interval: float = 120.0,
                 shrink_ratio: float = 0.85):
        import onnxruntime as ort  # optional dependency; caller falls back if missing

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"LaMa model not found at {model_path}; see README")
        so = ort.SessionOptions()
        # Few threads in ONNX Runtime's own pool, so the video loop keeps its CPU.
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3  # hide graph-optimizer warnings
        self._session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.min_interval = min_interval
        self.refresh_interval = refresh_interval
        self.shrink_ratio = shrink_ratio
        self._lock = threading.Lock()
        self._busy = False
        self._generation = 0          # bumped on reset; stale jobs are discarded
        self._result: np.ndarray | None = None
        self._version = 0             # bumped whenever a new result is published
        self._last_start = -1e9
        self._last_hole_px = 0
        self.last_duration = 0.0

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def version(self) -> int:
        return self._version

    def result(self) -> np.ndarray | None:
        with self._lock:
            return self._result

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._result = None
            self._version += 1
            self._last_start = -1e9
            self._last_hole_px = 0

    def maybe_start(self, image: np.ndarray, known: np.ndarray, now: float) -> None:
        """Start a background fill if one is due. Cheap to call every frame."""
        if self._busy:
            return
        hole_px = known.size - cv2.countNonZero(known)
        if hole_px == 0:
            return
        if self._result is not None:
            since = now - self._last_start
            shrunk = hole_px < self._last_hole_px * self.shrink_ratio
            if since < self.min_interval or not (shrunk or since >= self.refresh_interval):
                return
        self._busy = True
        self._last_start = now
        self._last_hole_px = hole_px
        gen = self._generation
        threading.Thread(target=self._run, args=(image.copy(), known.copy(), gen), daemon=True).start()

    def _run(self, image: np.ndarray, known: np.ndarray, gen: int) -> None:
        try:
            t = time.perf_counter()
            out = self.inpaint(image, known)
            self.last_duration = time.perf_counter() - t
            with self._lock:
                if gen == self._generation:
                    self._result = out
                    self._version += 1
        except Exception as e:  # never take the video loop down
            print(f"[ghostframe] warning: LaMa fill failed ({e})")
        finally:
            self._busy = False

    def inpaint(self, image: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Synchronous LaMa fill of the pixels where `known` == 0. Returns full-size BGR."""
        h, w = image.shape[:2]
        img = cv2.resize(image, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        hole = cv2.resize(cv2.bitwise_not(known), (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))  # cover resize fringe
        img_t = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_t = (hole > 0).astype(np.float32)[None, None]
        img_t = img_t * (1.0 - mask_t)  # hole content must not leak into the model
        out = self._session.run(None, {"image": img_t, "mask": mask_t})[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
