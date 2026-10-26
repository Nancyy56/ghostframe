"""Open-palm gesture detection and the debounced on/off toggle it drives."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

OPEN_PALM = "Open_Palm"
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "gesture_recognizer.task")


class PalmToggle:
    """Turns a stream of "is an open palm visible?" observations into toggles.

    - The palm must be held for `hold_time` seconds to toggle.
    - Gaps shorter than `gap_tolerance` (missed detections) don't reset the hold.
    - After a toggle, nothing can trigger for `cooldown` seconds, and the palm
      must be lowered first, so one long hold doesn't flip the state repeatedly.

    Time is passed in explicitly so the logic can be unit-tested.
    """

    def __init__(self, hold_time: float = 0.5, cooldown: float = 1.5, gap_tolerance: float = 0.25):
        self.hold_time = hold_time
        self.cooldown = cooldown
        self.gap_tolerance = gap_tolerance
        self.active = False
        self._hold_start: float | None = None
        self._last_seen = -1e9
        self._cooldown_until = -1e9
        self._needs_release = False

    def update(self, palm_visible: bool, now: float) -> bool:
        """Feed one observation. Returns True if the state toggled on this call."""
        if palm_visible:
            self._last_seen = now
        elif now - self._last_seen > self.gap_tolerance:
            self._hold_start = None
            self._needs_release = False

        if self._needs_release or now < self._cooldown_until:
            self._hold_start = None
            return False
        if palm_visible and self._hold_start is None:
            self._hold_start = now
        # Only fire on a frame where the palm is actually seen, not during a dropout.
        if palm_visible and self._hold_start is not None and now - self._hold_start >= self.hold_time:
            return self._fire(now)
        return False

    def toggle(self, now: float) -> None:
        """Manual (keyboard) toggle; also starts the cooldown."""
        self._fire(now)

    def _fire(self, now: float) -> bool:
        self.active = not self.active
        self._hold_start = None
        self._cooldown_until = now + self.cooldown
        self._needs_release = now - self._last_seen <= self.gap_tolerance
        return True

    def progress(self, now: float) -> float:
        """0..1 hold progress, for drawing a progress bar."""
        if self._hold_start is None:
            return 0.0
        return min(1.0, (now - self._hold_start) / self.hold_time)

    def cooling_down(self, now: float) -> bool:
        return now < self._cooldown_until or self._needs_release


@dataclass
class GestureReading:
    label: str = ""
    score: float = 0.0
    timestamp: float = 0.0  # time.monotonic() of the frame it came from
    hand: np.ndarray | None = None  # (21, 2) landmarks, normalized 0..1 x/y


class GestureDetector:
    """Asynchronous MediaPipe GestureRecognizer (LIVE_STREAM mode).

    `submit()` returns immediately; MediaPipe drops frames while it is busy, so
    the video loop is never blocked. `latest()` returns the newest result.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, min_score: float = 0.6,
                 input_width: int = 640):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"gesture model not found at {model_path}; see README for the download command")
        self.min_score = min_score
        self.input_width = input_width
        self._lock = threading.Lock()
        self._reading = GestureReading()
        self._last_ts_ms = -1
        opts = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self._on_result,
        )
        self._rec = vision.GestureRecognizer.create_from_options(opts)

    def _on_result(self, result, _image, ts_ms: int) -> None:
        reading = GestureReading(timestamp=ts_ms / 1000.0)
        if result.gestures:
            top = result.gestures[0][0]
            reading.label, reading.score = top.category_name, top.score
        if result.hand_landmarks:
            reading.hand = np.array([(lm.x, lm.y) for lm in result.hand_landmarks[0]], np.float32)
        with self._lock:
            self._reading = reading

    def submit(self, frame_bgr: np.ndarray, now: float) -> None:
        ts_ms = int(now * 1000)
        if ts_ms <= self._last_ts_ms:  # MediaPipe requires strictly increasing timestamps
            return
        self._last_ts_ms = ts_ms
        h, w = frame_bgr.shape[:2]
        if w > self.input_width:
            frame_bgr = cv2.resize(frame_bgr, (self.input_width, h * self.input_width // w),
                                   interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._rec.recognize_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

    def latest(self) -> GestureReading:
        with self._lock:
            return self._reading

    def palm_visible(self, now: float, max_age: float = 0.3) -> bool:
        r = self.latest()
        return r.label == OPEN_PALM and r.score >= self.min_score and now - r.timestamp <= max_age

    def close(self) -> None:
        self._rec.close()
