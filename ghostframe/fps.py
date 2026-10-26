"""Smoothed frames-per-second meter."""

import time


class FPSMeter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) * inst
        self._last = now
        return self.fps
