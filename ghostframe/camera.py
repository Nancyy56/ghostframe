"""Webcam / video-file capture with graceful failure and automatic reconnect."""

from __future__ import annotations

import time

import cv2
import numpy as np


class Camera:
    """Wraps cv2.VideoCapture.

    `source` is either an integer V4L2 device index (0 -> /dev/video0), a device
    path such as "/dev/video2", or a video file path (looped, for testing without
    a webcam). `read()` never raises: it returns None while the device is
    unavailable and retries opening it every `retry_interval` seconds.
    """

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, retry_interval: float = 2.0):
        # This OpenCV build's V4L2 backend can't open devices by path, so map
        # "/dev/videoN" to index N.
        if isinstance(source, str) and source.startswith("/dev/video") and source[10:].isdigit():
            source = int(source[10:])
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.retry_interval = retry_interval
        self.is_file = isinstance(source, str)
        self.cap: cv2.VideoCapture | None = None
        self.last_error = "not opened yet"
        self._next_retry = 0.0

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def open(self) -> bool:
        self.release()
        backend = cv2.CAP_ANY if self.is_file else cv2.CAP_V4L2
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            cap.release()
            self.last_error = f"cannot open camera source {self.source!r}"
            return False
        if not self.is_file:
            # MJPG is needed by most USB2 UVC webcams to reach 720p at 30 FPS;
            # raw YUYV is usually capped around 10 FPS at that resolution.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Don't set CAP_PROP_BUFFERSIZE=1: measured on the iVOOMi Netra it
            # halves capture rate (20 -> 10 FPS) because frames get dropped.
        self.cap = cap
        self.last_error = ""
        return True

    def describe(self) -> str:
        if not self.is_open:
            return f"closed ({self.last_error})"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        return f"{self.source!r} {w}x{h} @ {f:.0f} FPS (driver-reported)"

    def read(self) -> np.ndarray | None:
        if not self.is_open:
            now = time.monotonic()
            if now < self._next_retry:
                return None
            self._next_retry = now + self.retry_interval
            if not self.open():
                return None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame
        if self.is_file:
            # End of test video: loop it.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                return frame
        self.last_error = "frame read failed (camera disconnected?)"
        self.release()
        self._next_retry = time.monotonic() + self.retry_interval
        return None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
