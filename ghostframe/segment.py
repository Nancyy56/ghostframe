"""Person segmentation and mask refinement (smoothing, anti-flicker, feathering)."""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEGMENTER_MODELS = {
    # Measured on i5-12450H at 640x360 input: landscape ~8 ms, square ~11 ms.
    "landscape": "selfie_segmenter_landscape.tflite",
    "square": "selfie_segmenter.tflite",
}


class PersonSegmenter:
    """MediaPipe selfie segmenter, run synchronously so the mask matches the frame.

    `segment()` returns a float32 person-probability mask (0..1) at the reduced
    processing resolution (`input_width` wide).
    """

    def __init__(self, model: str = "landscape", input_width: int = 640):
        path = os.path.join(MODELS_DIR, SEGMENTER_MODELS[model])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"segmentation model not found at {path}; see README for the download command")
        self.input_width = input_width
        self._last_ts_ms = -1
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def segment(self, frame_bgr: np.ndarray, now: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.input_width, h * self.input_width // w),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(now * 1000), self._last_ts_ms + 1)  # must strictly increase
        self._last_ts_ms = ts_ms
        result = self._seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        return np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32, copy=True)

    def close(self) -> None:
        self._seg.close()


class MaskRefiner:
    """Turns raw per-frame masks into a stable, soft-edged mask.

    Hiding too much only shows a little extra background plate (almost
    invisible), but hiding too little reveals the person. So the refiner is
    biased towards covering:

    - `low`/`high`: probabilities are remapped so anything above `low` counts
      partly and above `high` fully (a soft threshold).
    - Temporal: the mask grows instantly but shrinks by at most a factor
      `decay` per frame, which removes flicker and covers fast motion.
    - `dilate_px` grows the mask outward to hide edge halos (hair, arms).
    - `feather_px` blurs the edge so the transition isn't a hard cut-out line.

    Pixel sizes are at the processing resolution (640 px wide by default).
    """

    def __init__(self, low: float = 0.2, high: float = 0.6, decay: float = 0.8,
                 dilate_px: int = 6, feather_px: int = 7):
        self.low, self.high, self.decay = low, high, decay
        self.dilate_px, self.feather_px = dilate_px, feather_px
        self._prev: np.ndarray | None = None
        self._kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
                        if dilate_px > 0 else None)

    def reset(self) -> None:
        self._prev = None

    def refine(self, raw: np.ndarray) -> np.ndarray:
        m = np.clip((raw - self.low) / (self.high - self.low), 0.0, 1.0)
        if self._prev is not None and self._prev.shape == m.shape:
            m = np.maximum(m, self._prev * self.decay)
        self._prev = m
        # Morphology and blur on uint8 are several times faster than on float32.
        m8 = (m * 255).astype(np.uint8)
        if self._kernel is not None:
            m8 = cv2.dilate(m8, self._kernel)
        if self.feather_px > 0:
            k = 2 * self.feather_px + 1
            m8 = cv2.GaussianBlur(m8, (k, k), 0)
        return m8.astype(np.float32) * (1 / 255.0)


def add_hand(mask: np.ndarray, hand: np.ndarray, grow: float = 0.45) -> np.ndarray:
    """Union the mask with the convex hull of hand landmarks (normalized 0..1).

    The selfie segmenter often misses raised fingers, which is exactly where
    the toggle gesture happens. Landmarks sit on joint centres, so the hull is
    grown to reach fingertips and finger edges, by `grow` times the
    wrist-to-middle-knuckle distance (so it scales with how close the hand is).
    """
    h, w = mask.shape[:2]
    pts = hand * (w, h)
    grow_px = max(4, int(grow * np.linalg.norm(pts[0] - pts[9])))
    poly = cv2.convexHull(np.round(pts).astype(np.int32))
    # Work only in the hand's bounding box (plus margin) to keep this cheap.
    x, y, bw, bh = cv2.boundingRect(poly)
    pad = grow_px + 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return mask
    poly = poly - (x0, y0)
    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
    # Fill + thick round-capped outline == dilating the polygon by a disk, but much cheaper.
    cv2.fillConvexPoly(roi, poly, 255)
    cv2.polylines(roi, [poly], True, 255, thickness=2 * grow_px)
    roi = cv2.GaussianBlur(roi, (9, 9), 0)
    out = mask.copy()
    np.maximum(out[y0:y1, x0:x1], roi.astype(np.float32) * (1 / 255.0), out=out[y0:y1, x0:x1])
    return out


def fill_down(mask: np.ndarray) -> np.ndarray:
    """Extend a person mask straight down to the bottom of the frame.

    A person in a webcam view continues below whatever part is detected (a
    detected face/shoulders sits on a torso). Segmenters often lose a still
    torso or T-shirt at the bottom edge, so for *learning* everything under a
    detected part is treated as body.
    """
    return np.maximum.accumulate(mask, axis=0)


def background_mask(raw: np.ndarray, threshold: float = 0.1, margin_px: int = 20,
                    down: bool = True) -> np.ndarray:
    """uint8 mask (255) of pixels that are safely NOT the person, for learning
    the background. Deliberately conservative: anything even slightly
    person-like, everything below it (`down`, see fill_down), plus a wide
    `margin_px` (edges, hair, shadows close to the body), is excluded so the
    person never gets baked into the plate.
    """
    # Done at half resolution with half the kernel: same margin, ~4x cheaper.
    h, w = raw.shape[:2]
    small = cv2.resize(raw, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    person = (small > threshold).astype(np.uint8) * 255
    if down:
        person = fill_down(person)
    r = max(1, margin_px // 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.bitwise_not(cv2.dilate(person, k))


def difference_mask(frame: np.ndarray, plate: np.ndarray, known: np.ndarray,
                    out_size: tuple[int, int], low: float = 18.0, high: float = 40.0,
                    work_width: int = 320) -> tuple[np.ndarray, float]:
    """Soft mask (0..1, size `out_size` = (w, h)) of pixels that differ from
    the known background, plus the fraction of known pixels that changed.

    This catches the person regardless of pose or whether a detector
    recognises them (a still T-shirt, a chin, an arm in a new place, a held
    object, their shadow). Unknown pixels get 0 here; callers treat them
    separately. A per-channel gain, estimated from known pixels, compensates
    global brightness shifts such as webcam auto-exposure.
    """
    h, w = frame.shape[:2]
    sw, sh = work_width, work_width * h // w
    f = cv2.GaussianBlur(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), (5, 5), 0)
    p = cv2.GaussianBlur(cv2.resize(plate, (sw, sh), interpolation=cv2.INTER_AREA), (5, 5), 0)
    k = cv2.resize(known, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    if np.count_nonzero(k) < 100:
        return np.zeros((out_size[1], out_size[0]), np.float32), 0.0
    f = f.astype(np.float32)
    p = p.astype(np.float32)
    gain = (np.median(f[k], axis=0) + 1.0) / (np.median(p[k], axis=0) + 1.0)
    p *= np.clip(gain, 0.5, 2.0)
    d = np.abs(f - p).max(axis=2)
    m = np.clip((d - low) / (high - low), 0.0, 1.0)
    m[~k] = 0.0
    m8 = (m * 255).astype(np.uint8)
    # Close small gaps inside the changed region (uniform clothing areas).
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    changed = float(np.count_nonzero(m8[k] > 127)) / np.count_nonzero(k)
    out = cv2.resize(m8, out_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1 / 255.0)
    return out, changed


def composite(frame: np.ndarray, plate: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Blend `plate` over `frame` where `mask` is high. `mask` may be lower resolution."""
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if strength < 1.0:
        mask = mask * np.float32(strength)
    return cv2.blendLinear(plate, frame, mask, 1.0 - mask)
