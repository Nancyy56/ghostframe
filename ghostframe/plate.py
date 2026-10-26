"""Background plate: learned automatically while you're in frame, or captured
manually from an empty room (countdown + multi-frame median). Saved/loaded as
a PNG whose alpha channel marks which pixels have really been seen.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "backgrounds", "plate.png")


class BackgroundPlate:
    """Holds the background image plus a `known` mask (255 = really seen).

    Automatic learning (`learn`): pixels that have been confidently NOT the
    person for `confirm_frames` consecutive frames are blended into the plate
    (a one-frame segmentation miss must not bake part of the person in).
    Pixels confirmed for the first time are copied directly; afterwards a
    running average (`learn_rate`) slowly follows lighting changes. As the person moves, the area behind them is revealed and
    filled in. Pixels never seen (e.g. behind someone who hasn't moved) are
    guessed by inpainting in `filled()`: instantly with a cheap OpenCV fill,
    then, if a `filler` (LaMa, see fill.py) is given, replaced by a much more
    realistic fill computed in the background. Guesses are still guesses.

    Manual capture (`start_capture`/`feed`): countdown to leave the frame, then
    the per-pixel median of `num_frames` frames, which rejects noise and brief
    movement. Gives a fully known, sharp plate.
    """

    def __init__(self, path: str = DEFAULT_PATH, countdown: float = 3.0, num_frames: int = 20,
                 learn_rate: float = 0.05, inpaint_width: int = 320, inpaint_interval: float = 0.5,
                 filler=None, filler_min_coverage: float = 0.4, confirm_frames: int = 8,
                 settle_seconds: float = 2.0, seam_px: int = 12):
        self.path = path
        self.countdown = countdown
        self.num_frames = num_frames
        self.learn_rate = learn_rate
        self.inpaint_width = inpaint_width
        self.inpaint_interval = inpaint_interval
        self.image: np.ndarray | None = None     # uint8 BGR
        self.known: np.ndarray | None = None     # uint8, 255 = seen
        self._acc: np.ndarray | None = None      # float32 running average
        self.confirm_frames = confirm_frames
        self.settle_seconds = settle_seconds
        self.seam_px = seam_px
        self._streak: np.ndarray | None = None   # consecutive background frames, learning resolution
        self._filled: np.ndarray | None = None   # full-size guess for unknown pixels (Telea or LaMa)
        self._soft: np.ndarray | None = None     # float32 0..1 blend weight of the real plate
        self._soft_at = -1e9
        self._soft_dirty = True
        self._cov_ref = (-1e9, 0.0)              # (time, coverage) when coverage last grew noticeably
        self._filled_at = -1e9
        self._dirty = True
        self._start: float | None = None
        self._frames: list[np.ndarray] = []
        self.filler = filler                     # optional LamaFiller
        self.filler_min_coverage = filler_min_coverage
        self._filler_version = -1
        self._using_filler = False

    # ---- state -------------------------------------------------------------

    @property
    def coverage(self) -> float:
        """Fraction of pixels really seen (0..1)."""
        return 0.0 if self.known is None else float(np.count_nonzero(self.known)) / self.known.size

    def matches(self, frame: np.ndarray) -> bool:
        return self.image is not None and self.image.shape == frame.shape

    @property
    def fill_status(self) -> str:
        """'' (nothing unknown / no filler), 'improving' (LaMa running), 'realistic' (LaMa fill in use)."""
        if self.filler is None or self.image is None or self.coverage >= 0.999:
            return ""
        if self._using_filler:
            return "realistic"
        return "improving" if self.filler.busy else ""

    def reset(self) -> None:
        self.image = self.known = self._acc = self._filled = self._soft = self._streak = None
        self._dirty = self._soft_dirty = True
        self._cov_ref = (-1e9, 0.0)
        self._reset_filler()

    def _set(self, image: np.ndarray, known: np.ndarray) -> None:
        self.image = image
        self.known = known
        self._acc = image.astype(np.float32)
        self._filled = self._soft = self._streak = None
        self._dirty = self._soft_dirty = True
        self._cov_ref = (-1e9, 0.0)
        self._reset_filler()

    def _reset_filler(self) -> None:
        self._using_filler = False
        if self.filler is not None:
            self.filler.reset()
            self._filler_version = self.filler.version

    # ---- automatic learning -------------------------------------------------

    def learn(self, frame: np.ndarray, background: np.ndarray, now: float | None = None,
              allow_new: bool = True) -> None:
        """`background`: uint8 mask (255 = safely not the person), any resolution.
        With `now`, also kicks off a background LaMa fill when one is due, so a
        realistic fill is usually ready before invisibility is switched on.
        `allow_new=False` only refreshes already-known pixels: used when no
        person is detected in a frame, because then the detector may simply be
        failing (e.g. only a chin and T-shirt in view) and nothing new can be
        trusted to be background."""
        h, w = frame.shape[:2]
        if not self.matches(frame):
            self._set(np.zeros_like(frame), np.zeros((h, w), np.uint8))
        # Count consecutive background frames per pixel (cheap: at the mask's
        # own, low resolution) and only trust pixels that stayed background.
        if self._streak is None or self._streak.shape != background.shape[:2]:
            self._streak = np.zeros(background.shape[:2], np.uint8)
        self._streak = cv2.add(self._streak, 1, mask=background)
        self._streak[background == 0] = 0
        if not allow_new:
            # Unseen pixels must not build up confirmation in untrusted frames.
            known_small = cv2.resize(self.known, self._streak.shape[::-1], interpolation=cv2.INTER_NEAREST)
            self._streak[known_small == 0] = 0
        confirmed = np.where(self._streak >= self.confirm_frames, 255, 0).astype(np.uint8)
        if confirmed.shape != (h, w):
            # Linear + threshold gives smooth edges (nearest-neighbour upscaling
            # produced visible stair steps).
            confirmed = cv2.resize(confirmed, (w, h), interpolation=cv2.INTER_LINEAR)
            confirmed = np.where(confirmed > 127, 255, 0).astype(np.uint8)
        new = cv2.bitwise_and(confirmed, cv2.bitwise_not(self.known))
        if not allow_new:
            confirmed = cv2.bitwise_and(confirmed, self.known)
        elif cv2.countNonZero(new):
            self._acc[new > 0] = frame[new > 0]
            self.known |= new
            self._dirty = self._soft_dirty = True
        cv2.accumulateWeighted(frame, self._acc, self.learn_rate, mask=confirmed)
        self.image = cv2.convertScaleAbs(self._acc)
        if self.filler is not None and now is not None and self._settled(now):
            self.filler.maybe_start(self.image, self.known, now)

    def _settled(self, now: float) -> bool:
        """True once coverage is high enough and has stopped growing for
        `settle_seconds`, so LaMa isn't run on a half-learned plate (which
        produced a dark, wrong fill)."""
        cov = self.coverage
        t_ref, cov_ref = self._cov_ref
        if cov - cov_ref > 0.01:
            self._cov_ref = (now, cov)
            return False
        return cov >= self.filler_min_coverage and now - t_ref >= self.settle_seconds

    def filled(self, now: float) -> np.ndarray | None:
        """The plate with never-seen pixels inpainted. Recomputed at most every
        `inpaint_interval` seconds, and only while there are unknown pixels."""
        if self.image is None:
            return None
        if self.coverage >= 0.999:
            return self.image
        if self.filler is not None and self.filler.version != self._filler_version:
            self._filler_version = self.filler.version
            result = self.filler.result()
            if result is not None and result.shape == self.image.shape:
                self._filled = result.copy()
                self._using_filler = True
        if not self._using_filler and (
                self._filled is None or (self._dirty and now - self._filled_at >= self.inpaint_interval)):
            h, w = self.image.shape[:2]
            sw, sh = self.inpaint_width, self.inpaint_width * h // w
            small = cv2.resize(self.image, (sw, sh), interpolation=cv2.INTER_AREA)
            hole = cv2.resize(cv2.bitwise_not(self.known), (sw, sh), interpolation=cv2.INTER_NEAREST)
            hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))  # cover resize fringe
            guess = cv2.resize(cv2.inpaint(small, hole, 3, cv2.INPAINT_TELEA), (w, h),
                               interpolation=cv2.INTER_LINEAR)
            self._filled = guess
            self._filled_at = now
            self._dirty = False
        # Blend real plate and guess with a soft seam instead of a hard cut-out.
        # The weight is the known mask eroded then blurred by `seam_px`, so it is
        # exactly 0 in unknown pixels (whose plate value is meaningless) and
        # ramps up to 1 just inside the known area.
        if self._soft is None or (self._soft_dirty and now - self._soft_at >= self.inpaint_interval):
            r = self.seam_px
            e = r + 2  # erode slightly past the blur's reach so holes get exactly 0
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2)
            soft = cv2.GaussianBlur(cv2.erode(self.known, k), (2 * r + 1, 2 * r + 1), 0)
            self._soft = soft.astype(np.float32) * (1 / 255.0)
            self._soft_at = now
            self._soft_dirty = False
        return cv2.blendLinear(self.image, self._filled, self._soft, 1.0 - self._soft)

    # ---- manual capture ----------------------------------------------------

    @property
    def capturing(self) -> bool:
        return self._start is not None

    def start_capture(self, now: float) -> None:
        self._start = now
        self._frames = []

    def cancel(self) -> None:
        self._start = None
        self._frames = []

    def seconds_left(self, now: float) -> float:
        return 0.0 if self._start is None else max(0.0, self._start + self.countdown - now)

    def capture_progress(self) -> float:
        return len(self._frames) / self.num_frames

    def feed(self, frame: np.ndarray, now: float) -> bool:
        """Feed live frames while capturing. Returns True when the plate is done."""
        if self._start is None or now < self._start + self.countdown:
            return False
        self._frames.append(frame.copy())
        if len(self._frames) < self.num_frames:
            return False
        image = np.median(np.stack(self._frames), axis=0).astype(np.uint8)
        self._set(image, np.full(image.shape[:2], 255, np.uint8))
        self.cancel()
        return True

    # ---- persistence ---------------------------------------------------------

    def save(self) -> None:
        if self.image is None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        cv2.imwrite(self.path, np.dstack([self.image, self.known]))
        # Also keep the realistic (LaMa) fill: people tend to sit in the same
        # spot, so next session it beats the basic fill until a fresh one is ready.
        if self._using_filler and self._filled is not None:
            cv2.imwrite(self._fill_path, self._filled)
        elif os.path.isfile(self._fill_path):
            os.remove(self._fill_path)

    @property
    def _fill_path(self) -> str:
        root, ext = os.path.splitext(self.path)
        return f"{root}_fill{ext}"

    def load(self) -> bool:
        if not os.path.isfile(self.path):
            return False
        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            return False
        if img.shape[2] == 4:
            self._set(np.ascontiguousarray(img[..., :3]), np.where(img[..., 3] > 127, 255, 0).astype(np.uint8))
            fill = cv2.imread(self._fill_path, cv2.IMREAD_COLOR) if os.path.isfile(self._fill_path) else None
            if fill is not None and fill.shape == self.image.shape:
                self._filled = fill
                self._using_filler = True  # until a fresh LaMa result replaces it
        else:  # plain 3-channel image (e.g. from stage 3): fully known
            self._set(img, np.full(img.shape[:2], 255, np.uint8))
        return True


class SceneChangeMonitor:
    """Detects that the camera moved or the lighting changed, so the learned
    background no longer applies and must be relearned automatically.

    Fed with the fraction of known *room* pixels (those the segmenter does not
    think are the person) that differ from the plate. A person leaning towards
    the camera is excluded by the segmenter; a moved camera changes the room
    itself. Fires once the fraction stays above `threshold` for `hold` seconds.
    """

    def __init__(self, threshold: float = 0.5, hold: float = 2.0):
        self.threshold = threshold
        self.hold = hold
        self._since: float | None = None

    def update(self, changed_fraction: float, now: float) -> bool:
        if changed_fraction <= self.threshold:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since >= self.hold:
            self._since = None
            return True
        return False

    @property
    def changing(self) -> bool:
        return self._since is not None
