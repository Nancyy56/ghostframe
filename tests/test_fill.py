"""Tests for the LaMa background filler (skipped if onnxruntime or the model is missing).

    ./.venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghostframe.plate import BackgroundPlate  # noqa: E402

try:
    from ghostframe.fill import LamaFiller
    FILLER = LamaFiller()
except Exception as e:  # pragma: no cover
    FILLER, SKIP_REASON = None, str(e)
else:
    SKIP_REASON = ""


def wait_idle(filler, timeout=60.0):
    end = time.monotonic() + timeout
    while filler.busy and time.monotonic() < end:
        time.sleep(0.01)


@unittest.skipIf(FILLER is None, f"LaMa unavailable: {SKIP_REASON}")
class LamaFillerTest(unittest.TestCase):
    def setUp(self):
        FILLER.reset()
        self._real = FILLER.inpaint

    def tearDown(self):
        FILLER.inpaint = self._real
        wait_idle(FILLER)
        FILLER.reset()

    def test_real_fill_continues_surroundings(self):
        # Left half dark, right half light; hole straddles the edge.
        img = np.zeros((180, 320, 3), np.uint8)
        img[:, 160:] = 200
        known = np.full((180, 320), 255, np.uint8)
        known[40:140, 120:200] = 0
        out = FILLER.inpaint(img, known)
        self.assertEqual(out.shape, img.shape)
        self.assertLess(out[90, 125].mean(), 60)      # dark side stays dark
        self.assertGreater(out[90, 195].mean(), 140)  # light side stays light

    def test_schedule_runs_once_then_only_when_hole_shrinks(self):
        calls = []
        FILLER.inpaint = lambda image, known: calls.append(1) or image.copy()
        img = np.zeros((36, 64, 3), np.uint8)
        known = np.full((36, 64), 255, np.uint8)
        known[:, :32] = 0                       # half unknown
        FILLER.maybe_start(img, known, now=0.0)
        wait_idle(FILLER)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(FILLER.result())
        FILLER.maybe_start(img, known, now=20.0)  # same hole, not refresh time yet
        wait_idle(FILLER)
        self.assertEqual(len(calls), 1)
        known[:, :16] = 255                      # hole halved
        FILLER.maybe_start(img, known, now=5.0)   # too soon (< min_interval)
        wait_idle(FILLER)
        self.assertEqual(len(calls), 1)
        FILLER.maybe_start(img, known, now=21.0)
        wait_idle(FILLER)
        self.assertEqual(len(calls), 2)
        FILLER.maybe_start(img, known, now=21.0 + FILLER.refresh_interval)  # periodic refresh
        wait_idle(FILLER)
        self.assertEqual(len(calls), 3)

    def test_nothing_unknown_means_no_job(self):
        calls = []
        FILLER.inpaint = lambda image, known: calls.append(1) or image.copy()
        FILLER.maybe_start(np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 255, np.uint8), now=0.0)
        wait_idle(FILLER)
        self.assertEqual(calls, [])

    def test_reset_discards_running_job(self):
        def slow(image, known):
            time.sleep(0.2)
            return image.copy()
        FILLER.inpaint = slow
        known = np.zeros((8, 8), np.uint8)
        FILLER.maybe_start(np.zeros((8, 8, 3), np.uint8), known, now=0.0)
        FILLER.reset()
        wait_idle(FILLER)
        self.assertIsNone(FILLER.result())

    def test_plate_uses_filler_result_for_holes(self):
        FILLER.inpaint = lambda image, known: np.full_like(image, 123)
        plate = BackgroundPlate(filler=FILLER, filler_min_coverage=0.4, seam_px=3,
                                settle_seconds=2.0, confirm_frames=2)
        frame = np.full((36, 64, 3), 50, np.uint8)
        bg = np.full((36, 64), 255, np.uint8)
        bg[:, 40:] = 0                             # person on the right
        for i in range(10):                        # 0.0 .. 0.9 s: not settled yet
            plate.learn(frame, bg, now=i * 0.1)
        wait_idle(FILLER)
        self.assertIsNone(FILLER.result())         # no LaMa on a still-changing plate
        plate.learn(frame, bg, now=3.0)            # coverage stable for >2 s: starts
        wait_idle(FILLER)
        out = plate.filled(now=3.1)
        self.assertEqual(plate.fill_status, "realistic")
        self.assertTrue(np.all(out[:, :30] == 50))   # seen pixels: real background
        self.assertTrue(np.all(out[:, 40:] == 123))  # holes: filler result


if __name__ == "__main__":
    unittest.main()
