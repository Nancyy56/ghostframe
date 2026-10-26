"""Tests for detector-independent person removal: difference mask, fill-down,
and the learning gate (no camera or models needed).

    ./.venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghostframe.plate import BackgroundPlate, SceneChangeMonitor  # noqa: E402
from ghostframe.segment import background_mask, difference_mask, fill_down  # noqa: E402

H, W = 180, 320


def room():
    rng = np.random.default_rng(1)
    base = rng.integers(40, 160, (H // 10, W // 10, 3)).astype(np.uint8)
    return np.kron(base, np.ones((10, 10, 1), np.uint8))  # blocky texture


class DifferenceMaskTest(unittest.TestCase):
    def setUp(self):
        self.plate = room()
        self.known = np.full((H, W), 255, np.uint8)

    def test_unchanged_room_is_clear(self):
        m, changed = difference_mask(self.plate, self.plate, self.known, (W, H))
        self.assertEqual(float(m.max()), 0.0)
        self.assertEqual(changed, 0.0)

    def test_new_object_is_detected_regardless_of_what_it_is(self):
        frame = self.plate.copy()
        frame[60:140, 100:180] = (230, 230, 230)   # e.g. a T-shirt the segmenter misses
        m, changed = difference_mask(frame, self.plate, self.known, (W, H))
        self.assertGreater(m[100, 140], 0.9)
        self.assertLess(m[20, 20], 0.1)
        self.assertGreater(changed, 0.05)
        self.assertLess(changed, 0.3)

    def test_auto_exposure_shift_is_compensated(self):
        frame = np.clip(self.plate.astype(int) * 1.3, 0, 255).astype(np.uint8)
        m, changed = difference_mask(frame, self.plate, self.known, (W, H))
        self.assertLess(changed, 0.02)

    def test_unknown_pixels_are_ignored(self):
        known = self.known.copy()
        known[:, 160:] = 0
        frame = self.plate.copy()
        frame[:, 200:] = 255
        m, _ = difference_mask(frame, self.plate, known, (W, H))
        self.assertEqual(float(m[:, 220:].max()), 0.0)


class FillDownTest(unittest.TestCase):
    def test_everything_below_detection_is_body(self):
        m = np.zeros((10, 6), np.uint8)
        m[3, 2] = 255                     # e.g. a detected chin
        out = fill_down(m)
        self.assertTrue(np.all(out[3:, 2] == 255))
        self.assertTrue(np.all(out[:3, 2] == 0))
        self.assertTrue(np.all(out[:, 0] == 0))

    def test_background_mask_excludes_area_below_person(self):
        raw = np.zeros((360, 640), np.float32)
        raw[50:120, 280:360] = 1.0        # only the head is detected
        bg = background_mask(raw)
        self.assertEqual(bg[170, 160], 0)  # (half res) torso area below the head: not learnable
        self.assertEqual(bg[20, 20], 255)  # far away: learnable
        bg2 = background_mask(raw, down=False)
        self.assertEqual(bg2[170, 160], 255)


class LearnGateTest(unittest.TestCase):
    def test_no_new_pixels_when_no_person_detected(self):
        plate = BackgroundPlate(confirm_frames=2)
        frame = room()
        bg = np.full((H, W), 255, np.uint8)
        bg[:, 200:] = 0
        for _ in range(3):
            plate.learn(frame, bg, allow_new=True)
        known_before = plate.coverage
        bg_all = np.full((H, W), 255, np.uint8)       # detector sees nobody...
        for _ in range(10):
            plate.learn(frame, bg_all, allow_new=False)  # ...so nothing new is trusted
        self.assertEqual(plate.coverage, known_before)
        plate.learn(frame, bg_all, allow_new=True)    # trusted again: needs fresh confirmation
        self.assertEqual(plate.coverage, known_before)
        plate.learn(frame, bg_all, allow_new=True)
        self.assertGreater(plate.coverage, known_before)

    def test_known_pixels_still_follow_lighting_without_person(self):
        plate = BackgroundPlate(confirm_frames=1, learn_rate=0.5)
        frame = room()
        bg = np.full((H, W), 255, np.uint8)
        plate.learn(frame, bg)
        brighter = np.clip(frame.astype(int) + 30, 0, 255).astype(np.uint8)
        for _ in range(10):
            plate.learn(brighter, bg, allow_new=False)
        self.assertLessEqual(np.abs(plate.image.astype(int) - brighter).max(), 1)


class SceneChangeMonitorTest(unittest.TestCase):
    def test_fires_only_after_sustained_change(self):
        m = SceneChangeMonitor(threshold=0.5, hold=2.0)
        self.assertFalse(m.update(0.9, 0.0))
        self.assertFalse(m.update(0.9, 1.0))
        self.assertTrue(m.update(0.9, 2.1))
        self.assertFalse(m.update(0.9, 2.2))   # re-armed after firing

    def test_brief_change_is_ignored(self):
        m = SceneChangeMonitor(threshold=0.5, hold=2.0)
        m.update(0.9, 0.0)
        m.update(0.1, 1.0)                     # back to normal: timer resets
        self.assertFalse(m.update(0.9, 2.5))
        self.assertFalse(m.changing is False)


if __name__ == "__main__":
    unittest.main()
