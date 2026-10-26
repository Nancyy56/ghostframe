"""Unit tests for mask refinement, compositing and plate capture (no camera or models needed).

    ./.venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghostframe.plate import BackgroundPlate  # noqa: E402
from ghostframe.segment import MaskRefiner, add_hand, composite  # noqa: E402


class CompositeTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((72, 128, 3), 10, np.uint8)
        self.plate = np.full((72, 128, 3), 200, np.uint8)

    def test_full_and_empty_mask(self):
        ones = np.ones((36, 64), np.float32)  # lower-res mask gets upscaled
        np.testing.assert_array_equal(composite(self.frame, self.plate, ones), self.plate)
        np.testing.assert_array_equal(composite(self.frame, self.plate, ones * 0), self.frame)

    def test_strength_blends(self):
        ones = np.ones((72, 128), np.float32)
        out = composite(self.frame, self.plate, ones, strength=0.5)
        self.assertTrue(np.all(np.abs(out.astype(int) - 105) <= 1))


class MaskRefinerTest(unittest.TestCase):
    def test_grows_instantly_shrinks_slowly(self):
        ref = MaskRefiner(low=0.2, high=0.6, decay=0.8, dilate_px=0, feather_px=0)
        full = np.ones((10, 10), np.float32)
        self.assertAlmostEqual(float(ref.refine(full).mean()), 1.0, places=2)
        # Person vanishes from the raw mask: refined mask fades over several frames.
        vals = [float(ref.refine(np.zeros((10, 10), np.float32)).mean()) for _ in range(3)]
        self.assertAlmostEqual(vals[0], 0.8, places=2)
        self.assertAlmostEqual(vals[1], 0.64, places=2)
        self.assertLess(vals[2], vals[1])

    def test_soft_threshold(self):
        ref = MaskRefiner(low=0.2, high=0.6, decay=0.0, dilate_px=0, feather_px=0)
        out = ref.refine(np.array([[0.1, 0.4, 0.9]], np.float32))
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=0.01)

    def test_dilation_grows_mask(self):
        raw = np.zeros((40, 40), np.float32)
        raw[18:22, 18:22] = 1
        out = MaskRefiner(decay=0.0, dilate_px=4, feather_px=0).refine(raw)
        self.assertGreater((out > 0.5).sum(), (raw > 0.5).sum() * 3)

    def test_reset_clears_history(self):
        ref = MaskRefiner(decay=0.9, dilate_px=0, feather_px=0)
        ref.refine(np.ones((5, 5), np.float32))
        ref.reset()
        self.assertEqual(float(ref.refine(np.zeros((5, 5), np.float32)).max()), 0.0)


class AddHandTest(unittest.TestCase):
    def test_hand_area_covered_and_rest_untouched(self):
        mask = np.zeros((360, 640), np.float32)
        # Fake landmarks: a small cluster around (0.5, 0.5); wrist (0) and middle knuckle (9) apart.
        hand = np.full((21, 2), 0.5, np.float32)
        hand[0] = (0.5, 0.6)
        hand[9] = (0.5, 0.45)
        out = add_hand(mask, hand)
        self.assertGreater(out[180, 320], 0.95)        # inside the hand
        self.assertEqual(out[10, 10], 0.0)             # far away: unchanged
        self.assertEqual(mask.max(), 0.0)              # input not modified

    def test_hand_partly_off_screen(self):
        hand = np.full((21, 2), 0.99, np.float32)
        hand[0] = (1.1, 1.2)
        out = add_hand(np.zeros((36, 64), np.float32), hand)
        self.assertEqual(out.shape, (36, 64))


class PlateTest(unittest.TestCase):
    def test_countdown_then_median(self):
        plate = BackgroundPlate(path=os.path.join(tempfile.mkdtemp(), "p.png"),
                                countdown=1.0, num_frames=5)
        plate.start_capture(0.0)
        bg = np.full((4, 4, 3), 100, np.uint8)
        self.assertFalse(plate.feed(bg + 50, 0.5))  # during countdown: ignored
        frames = [bg, bg, bg + 1, bg, np.full_like(bg, 255)]  # one outlier frame
        done = [plate.feed(f, 1.0 + i * 0.05) for i, f in enumerate(frames)]
        self.assertEqual(done, [False] * 4 + [True])
        self.assertFalse(plate.capturing)
        np.testing.assert_array_equal(plate.image, bg)  # median rejects the outlier

    def test_save_and_load_keeps_known_mask(self):
        path = os.path.join(tempfile.mkdtemp(), "sub", "p.png")
        a = BackgroundPlate(path=path)
        frame = np.random.randint(0, 255, (8, 8, 3), np.uint8)
        bg = np.zeros((8, 8), np.uint8)
        bg[:, :5] = 255
        for _ in range(a.confirm_frames):
            a.learn(frame, bg)
        a.save()
        b = BackgroundPlate(path=path)
        self.assertTrue(b.load())
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.known, b.known)
        self.assertAlmostEqual(b.coverage, 5 / 8)
        self.assertFalse(BackgroundPlate(path=path + ".missing").load())

    def test_realistic_fill_is_saved_and_reused(self):
        path = os.path.join(tempfile.mkdtemp(), "p.png")
        a = BackgroundPlate(path=path, seam_px=1)
        frame = np.full((20, 20, 3), 40, np.uint8)
        bg = np.zeros((20, 20), np.uint8)
        bg[:, :10] = 255
        for _ in range(a.confirm_frames):
            a.learn(frame, bg)
        a._filled = np.full_like(frame, 99)   # pretend LaMa produced this
        a._using_filler = True
        a.save()
        b = BackgroundPlate(path=path, seam_px=1)
        self.assertTrue(b.load())
        out = b.filled(now=0.0)
        self.assertTrue(np.all(out[:, 15:] == 99))   # hole uses the saved realistic fill
        a._using_filler = False
        a.save()                                     # no realistic fill any more: file removed
        self.assertFalse(os.path.exists(path.replace(".png", "_fill.png")))

    def test_load_plain_rgb_is_fully_known(self):
        import cv2
        path = os.path.join(tempfile.mkdtemp(), "p.png")
        cv2.imwrite(path, np.full((6, 6, 3), 77, np.uint8))
        b = BackgroundPlate(path=path)
        self.assertTrue(b.load())
        self.assertEqual(b.coverage, 1.0)


class LearnTest(unittest.TestCase):
    """A 'person' (value 255 block) moves across a static scene."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.scene = rng.integers(0, 200, (60, 100, 3), dtype=np.uint8)

    def frame_with_person(self, x):
        f = self.scene.copy()
        f[10:50, x:x + 30] = 255
        bg = np.full((60, 100), 255, np.uint8)
        bg[5:55, max(0, x - 5):x + 35] = 0  # person + margin is excluded
        return f, bg

    def test_static_person_leaves_hole_and_never_leaks(self):
        plate = BackgroundPlate(learn_rate=0.5)
        for _ in range(10):
            plate.learn(*self.frame_with_person(30))
        self.assertLess(plate.coverage, 0.8)
        known = plate.known > 0
        np.testing.assert_array_equal(plate.image[known], self.scene[known])
        self.assertFalse(np.any(plate.image == 255))  # person never baked in

    def learn_n(self, plate, x, n=None):
        for _ in range(n or plate.confirm_frames):
            plate.learn(*self.frame_with_person(x))

    def test_moving_person_reveals_background(self):
        plate = BackgroundPlate(learn_rate=0.5)
        self.learn_n(plate, 5)
        c0 = plate.coverage
        self.learn_n(plate, 65)
        self.assertGreater(plate.coverage, 0.99)
        self.assertLess(c0, 0.7)
        np.testing.assert_array_equal(plate.image, self.scene)

    def test_filled_has_no_holes(self):
        plate = BackgroundPlate(seam_px=3)
        self.learn_n(plate, 30)
        out = plate.filled(now=0.0)
        self.assertEqual(out.shape, self.scene.shape)
        known = plate.known > 0
        # Away from the seam, known pixels are exactly the real background.
        np.testing.assert_array_equal(out[:, :15], self.scene[:, :15])
        np.testing.assert_array_equal(out[:, 80:], self.scene[:, 80:])
        self.assertGreater(out[~known].mean(), 10)  # holes guessed, not left black

    def test_flickering_pixels_are_never_learned(self):
        """A region that is 'background' only every other frame (segmentation
        flicker on the person's edge) must not be learned."""
        plate = BackgroundPlate(confirm_frames=4)
        steady = self.frame_with_person(30)
        flicker_bg = steady[1].copy()
        flicker_bg[5:55, 25:65] = 255      # segmenter briefly misses the person
        for i in range(20):
            plate.learn(steady[0], flicker_bg if i % 2 else steady[1])
        self.assertFalse(np.any(plate.image == 255))
        self.assertFalse(np.any(plate.known[10:50, 30:60]))

    def test_soft_seam(self):
        plate = BackgroundPlate(seam_px=6)
        self.learn_n(plate, 30)
        plate.filled(now=0.0)
        w = plate._soft
        hole = plate.known == 0
        self.assertEqual(float(w[hole].max()), 0.0)        # unknown: never uses plate pixels
        self.assertAlmostEqual(float(w[30, 2]), 1.0, 3)    # far inside known: real pixels
        row = w[30, 5:25]
        self.assertTrue(np.any((row > 0.05) & (row < 0.95)))  # gradual ramp, not a hard step

    def test_running_average_follows_lighting(self):
        plate = BackgroundPlate(learn_rate=0.5, confirm_frames=1)
        bg = np.full((60, 100), 255, np.uint8)
        plate.learn(self.scene, bg)
        brighter = np.clip(self.scene.astype(int) + 40, 0, 255).astype(np.uint8)
        for _ in range(10):
            plate.learn(brighter, bg)
        self.assertLessEqual(np.abs(plate.image.astype(int) - brighter).max(), 1)

    def test_reset(self):
        plate = BackgroundPlate()
        plate.learn(*self.frame_with_person(30))
        plate.reset()
        self.assertIsNone(plate.image)
        self.assertEqual(plate.coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
