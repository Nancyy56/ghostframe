"""Unit tests for the PalmToggle debounce/cooldown state machine (no camera needed).

    ./.venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghostframe.gesture import PalmToggle  # noqa: E402

DT = 1 / 20  # 20 FPS observations


def feed(toggle, visible, start, duration):
    """Feed a constant observation for `duration` seconds; return (toggle count, end time)."""
    n, t = 0, start
    while t < start + duration - 1e-9:
        n += toggle.update(visible, t)
        t += DT
    return n, t


class PalmToggleTest(unittest.TestCase):
    def test_short_flash_does_not_toggle(self):
        tg = PalmToggle(hold_time=0.5)
        n, t = feed(tg, True, 0.0, 0.3)
        n2, _ = feed(tg, False, t, 1.0)
        self.assertEqual(n + n2, 0)
        self.assertFalse(tg.active)

    def test_hold_toggles_once(self):
        tg = PalmToggle(hold_time=0.5, cooldown=1.5)
        n, _ = feed(tg, True, 0.0, 5.0)  # holding for a long time
        self.assertEqual(n, 1)
        self.assertTrue(tg.active)

    def test_must_release_before_next_toggle(self):
        tg = PalmToggle(hold_time=0.5, cooldown=1.5)
        n1, t = feed(tg, True, 0.0, 1.0)
        n2, t = feed(tg, False, t, 2.0)   # lower hand, cooldown expires
        n3, t = feed(tg, True, t, 1.0)    # raise again
        self.assertEqual((n1, n2, n3), (1, 0, 1))
        self.assertFalse(tg.active)

    def test_cooldown_blocks_quick_retoggle(self):
        tg = PalmToggle(hold_time=0.5, cooldown=1.5)
        n1, t = feed(tg, True, 0.0, 0.6)  # toggles at 0.5s
        n2, t = feed(tg, False, t, 0.4)   # release
        n3, t = feed(tg, True, t, 0.6)    # still within 1.5s cooldown
        self.assertEqual((n1, n2, n3), (1, 0, 0))

    def test_brief_dropouts_do_not_reset_hold(self):
        tg = PalmToggle(hold_time=0.5, gap_tolerance=0.25)
        t, n = 0.0, 0
        for i in range(15):  # every 3rd observation missing
            n += tg.update(i % 3 != 2, t)
            t += DT
        self.assertEqual(n, 1)

    def test_long_gap_resets_hold(self):
        tg = PalmToggle(hold_time=0.5, gap_tolerance=0.25)
        n1, t = feed(tg, True, 0.0, 0.35)
        n2, t = feed(tg, False, t, 0.4)
        n3, t = feed(tg, True, t, 0.35)
        self.assertEqual(n1 + n2 + n3, 0)

    def test_keyboard_toggle_and_cooldown(self):
        tg = PalmToggle(hold_time=0.5, cooldown=1.5)
        tg.toggle(0.0)
        self.assertTrue(tg.active)
        n, _ = feed(tg, True, 0.1, 1.0)   # palm during cooldown: ignored
        self.assertEqual(n, 0)
        self.assertTrue(tg.active)

    def test_progress(self):
        tg = PalmToggle(hold_time=0.5)
        tg.update(True, 0.0)
        tg.update(True, 0.25)
        self.assertAlmostEqual(tg.progress(0.25), 0.5)


if __name__ == "__main__":
    unittest.main()
