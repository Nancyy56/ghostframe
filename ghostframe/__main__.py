"""GhostFrame entry point.

Stage 1: webcam capture + preview window + FPS overlay + camera error handling.
Stage 2: open-palm gesture toggle (hold + cooldown) with keyboard fallback.
Stage 3: background plate capture, person segmentation and compositing.
         The background is learned automatically while you're in frame; B
         captures an empty room manually for the cleanest result.

    python -m ghostframe                      # /dev/video0
    python -m ghostframe --source /dev/video2
    python -m ghostframe --source clip.mp4    # test without a webcam
    python -m ghostframe --no-window --max-frames 300   # headless benchmark

Keys: Space / I = toggle invisibility, B = capture empty-room background,
R = reset learned background, G = gestures on/off, M = show mask, Q / Esc = quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

# OpenCV's pip wheel bundles Qt5 without the Wayland plugin or fonts: use
# XWayland directly and the system fonts to avoid startup warnings.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # desktops often set "wayland;xcb" globally
if os.path.isdir("/usr/share/fonts/TTF"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/TTF")

import cv2
import numpy as np

from .camera import Camera
from .fps import FPSMeter
from .gesture import GestureDetector, PalmToggle
from .plate import DEFAULT_PATH as DEFAULT_PLATE, BackgroundPlate, SceneChangeMonitor
from .segment import (MaskRefiner, PersonSegmenter, add_hand, background_mask, composite,
                      difference_mask)

WINDOW = "GhostFrame"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, GREY, AMBER = (0, 255, 0), (0, 0, 255), (255, 255, 255), (170, 170, 170), (0, 191, 255)
FADE_SECONDS = 0.3        # ghost fade in/out duration
HAND_MAX_AGE = 0.2        # use hand landmarks for the mask only if this fresh
PLATE_PERSON_WARN = 0.01  # warn if >1% of the plate looks like a person
MIN_PERSON = 0.02         # segmenter must see >=2% person for new background to be learned
SCENE_CHANGED = 0.5       # >50% of the room differs: lighting/camera change, not a person


def parse_source(s: str) -> int | str:
    return int(s) if s.isdigit() else s


def text(img: np.ndarray, s: str, org: tuple[int, int], color=WHITE, scale: float = 0.7,
         thickness: int = 2) -> None:
    """Text on a dark box so it stays readable on any background.

    (A thicker black outline pass doesn't work with OpenCV 5's font: stroke
    width changes glyph advance, so the outline drifts past the text.)
    """
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def centered(img: np.ndarray, s: str, y: int, color=WHITE, scale: float = 1.0, thickness: int = 2) -> None:
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thickness)
    text(img, s, ((img.shape[1] - tw) // 2, y), color, scale, thickness)


def placeholder(width: int, height: int, message: str) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    text(img, "No camera signal", (40, height // 2 - 20), RED, 1.2)
    text(img, message, (40, height // 2 + 25), GREY, 0.6, 1)
    text(img, "Retrying... press Q or Esc to quit", (40, height // 2 + 60), GREY, 0.6, 1)
    return img


def draw_hud(frame: np.ndarray, fps: float, toggle: PalmToggle, gestures_on: bool,
             label: str, score: float, now: float, plate_status: tuple[str, tuple] | None = None) -> None:
    h, w = frame.shape[:2]
    text(frame, f"{fps:5.1f} FPS", (12, 30), GREEN)

    # Invisibility status (top-right).
    status = "GHOST: ON" if toggle.active else "GHOST: OFF"
    (tw, _), _ = cv2.getTextSize(status, FONT, 0.9, 2)
    text(frame, status, (w - tw - 16, 34), RED if toggle.active else GREEN, 0.9)
    if plate_status:
        msg, color = plate_status
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.55, 1)
        text(frame, msg, (w - tw - 16, 64), color, 0.55, 1)

    # Gesture line + hold progress bar.
    if not gestures_on:
        text(frame, "Gestures: off (G to enable)", (12, 62), GREY, 0.6, 1)
    else:
        seen = f"{label} {score:.2f}" if label else "no hand"
        text(frame, f"Gesture: {seen}", (12, 62), WHITE, 0.6, 1)
        if toggle.cooling_down(now):
            text(frame, "cooldown - lower your hand", (12, 88), AMBER, 0.6, 1)
        else:
            p = toggle.progress(now)
            if p > 0:
                cv2.rectangle(frame, (12, 76), (212, 90), (0, 0, 0), -1)
                cv2.rectangle(frame, (12, 76), (12 + int(200 * p), 90), GREEN, -1)

    text(frame, "Palm (hold) / Space: toggle   B: empty-room capture   R: relearn   G: gestures   M: mask   Q: quit",
         (12, h - 14), GREY, 0.5, 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ghostframe")
    p.add_argument("--source", default="0", help="device index, /dev/videoN, or video file")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-mirror", action="store_true", help="don't flip the preview horizontally")
    p.add_argument("--no-gestures", action="store_true", help="start with gesture detection off")
    p.add_argument("--hold", type=float, default=0.5, help="seconds to hold an open palm to toggle")
    p.add_argument("--cooldown", type=float, default=1.5, help="seconds before another toggle")
    p.add_argument("--plate", default=DEFAULT_PLATE, help="background plate image (loaded/saved)")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds to leave the frame before capture")
    p.add_argument("--no-learn", action="store_true",
                   help="don't learn the background automatically (use B to capture it)")
    p.add_argument("--no-lama", action="store_true",
                   help="don't use the LaMa model to fill never-seen background (faster, blurrier)")
    p.add_argument("--min-coverage", type=float, default=0.25,
                   help="fraction of the background that must have been seen before invisibility works")
    p.add_argument("--segmenter", choices=["landscape", "square"], default="landscape")
    p.add_argument("--dilate", type=int, default=6, help="grow the mask by N px (at 640 px width)")
    p.add_argument("--feather", type=int, default=7, help="edge blur radius in px (at 640 px width)")
    p.add_argument("--decay", type=float, default=0.8,
                   help="per-frame mask shrink factor; higher = steadier but laggier edges")
    p.add_argument("--ghost-on", action="store_true", help="start with invisibility on")
    p.add_argument("--record", metavar="FILE", help="also save the output video (no HUD) to FILE (.mp4)")
    p.add_argument("--no-window", action="store_true", help="headless (for testing/benchmarking)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    args = p.parse_args(argv)

    cam = Camera(parse_source(args.source), args.width, args.height, args.fps)
    if cam.open():
        print(f"[ghostframe] camera opened: {cam.describe()}")
    else:
        print(f"[ghostframe] warning: {cam.last_error}; will keep retrying", file=sys.stderr)

    detector: GestureDetector | None = None
    try:
        detector = GestureDetector()
    except Exception as e:  # missing model, MediaPipe init failure: keyboard still works
        print(f"[ghostframe] warning: gesture detection unavailable ({e}); "
              "use Space to toggle", file=sys.stderr)
    gestures_on = detector is not None and not args.no_gestures
    toggle = PalmToggle(hold_time=args.hold, cooldown=args.cooldown)
    if args.ghost_on:
        toggle.active = True

    try:
        segmenter = PersonSegmenter(args.segmenter)
    except Exception as e:
        print(f"[ghostframe] error: segmentation unavailable ({e})", file=sys.stderr)
        return 1
    refiner = MaskRefiner(dilate_px=args.dilate, feather_px=args.feather, decay=args.decay)

    filler = None
    if not args.no_lama and not args.no_learn:
        try:
            from .fill import LamaFiller
            filler = LamaFiller()
        except Exception as e:  # onnxruntime or model missing: fall back to the basic fill
            print(f"[ghostframe] warning: realistic background fill unavailable ({e}); "
                  "using basic fill", file=sys.stderr)
    plate = BackgroundPlate(args.plate, countdown=args.countdown, filler=filler)
    if plate.load():
        print(f"[ghostframe] loaded background plate {args.plate} ({plate.coverage:.0%} known)")
    learn = not args.no_learn
    scene_monitor = SceneChangeMonitor()
    plate_warning = ""

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.width, args.height)

    meter = FPSMeter()
    frames = 0
    ghost_frames = 0
    effect_frames = 0
    effect_ms = 0.0
    strength = 1.0 if toggle.active else 0.0  # fade level of the effect, 0..1
    show_mask = False
    writer = None
    last_fill = ""
    scene_changed = False
    last_mask: np.ndarray | None = None
    last_tick = time.monotonic()
    seen_labels: Counter[str] = Counter()
    t0 = time.perf_counter()
    try:
        while True:
            frame = cam.read()
            now = time.monotonic()
            dt, last_tick = now - last_tick, now
            if frame is None:
                if args.no_window:
                    # Headless: nothing to show, so fail fast instead of retrying forever.
                    print(f"[ghostframe] error: {cam.last_error}", file=sys.stderr)
                    return 1
                view = placeholder(args.width, args.height, cam.last_error)
                time.sleep(0.05)
            else:
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                frames += 1

                # --- Gestures ---
                label, score, hand = "", 0.0, None
                if gestures_on:
                    detector.submit(frame, now)
                    reading = detector.latest()
                    if now - reading.timestamp <= 0.3:
                        label, score = reading.label, reading.score
                    if now - reading.timestamp <= HAND_MAX_AGE:
                        hand = reading.hand
                    seen_labels[label or "no hand"] += 1
                    if toggle.update(detector.palm_visible(now), now):
                        print(f"[ghostframe] gesture toggle -> {'ON' if toggle.active else 'OFF'}")

                # --- Background plate capture ---
                if plate.capturing and plate.feed(frame, now):
                    plate.save()
                    person = float((segmenter.segment(plate.image, now) > 0.5).mean())
                    plate_warning = (f"person detected in background ({person:.0%}) - press B to recapture"
                                     if person > PLATE_PERSON_WARN else "")
                    print(f"[ghostframe] background captured and saved to {plate.path}"
                          + (f"; warning: {plate_warning}" if plate_warning else ""))
                    refiner.reset()

                # --- Ghost effect (fades in/out over FADE_SECONDS) ---
                plate_ok = plate.matches(frame) and plate.coverage >= args.min_coverage
                target = 1.0 if (toggle.active and plate_ok and not plate.capturing) else 0.0
                step = dt / FADE_SECONDS
                strength = min(target, strength + step) if target > strength else max(target, strength - step)

                out = frame
                learning = learn and not plate.capturing
                if strength > 0 or show_mask or learning:
                    t = time.perf_counter()
                    raw = segmenter.segment(frame, now)
                    # Only the segmenter's own evidence unlocks learning new background; a
                    # detected hand alone must not (the rest of the person would be learned).
                    person_seen = float(np.count_nonzero(raw > 0.5)) / raw.size >= MIN_PERSON
                    if hand is not None:
                        raw = add_hand(raw, hand)
                    # "Person" = what the segmenter sees PLUS anything that differs
                    # from the known room, so poses/actions/clothing the segmenter
                    # misses are still removed.
                    person = raw
                    scene_changed = False
                    have_plate = plate.matches(frame) and plate.coverage >= 0.2
                    if have_plate:
                        size = (raw.shape[1], raw.shape[0])
                        diff, _ = difference_mask(frame, plate.image, plate.known, size)
                        # How much of the visible *room* (known, not the person) changed?
                        room = (cv2.resize(plate.known, size, interpolation=cv2.INTER_NEAREST) > 0) & (raw < 0.1)
                        n_room = np.count_nonzero(room)
                        room_changed = float(np.count_nonzero(diff[room] > 0.5)) / n_room if n_room > 500 else 0.0
                        scene_changed = room_changed > SCENE_CHANGED
                        if not scene_changed:
                            person = np.maximum(raw, diff)
                        if learning and scene_monitor.update(room_changed, now):
                            print("[ghostframe] camera moved or lighting changed: relearning background")
                            plate.reset()
                            plate_warning = ""
                            refiner.reset()
                            have_plate = False
                    if learning:
                        plate.learn(frame, background_mask(person), now, allow_new=person_seen)
                    if strength > 0 or show_mask:
                        comp = person
                        if have_plate:
                            # Never-seen areas have always been covered by the person:
                            # always replace them.
                            unknown = cv2.resize(cv2.bitwise_not(plate.known), size,
                                                 interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255.0)
                            comp = np.maximum(comp, unknown)
                        last_mask = refiner.refine(comp)
                    if strength > 0 and plate.matches(frame) and plate.coverage >= args.min_coverage:
                        out = composite(frame, plate.filled(now), last_mask, strength)
                        ghost_frames += 1
                    effect_frames += 1
                    effect_ms += (time.perf_counter() - t) * 1000
                else:
                    refiner.reset()

                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                                 (out.shape[1], out.shape[0]))
                    writer.write(out)

                # --- Preview (HUD is drawn on a copy; `out` stays clean for stage 5 output) ---
                if show_mask and last_mask is not None:
                    m = cv2.resize(last_mask, (frame.shape[1], frame.shape[0]))
                    view = cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    view = out.copy()

                if plate.capturing:
                    left = plate.seconds_left(now)
                    if left > 0:
                        centered(view, "Step out of the frame", view.shape[0] // 2 - 40, AMBER, 1.2)
                        centered(view, f"{left:.0f}", view.shape[0] // 2 + 40, WHITE, 2.5, 4)
                    else:
                        centered(view, f"Capturing background... {plate.capture_progress():.0%}",
                                 view.shape[0] // 2, AMBER, 1.2)
                    plate_status = None
                elif plate.image is None:
                    plate_status = ("no background - press B" if not learn else "learning background...", AMBER)
                elif not plate.matches(frame):
                    plate_status = ("background size mismatch - press B or R", AMBER)
                elif scene_changed:
                    plate_status = ("room changed (camera moved / lighting) - relearning..." if learn
                                    else "room changed (camera moved / lighting) - press B", AMBER)
                elif plate_warning:
                    plate_status = (plate_warning, AMBER)
                elif plate.coverage < 0.98:
                    fill = plate.fill_status
                    if fill != last_fill:
                        if fill == "realistic":
                            print(f"[ghostframe] realistic fill ready "
                                  f"({filler.last_duration:.1f}s, {plate.coverage:.0%} seen)")
                        last_fill = fill
                    if fill == "realistic" and plate.coverage < 0.6:
                        hint = " - filled in; press B once for a perfect result"
                    elif fill == "realistic":
                        hint = " - rest filled in"
                    elif fill == "improving":
                        hint = " - improving fill..."
                    elif plate.coverage < 0.6:
                        hint = " - for a perfect result press B once and step aside"
                    else:
                        hint = " - move around a little" if learn else " - press B"
                    plate_status = (f"background {plate.coverage:.0%} seen{hint}",
                                    AMBER if not plate_ok else GREY)
                else:
                    plate_status = (f"background ready ({plate.coverage:.0%})", GREY)
                draw_hud(view, meter.tick(), toggle, gestures_on, label, score, now, plate_status)

            if not args.no_window:
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("i")):
                    toggle.toggle(time.monotonic())
                    print(f"[ghostframe] keyboard toggle -> {'ON' if toggle.active else 'OFF'}")
                elif key == ord("g") and detector is not None:
                    gestures_on = not gestures_on
                elif key == ord("b"):
                    if plate.capturing:
                        plate.cancel()
                    else:
                        plate.start_capture(time.monotonic())
                elif key == ord("m"):
                    show_mask = not show_mask
                elif key == ord("r"):
                    plate.reset()
                    plate_warning = ""
                    print("[ghostframe] background reset; relearning")
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if plate.image is not None and not plate.capturing:
            plate.save()  # keep what was learned for next time
        cam.release()
        if detector is not None:
            detector.close()
        segmenter.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    if frames:
        print(f"[ghostframe] {frames} frames in {elapsed:.1f}s = {frames / elapsed:.1f} FPS average")
    if effect_frames:
        print(f"[ghostframe] segmentation/learning/effect: {effect_frames} frames "
              f"({ghost_frames} with ghost on), {effect_ms / effect_frames:.1f} ms/frame")
    if plate.image is not None:
        print(f"[ghostframe] background {plate.coverage:.0%} known, saved to {plate.path}")
    if seen_labels:
        print(f"[ghostframe] gesture readings per frame: {dict(seen_labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
