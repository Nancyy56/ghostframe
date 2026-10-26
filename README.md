# GhostFrame

Make yourself disappear from your live webcam feed with an open-palm gesture,
while the room behind you stays visible. All processing is local.

## Status

| Stage | Feature | Status |
|---|---|---|
| 1 | Webcam capture, preview, FPS display, camera error handling | **Done, tested** (see below) |
| 2 | Open-palm gesture toggle (cooldown + keyboard fallback) | **Done, tested** (palm toggle confirmed by user) |
| 3 | Background plate capture, person segmentation, compositing, edge smoothing | **Done, partly tested** (see below) |
| 4 | UI: status indicators, background-capture controls | Not started |
| 5 | Virtual camera output (v4l2loopback / OBS) | Not started, needs a kernel module (system change); the browser extension covers web meetings without it |
| 6 | Testing in Zoom / Google Meet / Microsoft Teams | Not started |

## Browser extension

For Google Meet, Teams (web) and Zoom (web) in Chrome/Edge/Brave, see
[extension/README.md](extension/README.md): the same pipeline ported to a
Chromium extension, with no system changes needed.

## Setup (Linux)

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
mkdir -p models
curl -fL -o models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
curl -fL -o models/selfie_segmenter_landscape.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite
curl -fL -o models/selfie_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
# optional, 92 MB, Apache-2.0: realistic fill of background hidden behind you
curl -fL -o models/inpainting_lama_2025jan.onnx \
  https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx
```

The models are downloaded once; all inference runs locally. Without the
gesture model, GhostFrame still runs and you toggle with the keyboard. The
segmentation model is required.

## Run

```sh
./.venv/bin/python -m ghostframe                     # /dev/video0
./.venv/bin/python -m ghostframe --source /dev/video2
./.venv/bin/python -m ghostframe --source clip.mp4   # test without a webcam (loops)
./.venv/bin/python -m ghostframe --no-window --max-frames 300   # headless benchmark
```

### Controls

| Input | Action |
|---|---|
| Hold an **open palm** toward the camera for 0.5 s | Toggle invisibility |
| `Space` or `I` | Toggle invisibility (keyboard fallback) |
| `B` | Capture an empty-room background manually (press again to cancel) |
| `R` | Forget the learned background and relearn from scratch |
| `G` | Turn gesture detection on/off |
| `M` | Show the person mask instead of the video (for tuning) |
| `Q` / `Esc` | Quit |

After any toggle there is a 1.5 s cooldown, and you must lower your hand
before the palm can trigger again, so holding your palm up doesn't flip the
state back and forth. Tune with `--hold` and `--cooldown`. A progress bar under
the gesture readout shows the hold filling up.

The preview is mirrored by default (`--no-mirror` to disable).

### Using it

**Automatic (default): no need to leave the frame.** Just start GhostFrame and
sit as usual. It learns the background continuously: pixels that have been
confidently *not* you (outside your segmented body and hand, plus a 20 px
safety margin so edges and shadows aren't baked in) for **8 frames in a row**
are blended into a background model. The 8-frame confirmation stops a
single-frame segmentation miss from baking part of you into the background. The status line shows how much is known, e.g.
`background 73% known - move around a little`.

- Once **50%** is known (`--min-coverage`), invisibility works: hold an open
  palm (or press Space) and you fade out over 0.3 s.
- Areas the camera has **never seen** (directly behind you, if you sit still)
  are filled in two steps: instantly with a basic OpenCV fill (blurry smear),
  then with the **LaMa** inpainting model once the learned background has
  stopped growing for 2 s (about 12–20 s after start). Running it earlier, on a
  half-learned background, produced a dark, wrong fill. LaMa is the
  inpainting model, which continues edges and structure (wall corners, window
  frames) through the hidden area. LaMa runs in a background thread limited to
  2 CPU threads, so the video keeps its frame rate; it re-runs only when more
  of the room has been revealed, or every 2 minutes to follow lighting. The
  status line shows `improving fill...` and then `rest filled in`.
  `--no-lama` disables it. Moving around still helps: anything really seen
  replaces the guess.
- The real background and the fill are joined with a soft 12 px seam, not a
  hard cut-out edge.
- The realistic fill is saved next to the plate (`backgrounds/plate_fill.png`)
  and reused at the next start until a fresh one is ready, so if you sit in
  your usual spot it looks right from the first second.
- Known areas keep updating slowly, so gradual lighting changes are followed.
- What has been learned is saved on exit to `backgrounds/plate.png` (alpha
  channel = which pixels were really seen; kept locally, git-ignored) and
  reloaded next time. Press **R** after moving the camera.

**Manual (best quality):** press **B**, step out of the frame during the 3 s
countdown (`--countdown`), and it takes the median of 20 frames as a complete,
sharp background. If that capture seems to contain a person, the status line
warns you to recapture. Learning continues afterwards to follow lighting.
Use `--no-learn` to rely on manual capture only.

### Removing the whole person, whatever they do

Recognising a person can fail (a close, low camera that sees only a chin and
a T-shirt; an unusual pose). So the mask is not only "what the segmenter
thinks is a person":

- **Difference detection:** anything that differs from the known room (with
  auto-exposure compensated) is treated as the person: a still T-shirt, an arm
  in a new position, a held object, a shadow.
- **Never-seen areas are always replaced.** If the camera has never seen a
  spot empty, the person has always been there.
- **Stricter learning:** new background is learned only in frames where a
  person is detected (if the detector sees nobody it may simply be failing),
  everything *below* a detected head/shoulders is treated as body, and known
  room pixels are only updated where the frame still matches them.
- **Camera moved / lights changed:** if most of the visible room (excluding the
  person) differs for 2 s, the background is relearned automatically.
- Invisibility is available once 25% of the room is known (`--min-coverage`);
  when little is known the status line suggests a one-time **B** capture.

Verified on a synthetic scenario in the user's room with an undetectable
"T-shirt" and an undetectable moving "arm": both were removed and the T-shirt
was never learned into the background (`--record out.mp4` saves the clean
output for such checks). 42 unit tests pass.

### Tuning the mask

| Option | Default | Effect |
|---|---|---|
| `--dilate N` | 6 | Grow the mask outward by N px (at 640 px width). Raise if you see a halo/outline. |
| `--feather N` | 7 | Edge blur radius. Higher = softer edge, but a wider blurry band. |
| `--decay F` | 0.8 | How fast the mask may shrink per frame. Higher = less flicker, but the edge lags behind fast motion. |
| `--no-lama` | off | Use only the basic fill for never-seen areas (saves CPU). |
| `--min-coverage F` | 0.5 | Fraction of the background that must be known before invisibility works. |
| `--segmenter` | landscape | `square` is the other model (slightly slower, sometimes better for close-ups). |

How the mask is built each frame: MediaPipe selfie segmentation (640x360) →
soft threshold → **grow-fast / shrink-slow** temporal filter (covering a bit
too much only shows a sliver of room; covering too little reveals you) →
union with the hand from the gesture tracker's landmarks (the segmenter often
misses raised fingers, which is where the toggle gesture happens) → dilate →
feather → blend with the plate at full resolution.

If the camera is missing or unplugged, the window shows a "No camera signal"
screen and retries every 2 s instead of crashing. In `--no-window` mode it exits
with status 1.

## Tested so far

Hardware: HP Victus 15-fa0xxx (i5-12450H), Arch-based Omarchy, Hyprland (Wayland),
external iVOOMi Netra 1080 FHD USB webcam.

- Missing camera / invalid device: clean error, no crash.
- Synthetic 720p30 video file: ~380 FPS through the pipeline (capture is not the bottleneck).
- Real webcam at 1280x720 MJPG: ~19–20 FPS measured in normal room light, both
  headless and with the preview window.
- **Not yet tested:** unplugging the camera mid-session (reconnect path), and
  closing the window with the title-bar button.

Stage 2 (gestures):

- Toggle logic (hold, cooldown, release-before-retrigger, dropout tolerance,
  keyboard toggle): 8 unit tests, `./.venv/bin/python -m unittest discover -s tests`.
- Recognizer runs asynchronously (MediaPipe `LIVE_STREAM` mode, ~23 ms per
  frame on the i5-12450H) and costs no measurable FPS: 19.0 FPS with and without it.
- Recognition verified on MediaPipe sample images (Victory, Thumb_Up) and on
  the live webcam (Victory, Thumb_Up, Closed_Fist recognised).
- Missing model file: falls back to keyboard-only with a warning.
- **Not yet tested:** an actual open palm triggering the toggle live, and the
  `Space` / `G` keys in the window (needs a person at the keyboard).

MediaPipe prints a few `W0000`/`I0000` log lines at startup; they are harmless.

Stage 3 (background replacement):

- 10 unit tests: compositing, soft threshold, grow-fast/shrink-slow filter,
  dilation, hand-hull coverage (including hands partly off-screen), plate
  countdown + median capture rejecting an outlier frame, plate save/load.
  18 tests in total pass.
- Visually checked on three MediaPipe sample photos with a checkerboard plate:
  person fully replaced, including raised fingers once the hand hull was added.
- Live webcam with a synthetic plate: effect costs **15.6 ms/frame**
  (segmentation ~10 ms, refine ~2.6 ms, hand ~0.8 ms, blend ~3 ms);
  18.9 FPS with the effect vs 19.3 FPS without, so the camera remains the limit.
- Missing plate / wrong-size plate: passthrough with a status message, no crash.
- Segmentation models compared on this CPU: landscape 7.7 ms, square 11 ms,
  multiclass 170 ms (too slow, not used).
- Manual capture with **B** and the effect in a real room: confirmed working by the user.
- **Not yet tested:** the person-in-plate warning firing on a real capture.

Realistic fill (LaMa):

- Compared on a person-shaped hole in a real room photo: OpenCV Telea (24 ms,
  smeared blob), xphoto FSR_FAST (361 ms, decent), ShiftMap (741 ms, blocky),
  LaMa (~8 s, continues wall and window edges; clearly best).
- LaMa via ONNX Runtime with 2 threads: 7–13 s per fill on the i5-12450H.
  `cv2.dnn` was no faster and can't be limited to its own threads.
- 5 unit tests (29 total): real fill keeps both sides of an edge; scheduling
  (first run, rerun only when the hole shrinks by 15% and ≥10 s passed,
  2-minute refresh); no job when nothing is unknown; reset discards a running
  job; the plate uses the result for holes only.
- Synthetic clip (the sample-photo person sitting almost still in the user's
  room): realistic fill ready after 13.4 s at 74% seen; processing stayed at
  24.4 FPS while LaMa ran. Visually, the hidden wall curve and window frame
  are continued instead of a dark smear.
- **Bug found from user testing and fixed:** LaMa had been started on a
  half-learned background (dark, flat fill); the fill was pasted with a
  stair-stepped hard edge; and single-frame segmentation misses baked parts
  of the person (light streaks, a skin-coloured patch) into the background.
  Fixes: wait until coverage is stable for 2 s, 8-frame confirmation before
  learning, linear mask upscaling, soft seam. Re-running LaMa on the user's
  own saved background gave a clean continuation of the wall and door.
- 3 more unit tests (32 total): flickering pixels are never learned; LaMa
  doesn't start on a still-changing background; the seam is a gradual ramp
  with exactly zero weight inside the hole; the saved fill is reused.
- Real-time replay (seated person, user's room): basic fill for the first
  ~12 s, realistic fill afterwards with no seam or blob.
- Live webcam with a person present: realistic fill ready after 11.6 s at
  72% seen; 18.3 FPS.

Automatic background learning:

- 6 more unit tests (24 total): a still person leaves a hole and is never
  baked into the plate; a moving person reveals the background; holes are
  inpainted, not black; lighting drift is followed; save/load keeps the
  known mask; reset.
- Synthetic moving-person demo (sample photo over a test pattern): 73% known
  while still (inpainted guess behind the person), 90% after moving, and the
  real pattern appears behind the person.
- Live webcam from an empty plate: 81–99% known within ~16 s; the learned plate
  was checked with the segmenter and contained **0.00%** person-like pixels.
- Cost with learning + effect on: **28 ms/frame**, 18.6 FPS (camera-limited at
  ~19). Profiling cut it from 47.5 ms (`cv2.copyTo` instead of `np.where`,
  safety-margin dilation at half resolution). Inpainting (~17 ms at 320x180)
  runs at most twice a second, only while unknown areas remain.

### Performance notes

- The camera only reaches 30 FPS at 720p in **MJPG**; raw YUYV is capped at
  10 FPS at that size. GhostFrame requests MJPG.
- Setting `CAP_PROP_BUFFERSIZE=1` halved the frame rate (20 → 10 FPS) on this
  camera, so it isn't used.
- The remaining gap to 30 FPS is most likely auto-exposure lengthening frame time
  in dim light (the camera runs in "Aperture Priority" auto-exposure). More light
  usually helps. `v4l2-ctl -d /dev/video0 -l` shows the controls.
- OpenCV's pip wheel bundles Qt5 without a Wayland plugin, so GhostFrame runs its
  window through XWayland. Qt prints harmless "Cannot find font directory" warnings.

## Why not a different model?

The fill model isn't the weak point: run on the user's real saved background,
LaMa continued the wall and door convincingly. The problems were in when and
how the fill was used (fixed above). Alternatives considered:

| Option | Behind a still person | Speed here | Verdict |
|---|---|---|---|
| Real pixels (B capture or learned) | Perfect | Instant | Always best where available |
| LaMa (used) | Good for walls, corners, frames | ~8–13 s once, in background | Best local option |
| MI-GAN | Slightly below LaMa | Much faster | Possible future speed-up |
| Stable Diffusion inpainting | Invents detail, hallucinates, varies per run | Minutes on CPU; tight on a 4 GB GTX 1650 | Too slow and unstable |
| Video inpainting (ProPainter, E2FGVI) | Uses other frames, but the same area stays covered | Needs a strong GPU, not real-time | Doesn't fit |

No model can know what is really behind someone the camera never saw past.

## Known limitations

### A background plate cannot reveal what is behind you

GhostFrame "removes" you by replacing your pixels with a **background plate**:
an image of the room, either learned while you were in other positions or
captured empty with **B**. The camera physically cannot see through you, so:

- **Never-seen areas are guesses.** With automatic learning, anything that has
  been behind you the whole time is invented by inpainting from its
  surroundings. LaMa makes this plausible for walls, corners and frames, but
  it cannot know about a poster, a shelf's contents or a person behind you
  that the camera never saw; it will draw something generic instead. In the
  first ~10–15 s (before LaMa finishes) the guess is a blurry smear. Moving
  around replaces guesses with the real view.

- **Anything that changes behind you while you cover it will not show.**
  If someone walks behind you, a door opens, a screen behind you changes, or
  an object is moved into the area your body covers, the plate still shows the old
  frozen image in that region. The camera physically cannot see through you.
- **Lighting changes break the illusion.** If daylight shifts or a lamp is
  switched on, the live room and the frozen plate no longer match, and a
  "ghost-shaped" patch with different brightness or colour becomes visible.
- **The camera must not move.** Any shift (bumping the desk, a laptop lid
  wobble, autofocus/zoom changes) misaligns the plate with the live view.
  Re-capture the plate after moving the camera.
- **Shadows and reflections you cast stay visible**, since they fall outside the
  segmented person mask.
- Segmentation is imperfect: expect occasional edge halos or flicker around hair,
  hands and fast motion, especially in low light.

- **Auto-exposure makes it worse.** When you leave the frame for the plate,
  or when you sit down in front of the camera, the webcam's automatic exposure
  and white balance readjust, so the plate and live room can differ in
  brightness. Locking exposure/white balance is planned for stage 4.

Implemented mitigations: automatic background learning (which also follows
slow lighting changes in visible areas), one-key recapture (B) and reset (R),
temporal mask smoothing, dilation and feathering. Planned: exposure/white-balance locking. None of
them can reconstruct genuinely unseen content behind you.
