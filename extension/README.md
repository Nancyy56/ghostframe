# GhostFrame browser extension

Disappear from your webcam in **Google Meet, Microsoft Teams (web) and Zoom
(web client)**: hold an open palm to the camera and you fade out, with the room
behind you shown where you were. Works for any user in any room, with no
setup: the room is learned automatically while you sit normally. All video
processing runs on your device.

Chromium browsers (Chrome, Edge, Brave, Chromium) 116+. Does not affect the
Zoom / Teams **desktop** apps; the Python app in the parent folder covers those
via a virtual camera.

## Using it

1. Install, then open a meeting and turn your camera on (reload the meeting
   tab once if it was already open).
2. Sit normally for a few seconds while GhostFrame learns the room.
3. **Hold an open palm** to the camera for half a second → you fade out. Do it
   again to come back. Alternatives: **Alt+Shift+G** (changeable at
   `chrome://extensions/shortcuts`) or the toolbar popup. The toolbar badge
   shows **ON** while you're invisible.

Popup: live status, how much of the room has been seen, **Capture empty
room** (step out of view for 3 s for a perfect background), **Relearn**, and
on/off switches for the extension and the gesture.

## What it does each frame

| Step | How | Where |
|---|---|---|
| Find the person | MediaPipe selfie segmentation + hand landmarks (raised fingers) | segmentation & gesture workers |
| …including what a detector misses | Anything that differs from the learned room (still T-shirt, new arm position, held object, shadow), with auto-exposure compensation | processor, 160×90 |
| Learn the room | Pixels confirmed "not person" for 8 frames; nothing new is learned in frames where no person is detected, nor in the first 1.5 s (before hand detection runs); below a detected person, a widening "body cone" is treated as body (shoulders, arms, clothing) | processor, 320×180 |
| Match exposure | The room is stored at a fixed reference exposure; a smooth gain map (live vs stored room, measured where the room is visible) normalises what's learned and is applied to everything shown, so areas learned or filled at different times match the camera's current exposure and white balance | processor + WebGL2 |
| Camera moved / lights changed | Most of the visible room differs for 2 s → relearn automatically | processor |
| Never-seen areas | Instant smooth fill; **LaMa** in the background (~20 s on one CPU core, cached and saved) for the realistic fill; **MI-GAN** (~0.3–2 s) only for small holes (<15% of the frame) | LaMa & MI-GAN workers |
| Composite | Learned room + fill, soft seam, feathered person mask, 0.3 s fade | WebGL2, full resolution |
| Remember | Learned room + realistic fill saved per camera in the extension's IndexedDB | processor |

## Architecture

```
meeting page (any CSP)                         extension frame (own CSP, WebAssembly OK)
┌───────────────────────────────┐  full frame +   ┌───────────────────────────────────────┐
│ hook.js wraps getUserMedia     │  320px copy     │ processor.js  (logic + WebGL2 compositor)│
│ MediaStreamTrackProcessor  ────┼───────────────▶ │   ├─ segment-worker.js  (MediaPipe)      │
│ MediaStreamTrackGenerator  ◀───┼──────────────── │   ├─ gesture-worker.js  (MediaPipe)      │
└───────────────────────────────┘  processed frame │   ├─ fill-worker.js     (MI-GAN / ORT)   │
                                                   │   └─ lama-worker.js     (LaMa / ORT)     │
        ▲ commands/status                          └──────────────▲────────────────────────┘
        │                                                         │ port
  popup ─┴──────── background.js (service worker: shortcut, badge, routing) ─┘
```

Design decisions, each verified:

- **Processing lives in an extension frame**, not the page: Teams' CSP blocks
  WebAssembly in the page (checked live headers: Meet and Zoom allow it, Teams
  does not). The end-to-end test runs against Teams' strictest policy.
- **Frames travel as transferred ImageBitmaps** (~37 ms round trip): Chromium
  151 can't transfer a live `MediaStreamTrack` between frames (tested).
- **MediaStreamTrackProcessor/Generator** keep processing in background tabs,
  where canvas/rAF pipelines get throttled (30 FPS measured in a background tab).
- **Segmentation and gestures run in workers** (results at most a frame old,
  covered by the difference mask and grow-fast/shrink-slow filtering).
- **LaMa for large holes, MI-GAN only for small ones.** On learned
  backgrounds with a large never-seen area, LaMa's mean error was 8.5 vs
  MI-GAN's 17.8 (dim real room) and 29.6 vs 52.2 (generated room); MI-GAN
  invents furniture and lettering, LaMa continues walls and floors. LaMa's
  Fourier convolutions compute wrong results on ONNX Runtime Web's WebGPU
  backend, so it runs on WASM (~20 s, one core, rarely, cached).
- **Exposure normalisation**: webcams keep re-adjusting exposure; without it,
  areas behind the person that were learned or filled earlier show up as a
  lighter/darker patch (reported from a real call).
- **Fail-safe**: if anything is unsupported, fails, or the processor doesn't
  answer within 1 s, the site gets the real camera. GhostFrame never breaks a
  meeting.

## Measured performance

Intel i5-12450H laptop, Intel UHD (WebGL) / GTX 1650 (WebGPU), Chromium 151,
1280×720 at 30 FPS fake camera:

| | Output FPS | Processor time/frame |
|---|---|---|
| Visible | 30 | ~10 ms |
| Invisible | 30.7 | ~15 ms |
| Invisible, tab in background | 30 | – |
| LaMa fill (background worker) | 30 during the fill | ~20 s per fill (WASM) |
| MI-GAN fill, small holes (background worker) | 30 during the fill | 0.25–1 s (WebGPU) / 1.8 s (WASM) |

## Tests

```sh
npm install
npm run setup        # copy MediaPipe + ONNX Runtime Web, fetch models (~124 MB)
npm test             # 25 unit tests (core logic) + end-to-end in headless Chromium
```

End-to-end options (environment variables):

| Variable | Effect |
|---|---|
| `GF_FAKE_CAMERA=clip.mjpeg` | Use a clip with a person; enables learning/invisibility/gesture/persistence checks |
| `GF_EMPTY_ROOM=room.png` | Same room without the person; enables the "person removed" pixel checks |
| `GF_HEADFUL=1` | Real GPU window; frame-rate checks become hard checks |
| `GF_NO_WEBGPU=1` | Test the WASM fill fallback (what Linux Chromium users get by default) |
| `GF_ZIP=dist/ghostframe-0.3.0.zip` | Test the packaged zip instead of the source tree |
| `GF_SAVE_FRAMES=dir` | Save output frames for inspection |

Verified on three scenes: a dim real room and a generated room (window,
bookshelf), each with a person, an undetectable T-shirt and an undetectable
moving arm, plus the generated room with the camera exposure drifting
0.75×–1.3×. 25/25 checks on the packaged zip for each, 0.0–4.7% of the
person's pixels still showing (compression noise and fill colour matches).

## Release

```sh
npm run package      # -> dist/ghostframe-<version>.zip (~115 MB), checks all files are present
```

Upload to the Chrome Web Store with `PRIVACY.md` as the privacy policy. The
listing's "single purpose" is: modify the user's own camera video on supported
meeting sites. Permissions: site access to the four meeting domains, `storage`.

## Known limitations

- **Not yet tested in live Meet / Teams / Zoom calls** (needs accounts). The
  integration is standard `getUserMedia` wrapping, tested against a page with
  Teams' CSP, but each site's real call flow still needs a manual check.
- **Never-seen background is imagined.** If you never move, the area behind
  you is generated: plausible walls and floors (soft and slightly blurry for
  the first ~20 s, then LaMa), but it can't know what's really there (a
  picture or lamp that was always behind you won't appear). Moving a little
  or **Capture empty room** (once per room; saved) replaces guesses with the
  real room.
- **Clothing or objects detached from the detected body** (sticking out
  sideways beyond the shoulders) can still be learned as background and then
  smear into the fill; seen in a synthetic test with drifting exposure.
- Faint outlines can remain where something moved through an already-learned
  area (its soft edges are slowly blended into the stored room).
- **Static background assumption**: things that change behind you while you
  cover them don't show; a moving person or object behind you is treated as
  part of you (removed). Big lighting changes trigger an automatic relearn.
- Chromium only (uses `MediaStreamTrackProcessor`/`Generator`); Firefox and
  Safari don't support these on the main thread.
