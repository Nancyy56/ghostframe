# GhostFrame privacy policy

_Last updated: 24 October 2026_

GhostFrame is a browser extension that lets you disappear from your own
webcam video in Google Meet, Microsoft Teams (web) and Zoom (web client).

## What GhostFrame processes

- **Your camera video**, only while a supported meeting site uses your camera.
  Each frame is analysed and changed **inside your browser**, on your device,
  and then handed to the meeting site in place of the original frame.
- **A model of your room's background**, learned from the camera so it can be
  shown where you are. It is stored in the extension's own local browser
  storage (IndexedDB) so you don't need to set it up again. It never leaves
  your device. You can delete it at any time with **Relearn** in the
  extension popup, or by removing the extension.
- **Your settings** (enabled, gesture on/off), in the extension's local storage.

## What GhostFrame does not do

- It does not send any video, image, background, or usage data anywhere. It
  makes no network requests: all models and code are packaged with the
  extension.
- It does not record, save or upload video frames. Only the learned
  background image is kept, locally, as described above.
- It has no analytics, tracking, advertising or accounts.
- It does not read the meeting's other content, chat, participants or audio
  (audio passes through unchanged).

## Permissions

- **Site access** to `meet.google.com`, `teams.microsoft.com`,
  `teams.live.com` and `*.zoom.us`: needed to replace the camera video on
  those sites. GhostFrame does nothing on other sites.
- **storage**: to remember your settings.

What you send to other meeting participants is, as always, the video the
meeting site transmits; GhostFrame only changes that video on your device
before the site receives it.

## Contact

Questions: nancyyyy288@gmail.com
