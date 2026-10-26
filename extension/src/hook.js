// GhostFrame page hook (MAIN world, runs before the page's own scripts).
//
// Wraps navigator.mediaDevices.getUserMedia: when a site asks for the camera,
// it gets a processed video track instead. Frames are read with
// MediaStreamTrackProcessor and written with MediaStreamTrackGenerator, which
// keep running when the meeting tab is in the background (canvas/rAF based
// pipelines get throttled there). Each frame is sent as an ImageBitmap to the
// extension's processor frame and the processed bitmap comes back.
//
// Safety: if anything is unsupported or fails, the original camera stream is
// returned unchanged, so GhostFrame can never break a meeting.
(() => {
  if (window.__ghostframeHooked) return;
  window.__ghostframeHooked = true;

  const md = navigator.mediaDevices;
  if (!md || typeof md.getUserMedia !== "function") return;
  const supported =
    typeof window.MediaStreamTrackProcessor === "function" &&
    typeof window.MediaStreamTrackGenerator === "function" &&
    typeof window.VideoFrame === "function";

  const FRAME_ID = "__ghostframe_processor";
  const MAX_IN_FLIGHT = 2;
  const originalGUM = md.getUserMedia.bind(md);
  const status = (window.__ghostframe = { supported, streams: 0, sent: 0, received: 0, passthrough: 0, ready: false });

  // ---- channel to the processor frame -------------------------------------
  let procWin = null;
  let procOrigin = null;
  const waiting = new Map(); // id -> callback(bitmap|null)
  let nextId = 1;

  function findProcessor() {
    if (procWin) return true;
    const el = document.getElementById(FRAME_ID);
    if (!el || !el.contentWindow) return false;
    try {
      procOrigin = new URL(el.src).origin;
    } catch {
      return false;
    }
    procWin = el.contentWindow;
    procWin.postMessage({ gf: "hello" }, procOrigin);
    return true;
  }

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || typeof d !== "object" || typeof d.gf !== "string") return;
    if (!procWin) findProcessor();
    if (!procWin || e.source !== procWin || e.origin !== procOrigin) return;
    if (d.gf === "ready") status.ready = true;
    else if (d.gf === "result") {
      const cb = waiting.get(d.id);
      waiting.delete(d.id);
      if (cb) cb(d.bitmap || null);
      else if (d.bitmap) d.bitmap.close();
    } else if (d.gf === "state") status.state = d.state;
  });

  function process(bitmap, small, cam) {
    return new Promise((resolve) => {
      const id = nextId++;
      waiting.set(id, resolve);
      procWin.postMessage({ gf: "frame", id, bitmap, small, cam }, procOrigin, [bitmap, small]);
      status.sent++;
      setTimeout(() => {
        if (waiting.delete(id)) resolve(null); // processor stuck: don't stall the call
      }, 1000);
    });
  }

  // ---- wrapping a camera track -----------------------------------------------
  function wrap(stream) {
    const source = stream.getVideoTracks()[0];
    const reader = new MediaStreamTrackProcessor({ track: source }).readable.getReader();
    const generator = new MediaStreamTrackGenerator({ kind: "video" });
    const writer = generator.writable.getWriter();
    let inFlight = 0;
    let closed = false;
    let lastWritten = -1;
    const cam = (source.getSettings && source.getSettings().deviceId) || "";

    const write = (frame) => {
      // Results can arrive out of order after a timeout; never go back in time.
      if (closed || frame.timestamp <= lastWritten) return frame.close();
      lastWritten = frame.timestamp;
      writer.write(frame).catch(() => frame.close());
    };

    (async () => {
      try {
        for (;;) {
          const { value: frame, done } = await reader.read();
          if (done || closed) {
            if (frame) frame.close();
            break;
          }
          if (!findProcessor() || !status.ready) {
            if (procWin && status.passthrough % 15 === 0) procWin.postMessage({ gf: "hello" }, procOrigin);
            status.passthrough++;
            write(frame); // processor not loaded yet: show the real camera
            continue;
          }
          if (inFlight >= MAX_IN_FLIGHT) {
            frame.close(); // processor busy: drop this frame
            continue;
          }
          inFlight++;
          const timestamp = frame.timestamp;
          // Full frame for compositing + a small copy for analysis, both made
          // from the camera frame directly (avoids a GPU readback in the processor).
          const sw = 320, sh = Math.round((320 * frame.displayHeight) / frame.displayWidth);
          const [bitmap, small] = await Promise.all([
            createImageBitmap(frame),
            createImageBitmap(frame, { resizeWidth: sw, resizeHeight: sh, resizeQuality: "medium" }),
          ]);
          frame.close();
          process(bitmap, small, cam).then((out) => {
            inFlight--;
            if (!out) return;
            status.received++;
            write(new VideoFrame(out, { timestamp }));
            out.close();
          });
        }
      } catch (err) {
        console.warn("[GhostFrame] video pipeline stopped:", err);
      } finally {
        closed = true;
        writer.close().catch(() => {});
      }
    })();

    // Make the generated track behave like the camera track for the site.
    const stopGenerated = generator.stop.bind(generator);
    generator.stop = () => {
      closed = true;
      reader.cancel().catch(() => {});
      source.stop();
      stopGenerated();
    };
    generator.getSettings = () => ({ ...source.getSettings() });
    generator.getCapabilities = () => source.getCapabilities();
    generator.getConstraints = () => source.getConstraints();
    generator.applyConstraints = (c) => source.applyConstraints(c);
    Object.defineProperty(generator, "label", { get: () => source.label });
    source.addEventListener("ended", () => generator.stop());

    status.streams++;
    return new MediaStream([generator, ...stream.getAudioTracks()]);
  }

  md.getUserMedia = async function getUserMedia(constraints) {
    const stream = await originalGUM(constraints);
    if (!supported || !constraints || !constraints.video || stream.getVideoTracks().length === 0) return stream;
    try {
      return wrap(stream);
    } catch (err) {
      console.warn("[GhostFrame] falling back to the original camera:", err);
      return stream;
    }
  };
})();
