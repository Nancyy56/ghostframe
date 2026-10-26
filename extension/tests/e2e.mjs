// End-to-end test: loads the extension into Chromium with a fake camera and a
// local "meeting" page served with Microsoft Teams' strict CSP (no WebAssembly,
// Trusted Types, default-src 'none'), then checks the camera is intercepted and
// processed video reaches the page, also while the tab is in the background.
//
//   npm test                                   (generated test pattern as camera)
//   GF_FAKE_CAMERA=person.mjpeg npm test       (also runs the person/invisibility checks)
//   GF_SAVE_FRAMES=out/ ...                    (saves output frames for inspection)
//   GF_HEADFUL=1 ...                           (real GPU window; enables frame-rate checks)
//   GF_ZIP=dist/ghostframe-x.y.z.zip ...       (test the packaged zip instead of the source tree)
//
// Requires: Chromium (CHROMIUM env var, default /usr/bin/chromium), ffmpeg.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import puppeteer from "puppeteer-core";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const CHROMIUM = process.env.CHROMIUM || "/usr/bin/chromium";
const PORT = 8765;
const TEST_ORIGIN = `http://localhost:${PORT}/*`;
const TEAMS_CSP =
  "base-uri 'none';default-src 'none';form-action 'none';frame-ancestors 'none';object-src 'none';" +
  "require-trusted-types-for 'script';script-src 'nonce-gftest' 'unsafe-inline';style-src 'unsafe-inline';trusted-types 'none'";

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ghostframe-e2e-"));
const results = [];
const HEADFUL = !!process.env.GF_HEADFUL; // real GPU: frame-rate checks are meaningful
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
};

// Frame-rate checks only mean something with a real GPU (headless Chromium
// renders WebGL in software); headless they're reported as INFO.
const perf = (name, ok, detail) => {
  if (HEADFUL) return check(name, ok, detail);
  console.log(`INFO  ${name}  (${detail}; software GL - run with GF_HEADFUL=1 for a real check)`);
};

// ---- test copy of the extension that also runs on the local test page -------
function prepareExtension() {
  const dir = path.join(tmp, "ext");
  if (process.env.GF_ZIP) {
    // Test the packaged artifact exactly as it would be uploaded to the store.
    fs.mkdirSync(dir, { recursive: true });
    execFileSync("unzip", ["-q", path.resolve(process.env.GF_ZIP), "-d", dir]);
  } else {
    fs.cpSync(ROOT, dir, { recursive: true, filter: (p) => !/node_modules|[\\/](tests|dist)([\\/]|$)/.test(p) });
  }
  const mf = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  for (const cs of mf.content_scripts) cs.matches.push(TEST_ORIGIN);
  for (const war of mf.web_accessible_resources) war.matches.push(TEST_ORIGIN);
  fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(mf, null, 2));
  return dir;
}

function fakeCamera() {
  if (process.env.GF_FAKE_CAMERA) return path.resolve(process.env.GF_FAKE_CAMERA);
  const file = path.join(tmp, "camera.mjpeg");
  execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
    "-t", "10", "-q:v", "5", file]);
  return file;
}

const PAGE = `<!doctype html><meta charset=utf-8><title>meeting</title>
<video id=v autoplay muted playsinline width=320></video>
<script nonce="gftest">
window.T = { log: [] };
window.startCamera = async () => {
  const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: true });
  window.stream = s;
  const v = document.getElementById("v"); v.srcObject = s; await v.play();
  const t = s.getVideoTracks()[0];
  return { tracks: s.getTracks().map(x => x.kind), settings: t.getSettings(), label: t.label, ctor: t.constructor.name };
};
window.countFrames = (ms) => new Promise(res => {
  const v = document.getElementById("v"); let n = 0; const t0 = performance.now();
  const tick = () => { n++; if (performance.now() - t0 < ms) v.requestVideoFrameCallback(tick); else res(n * 1000 / ms); };
  v.requestVideoFrameCallback(tick);
});
window.meanPixel = () => { const v = document.getElementById("v"); const c = new OffscreenCanvas(64, 36);
  const g = c.getContext("2d"); g.drawImage(v, 0, 0, 64, 36); const d = g.getImageData(0, 0, 64, 36).data;
  let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i] + d[i + 1] + d[i + 2]; return s / (d.length / 4 * 3); };
</script>`;

const server = http.createServer((req, res) => {
  res.writeHead(200, { "content-type": "text/html", "content-security-policy": TEAMS_CSP });
  res.end(PAGE);
});
await new Promise((r) => server.listen(PORT, r));

const browser = await puppeteer.launch({
  executablePath: CHROMIUM,
  headless: !HEADFUL,
  pipe: true,
  enableExtensions: [prepareExtension()],
  // WebGPU is off by default in Linux Chromium; enable it for GPU runs unless
  // GF_NO_WEBGPU=1 (tests the WASM fallback that those users get).
  args: [...(HEADFUL && !process.env.GF_NO_WEBGPU ? ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] : []),
    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${fakeCamera()}`, "--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PERSON = !!process.env.GF_FAKE_CAMERA; // person checks need a clip with a person in it
const SAVE = process.env.GF_SAVE_FRAMES;     // directory to save output frames for inspection
const EMPTY_ROOM = process.env.GF_EMPTY_ROOM; // image of the room without the person: enables pixel checks
const skip = (name) => { results.push({ name, ok: true }); console.log(`SKIP  ${name}  (needs GF_FAKE_CAMERA with a person)`); };

async function waitFor(page, fn, ms, arg) {
  const end = Date.now() + ms;
  for (;;) {
    const v = await page.evaluate(fn, arg);
    if (v || Date.now() > end) return v;
    await sleep(250);
  }
}
const state = (page) => page.evaluate(() => (window.__ghostframe && window.__ghostframe.state) || {});

// Decode an image to 160x90 RGB with ffmpeg (no image libraries needed).
// Brightness (luma) only: JPEG colour subsampling smears saturated colour edges
// (e.g. book spines), which would otherwise look like "person" pixels.
const rgb = (file) => execFileSync("ffmpeg", ["-loglevel", "error", "-i", file, "-vf", "scale=160:90:flags=area",
  "-f", "rawvideo", "-pix_fmt", "gray", "-"]);
const meanAbsDiff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };
const pxDiff = (a, b, i) => Math.abs(a[i] - b[i]);
/**
 * Fraction of the person's pixels that are still showing: pixels where the
 * visible frame clearly differs from the empty room (= the person) and the
 * invisible output is unchanged from the visible frame. Independent of how
 * well never-seen background is filled in.
 */
function personLeft(visible, invisible, room) {
  let person = 0, left = 0;
  for (let i = 0; i < room.length; i++) {
    if (pxDiff(visible, room, i) <= 30) continue;
    person++;
    if (pxDiff(invisible, visible, i) < 10) left++;
  }
  return person ? left / person : 0;
}

async function saveFrame(page, name) {
  const url = await page.evaluate(() => {
    const v = document.getElementById("v"), c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
    return c.toDataURL("image/png");
  });
  const file = path.join(SAVE || tmp, name + ".png");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.from(url.split(",")[1], "base64"));
  return file;
}

try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`http://localhost:${PORT}/`);

  const info = await page.evaluate(() => window.startCamera());
  check("page receives video + audio tracks", info.tracks.includes("video") && info.tracks.includes("audio"), info.tracks.join(","));
  check("camera track is GhostFrame's processed track", info.ctor === "MediaStreamTrackGenerator", info.ctor);
  check("track reports camera settings", info.settings.width === 1280 && info.settings.height === 720,
    `${info.settings.width}x${info.settings.height}`);

  const ready = await waitFor(page, () => window.__ghostframe && window.__ghostframe.state && window.__ghostframe.state.ready, 20000);
  const s0 = await state(page);
  check("processor (MediaPipe) starts under strict CSP", !!ready, s0.initError || "");
  await sleep(3000);
  const fps = await page.evaluate(() => window.countFrames(3000));
  const st = await page.evaluate(() => ({ ...window.__ghostframe, state: undefined }));
  const s1 = await state(page);
  check("frames go through the processor", st.received > 30, `sent ${st.sent}, received ${st.received}, passthrough ${st.passthrough}`);
  perf("output frame rate >= 20 FPS", fps >= 20, `${fps.toFixed(1)} FPS; processor ${s1.ms} ms/frame`);
  const mean = await page.evaluate(() => window.meanPixel());
  check("output is real video, not black", mean > 20, `mean ${mean.toFixed(0)}`);
  const gReady = await waitFor(page, () => window.__ghostframe.state && window.__ghostframe.state.gesturesReady, 15000);
  check("gesture worker loads", !!gReady);

  // Commands go through the same path as the popup: extension page -> service worker -> processor port.
  const sw = await browser.waitForTarget((t) => t.type() === "service_worker" && t.url().endsWith("src/background.js"));
  const extId = new URL(sw.url()).host;
  const ctl = await browser.newPage();
  await ctl.goto(`chrome-extension://${extId}/popup/popup.html`);
  const meetingTab = await ctl.evaluate(async () => {
    for (const t of await chrome.tabs.query({})) {
      const r = await chrome.runtime.sendMessage({ type: "popup:status", tabId: t.id });
      if (r && r.status && r.status.frames > 0) return t.id;
    }
    return null;
  });
  check("service worker knows the meeting tab", meetingTab !== null, String(meetingTab));
  const cmd = (command) => ctl.evaluate((tabId, c) => chrome.runtime.sendMessage({ type: "popup:command", tabId, command: c }), meetingTab, command);
  await page.bringToFront();

  if (PERSON) {
    const cov = await waitFor(page, () => (window.__ghostframe.state || {}).coverage >= 0.25, 25000);
    const s2 = await state(page);
    check("learns the room while the person is in view", !!cov, `${Math.round(s2.coverage * 100)}% seen`);
    const sawGesture = await waitFor(page, () => !!(window.__ghostframe.state || {}).gesture, 10000);
    check("recognises hand gestures", !!sawGesture, (await state(page)).gesture || "none");
    const fVisible = await saveFrame(page, "1-visible");
    await cmd({ type: "set-invisible", value: true });
    await sleep(1500);
    const s3 = await state(page);
    check("invisible via popup/shortcut command", s3.invisible === true);
    const fInvisible = await saveFrame(page, "2-invisible");
    const fpsInv = await page.evaluate(() => window.countFrames(3000));
    perf("frame rate while invisible >= 20 FPS", fpsInv >= 20, `${fpsInv.toFixed(1)} FPS; processor ${(await state(page)).ms} ms/frame`);
    await sleep(8000); // the clip's second half: undetectable "arm" moving through the learned room
    if (fs.existsSync(path.join(ROOT, "models", "inpainting_lama_2025jan.onnx"))) {
      // Large hole here: LaMa (WASM, ~30 s) is the one that should produce it.
      const real = await waitFor(page, () => (window.__ghostframe.state || {}).fillSource === "lama", 150000);
      const s4 = await state(page);
      check("realistic (LaMa) fill computed in the background", !!real,
        `source ${s4.fillSource}, ${s4.fillBackend}, ms ${JSON.stringify(s4.fillMs)}`);
      const fpsL = await page.evaluate(() => window.countFrames(2000));
      perf("frame rate with realistic fill >= 20 FPS", fpsL >= 20, `${fpsL.toFixed(1)} FPS`);
    } else skip("realistic fill (model missing: npm run setup)");
    const fLater = await saveFrame(page, "3-invisible-later");
    if (EMPTY_ROOM) {
      // The person must actually be gone, including parts a person detector can't see.
      const room = rgb(EMPTY_ROOM), vis = rgb(fVisible), inv = rgb(fInvisible), later = rgb(fLater);
      const left1 = personLeft(vis, inv, room), left2 = personLeft(vis, later, room);
      check("person removed (incl. undetectable clothing)", left1 < 0.05, `${(left1 * 100).toFixed(1)}% of person pixels still showing`);
      check("still removed later in the clip (new movement)", left2 < 0.05, `${(left2 * 100).toFixed(1)}% still showing`);
      // No visible patch: brightness where the person was, relative to a ring of
      // room around it, must match the empty room's ratio. Independent of the
      // camera's exposure (a stale-exposure background shows up as a lighter
      // or darker "shadow" patch).
      const patch = (img) => {
        const W = 160, H = 90, P = new Uint8Array(W * H);
        for (let i = 0; i < W * H; i++) P[i] = pxDiff(vis, room, i) > 30 ? 1 : 0;
        const ring = new Uint8Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          if (P[y * W + x]) continue;
          for (let d = 1; d <= 8 && !ring[y * W + x]; d++) {
            if ((x - d >= 0 && P[y * W + x - d]) || (x + d < W && P[y * W + x + d]) || (y - d >= 0 && P[(y - d) * W + x])) ring[y * W + x] = 1;
          }
        }
        const mean = (a, m) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) if (m[i]) { s += a[i]; n++; } return s / Math.max(1, n); };
        return mean(img, P) / Math.max(1, mean(img, ring)) / (mean(room, P) / Math.max(1, mean(room, ring)));
      };
      const p1 = patch(inv), p2 = patch(later);
      // Information, not pass/fail: it also reflects the guessed content of
      // never-seen areas (e.g. wall continued where a dark window really is).
      console.log(`INFO  brightness where the person was vs around it: ${p1.toFixed(3)} / ${p2.toFixed(3)} (1.000 = seamless)`);
      // Fill fidelity is information only: never-seen areas are imagined, not known.
      console.log(`INFO  mean |output - empty room|: visible ${meanAbsDiff(vis, room).toFixed(1)}, ` +
        `invisible ${meanAbsDiff(inv, room).toFixed(1)}, later ${meanAbsDiff(later, room).toFixed(1)}`);
    } else {
      skip("pixel check against the empty room (set GF_EMPTY_ROOM)");
    }
    const popup = await browser.newPage();
    await popup.goto(`chrome-extension://${extId}/popup/popup.html?tab=${meetingTab}`);
    await sleep(1200);
    const ui = await popup.evaluate(() => ({ badge: document.getElementById("badge").textContent,
      button: document.getElementById("invisible").textContent, bg: document.getElementById("bgText").textContent,
      live: !document.getElementById("live").hidden }));
    check("popup shows live status", ui.live && ui.badge === "invisible" && ui.button === "Become visible" && /% seen/.test(ui.bg),
      JSON.stringify(ui));
    await popup.close();
    await page.bringToFront();
    const badge = await ctl.evaluate((tabId) => chrome.action.getBadgeText({ tabId }), meetingTab);
    check("toolbar badge shows ON", badge === "ON", JSON.stringify(badge));
    await cmd({ type: "set-invisible", value: false });

    // Persistence: closing the meeting page saves the background; reopening restores it.
    const covBefore = (await state(page)).coverage;
    await page.reload();
    await page.evaluate(() => window.startCamera());
    const restored = await waitFor(page, (c) => ((window.__ghostframe.state || {}).coverage || 0) >= c * 0.8, 8000, covBefore);
    check("background is restored after reopening", !!restored, `before ${covBefore}, after ${(await state(page)).coverage}`);
    const sent = await cmd({ type: "relearn" });
    const dropped = await waitFor(page, () => ((window.__ghostframe.state || {}).coverage || 0) < 0.1, 2000);
    check("relearn clears the background", !!dropped, `command ${JSON.stringify(sent)}, coverage now ${(await state(page)).coverage}`);
  } else {
    for (const n of ["learns the room", "recognises gestures", "invisible via command", "frame rate while invisible",
      "toolbar badge", "background restored", "relearn"]) skip(n);
  }

  // Background tab: open another tab in front, processing must continue.
  const before = (await page.evaluate(() => window.__ghostframe.received));
  await ctl.bringToFront();
  await sleep(4000);
  const bgFps = ((await page.evaluate(() => window.__ghostframe.received)) - before) / 4;
  perf("keeps processing while the tab is in the background", bgFps >= 15, `${bgFps.toFixed(1)} FPS in background`);
  await ctl.close();
  await page.bringToFront();

  const stopped = await page.evaluate(async () => {
    window.stream.getVideoTracks()[0].stop();
    await new Promise((r) => setTimeout(r, 500));
    return window.stream.getVideoTracks()[0].readyState;
  });
  check("stop() ends the track", stopped === "ended", stopped);

  // Fallback: without the processor frame, the page still gets live video.
  await page.reload();
  await page.evaluate(() => document.getElementById("__ghostframe_processor")?.remove());
  const info2 = await page.evaluate(() => window.startCamera());
  await sleep(2000);
  const fps2 = await page.evaluate(() => window.countFrames(2000));
  check("without processor: falls back to live camera", fps2 >= 20, `${info2.ctor}, ${fps2.toFixed(1)} FPS`);

  check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
} finally {
  await browser.close();
  server.close();
  fs.rmSync(tmp, { recursive: true, force: true });
}

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
