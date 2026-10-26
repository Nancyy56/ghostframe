// Popup: status of GhostFrame in the active tab + controls.
const $ = (id) => document.getElementById(id);
const DEFAULTS = { enabled: true, gestures: true, holdTime: 0.5 };
let tabId = null;

const BG_TEXT = {
  none: "not started",
  learning: "learning…",
  partial: "partly learned",
  ready: "ready",
  "scene-changed": "room changed – relearning",
  "capture-countdown": "step out of view…",
  capturing: "capturing…",
};

function command(cmd) {
  if (tabId === null) return;
  chrome.runtime.sendMessage({ type: "popup:command", tabId, command: cmd });
  setTimeout(refresh, 150);
}

function render(s) {
  $("idle").hidden = !!s;
  $("live").hidden = !s;
  const badge = $("badge");
  if (!s) {
    badge.textContent = "not in a call";
    badge.className = "pill";
    return;
  }
  if (!s.ready) {
    badge.textContent = s.initError ? "error" : "starting…";
    badge.className = "pill";
  } else {
    badge.textContent = s.invisible && s.enabled ? "invisible" : "visible";
    badge.className = "pill " + (s.invisible && s.enabled ? "on" : "ready");
  }
  const inv = $("invisible");
  inv.textContent = s.invisible ? "Become visible" : "Become invisible";
  inv.className = "primary" + (s.invisible ? " on" : "");
  inv.disabled = !s.ready || !s.enabled;

  const pct = Math.round(s.coverage * 100);
  $("bgBar").style.width = pct + "%";
  let text = (BG_TEXT[s.background] || s.background) + (s.coverage ? ` · ${pct}% seen` : "");
  if (s.background === "capture-countdown") text = `step out of view… ${Math.ceil(s.captureLeft)}`;
  $("bgText").textContent = text;
  const fillNote = s.fill.endsWith("improving") ? " Improving the fill of hidden areas…" : s.fill === "realistic" ? " Hidden areas: realistic fill." : "";
  $("bgHint").textContent =
    s.coverage > 0 && s.coverage < 0.6 && !s.background.startsWith("captur")
      ? "Parts of the room have never been seen and are filled in. For a perfect result, use Capture empty room once."
      : s.coverage < 0.25 && s.ready ? "Learning the room while you sit normally; invisibility works from 25%." : "";
  if (s.coverage > 0 && s.coverage < 0.995) $("bgHint").textContent += fillNote;
  $("capture").textContent = s.background.startsWith("captur") ? "Cancel capture" : "Capture empty room";
  $("capture").disabled = $("relearn").disabled = !s.ready || !s.enabled;
  $("perf").textContent = s.ready
    ? `${s.resolution} · ${s.fps} FPS · ${s.ms} ms/frame` + (s.gesturesReady ? "" : " · gestures loading") +
      (s.fillSource ? ` · fill: ${s.fillSource === "lama" ? "LaMa" : "MI-GAN"}` : "")
    : s.initError;
}

async function refresh() {
  if (tabId === null) return render(null);
  const r = await chrome.runtime.sendMessage({ type: "popup:status", tabId }).catch(() => null);
  render(r && r.status);
}

async function init() {
  $("version").textContent = "v" + chrome.runtime.getManifest().version;
  // ?tab=<id> lets the end-to-end test render the popup for a specific tab.
  const forced = Number(new URLSearchParams(location.search).get("tab"));
  const [tab] = forced ? [{ id: forced }] : await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = tab ? tab.id : null;
  const settings = { ...DEFAULTS, ...(await chrome.storage.local.get(DEFAULTS)) };
  $("enabled").checked = settings.enabled;
  $("gestures").checked = settings.gestures;
  $("enabled").onchange = (e) => chrome.storage.local.set({ enabled: e.target.checked });
  $("gestures").onchange = (e) => chrome.storage.local.set({ gestures: e.target.checked });
  $("invisible").onclick = () => command({ type: "toggle" });
  $("capture").onclick = () => command({ type: "capture" });
  $("relearn").onclick = () => command({ type: "relearn" });
  const cmds = await chrome.commands.getAll();
  const c = cmds.find((x) => x.name === "toggle-invisible");
  if (c && c.shortcut) $("shortcut").textContent = c.shortcut;
  refresh();
  setInterval(refresh, 500);
}
init();
