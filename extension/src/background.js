// GhostFrame service worker: routes the keyboard shortcut and popup commands
// to the processor frame of the right tab, and shows the state on the toolbar
// badge. Processor frames connect with a port named "processor" (they
// reconnect if this worker is restarted).

const tabs = new Map(); // tabId -> Map(port -> latest status)

function best(tabId) {
  const entry = tabs.get(tabId);
  if (!entry || entry.size === 0) return null;
  // A meeting page can contain several frames; prefer the one processing video.
  let pick = null;
  for (const [port, status] of entry) {
    if (!pick || (status && status.frames > ((pick.status && pick.status.frames) || 0))) pick = { port, status };
  }
  return pick;
}

function updateBadge(tabId) {
  const b = best(tabId);
  const on = !!(b && b.status && b.status.invisible && b.status.enabled);
  chrome.action.setBadgeText({ tabId, text: on ? "ON" : "" }).catch(() => {});
  chrome.action.setBadgeBackgroundColor({ tabId, color: "#d93025" }).catch(() => {});
  chrome.action.setTitle({ tabId, title: on ? "GhostFrame: you are invisible" : "GhostFrame" }).catch(() => {});
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "processor") return;
  const tabId = port.sender && port.sender.tab && port.sender.tab.id;
  if (tabId === undefined) return;
  if (!tabs.has(tabId)) tabs.set(tabId, new Map());
  tabs.get(tabId).set(port, null);
  port.onMessage.addListener((m) => {
    if (m && m.type === "status") {
      tabs.get(tabId)?.set(port, m.status);
      updateBadge(tabId);
    }
  });
  port.onDisconnect.addListener(() => {
    const entry = tabs.get(tabId);
    if (entry) {
      entry.delete(port);
      if (entry.size === 0) tabs.delete(tabId);
    }
    updateBadge(tabId);
  });
});

function send(tabId, command) {
  const b = best(tabId);
  if (!b) return false;
  b.port.postMessage(command);
  return true;
}

chrome.commands.onCommand.addListener(async (name, tab) => {
  if (name !== "toggle-invisible") return;
  const id = tab ? tab.id : (await chrome.tabs.query({ active: true, currentWindow: true }))[0]?.id;
  if (id !== undefined) send(id, { type: "toggle" });
});

chrome.runtime.onMessage.addListener((m, sender, reply) => {
  if (!m || typeof m.type !== "string" || !m.type.startsWith("popup:")) return;
  if (m.type === "popup:status") reply({ status: best(m.tabId)?.status || null });
  else if (m.type === "popup:command") reply({ ok: send(m.tabId, m.command) });
});
