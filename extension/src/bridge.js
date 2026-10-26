// GhostFrame bridge (isolated content-script world).
//
// Injects the hidden processor frame. Video processing (MediaPipe needs
// WebAssembly) runs inside this extension-origin frame because some meeting
// sites' Content-Security-Policy forbids WebAssembly in the page itself
// (e.g. Teams). The page-side hook (hook.js) talks to it with postMessage.
(() => {
  const ID = "__ghostframe_processor";
  if (document.getElementById(ID)) return;
  const frame = document.createElement("iframe");
  frame.id = ID;
  frame.src = chrome.runtime.getURL("src/processor.html");
  frame.setAttribute("aria-hidden", "true");
  frame.tabIndex = -1;
  frame.style.cssText =
    "position:fixed!important;width:1px!important;height:1px!important;border:0!important;" +
    "opacity:0!important;pointer-events:none!important;right:0!important;bottom:0!important;";
  const attach = () => (document.body || document.documentElement).appendChild(frame);
  if (document.documentElement) attach();
  else document.addEventListener("readystatechange", attach, { once: true });
})();
