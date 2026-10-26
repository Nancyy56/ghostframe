// Loads the MediaPipe WASM runtime in a way that works inside an extension
// frame and a module worker.
//
// MediaPipe's own loader injects the WASM glue code as a classic <script>
// (or importScripts in workers). Module workers have no importScripts, and the
// ES-module glue we package can't run as a classic script ("ModuleFactory not
// set"). So we import the module ourselves (it sets globalThis.ModuleFactory)
// and give MediaPipe a fileset without a loader path, which makes it skip
// loading and use the factory directly.

let fileset = null;

export async function visionFileset() {
  if (!fileset) {
    await import(new URL("../vendor/mediapipe/wasm/vision_wasm_module_internal.js", import.meta.url).href);
    if (typeof globalThis.ModuleFactory !== "function") throw new Error("MediaPipe runtime failed to load");
    fileset = {
      wasmLoaderPath: "",
      wasmBinaryPath: new URL("../vendor/mediapipe/wasm/vision_wasm_module_internal.wasm", import.meta.url).href,
    };
  }
  return fileset;
}

export const modelUrl = (name) => new URL(`../models/${name}`, import.meta.url).href;
