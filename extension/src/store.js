// Persists the learned background in the extension's IndexedDB (local to this
// browser profile; shared by all meeting sites because the processor frame
// always has the extension's origin). Keyed by camera + resolution.

const DB = "ghostframe";
const STORE = "backgrounds";

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  try {
    return await new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const req = fn(t.objectStore(STORE));
      t.oncomplete = () => resolve(req && req.result);
      t.onerror = () => reject(t.error);
    });
  } finally {
    db.close();
  }
}

/** record: { width, height, plate: Blob (PNG, full res), known: Uint8Array, lowW, lowH, lowPlate: Float32Array, fill?: Blob } */
export const saveBackground = (key, record) => tx("readwrite", (s) => s.put({ ...record, savedAt: Date.now() }, key));
export const loadBackground = (key) => tx("readonly", (s) => s.get(key));
export const deleteBackground = (key) => tx("readwrite", (s) => s.delete(key));
export const clearBackgrounds = () => tx("readwrite", (s) => s.clear());
