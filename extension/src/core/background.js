// Learned background ("plate") bookkeeping at the processing resolution.
// Port of ghostframe/plate.py BackgroundPlate (learning part) and
// SceneChangeMonitor. The full-resolution plate lives on the GPU (gpu.js); every
// learn() returns the masks the GPU applies so both stay in lockstep.

export class BackgroundModel {
  constructor(w, h, { learnRate = 0.05, confirmFrames = 8, minCoverageForFill = 0.25, settleSeconds = 2 } = {}) {
    Object.assign(this, { w, h, learnRate, confirmFrames, minCoverageForFill, settleSeconds });
    this.reset();
  }

  reset() {
    const n = this.w * this.h;
    this.plate = new Float32Array(n * 3); // RGB 0..255
    this.known = new Uint8Array(n); // 255 = really seen
    this.streak = new Uint8Array(n); // consecutive background frames
    this.knownCount = 0;
    this.version = 0; // bumps when `known` grows
    this._covRef = { t: -1e9, cov: 0 };
  }

  get coverage() {
    return this.knownCount / (this.w * this.h);
  }

  /**
   * frame: RGBA Uint8ClampedArray (w*h*4). background: Uint8Array 255 where
   * safely not the person. allowNew=false (no person detected in this frame,
   * so the detector may be failing): only refresh already-known pixels.
   * gain (optional, RGB per pixel): the camera's current exposure relative
   * to the plate's reference exposure. Frames are divided by it before being
   * learned, so the whole plate stays at one reference exposure no matter
   * when each part was seen; the compositor multiplies by the current gain.
   * Returns { newMask, updateMask } (Uint8Array 255/0) for the GPU plate.
   */
  learn(frame, background, allowNew = true, gain = null) {
    const n = this.w * this.h;
    const newMask = new Uint8Array(n);
    const updateMask = new Uint8Array(n);
    const rate = this.learnRate;
    let grew = 0;
    for (let i = 0; i < n; i++) {
      if (background[i] && (allowNew || this.known[i])) this.streak[i] = Math.min(255, this.streak[i] + 1);
      else this.streak[i] = 0;
      if (this.streak[i] < this.confirmFrames) continue;
      const j = i * 3, f = i * 4;
      const r = gain ? frame[f] / gain[j] : frame[f];
      const g = gain ? frame[f + 1] / gain[j + 1] : frame[f + 1];
      const b = gain ? frame[f + 2] / gain[j + 2] : frame[f + 2];
      if (!this.known[i]) {
        this.known[i] = 255;
        newMask[i] = 255;
        this.plate[j] = r;
        this.plate[j + 1] = g;
        this.plate[j + 2] = b;
        grew++;
      } else {
        updateMask[i] = 255;
        this.plate[j] += rate * (r - this.plate[j]);
        this.plate[j + 1] += rate * (g - this.plate[j + 1]);
        this.plate[j + 2] += rate * (b - this.plate[j + 2]);
      }
    }
    if (grew) {
      this.knownCount += grew;
      this.version++;
    }
    return { newMask, updateMask };
  }

  /**
   * True once enough is known and coverage stopped growing for settleSeconds,
   * so an expensive fill (LaMa) isn't run on a half-learned plate.
   */
  settled(now) {
    const cov = this.coverage;
    if (cov - this._covRef.cov > 0.01) {
      this._covRef = { t: now, cov };
      return false;
    }
    return cov >= this.minCoverageForFill && now - this._covRef.t >= this.settleSeconds;
  }

  /**
   * Manual empty-room capture: frame k (0-based) of a capture. Marks every
   * pixel known; the plate becomes the running mean of the captured frames.
   * Returns the blend rate the GPU plate must use for this frame (1 = copy).
   */
  captureFrame(frame, k) {
    const n = this.w * this.h, r = 1 / (k + 1);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) this.plate[i * 3 + c] += r * (frame[i * 4 + c] - this.plate[i * 3 + c]);
    }
    if (this.knownCount !== n) {
      this.known.fill(255);
      this.knownCount = n;
      this.version++;
    }
    this.streak.fill(0);
    return r;
  }

  /** Restore from saved state (plate RGB float + known), e.g. from IndexedDB. */
  load(plate, known) {
    this.reset();
    this.plate.set(plate);
    this.known.set(known);
    this.knownCount = known.reduce((s, v) => s + (v ? 1 : 0), 0);
    this.version++;
  }
}

/** Fires when most of the visible room keeps differing: camera moved or lighting changed. */
export class SceneChangeMonitor {
  constructor({ threshold = 0.5, hold = 2 } = {}) {
    this.threshold = threshold;
    this.hold = hold;
    this._since = null;
  }

  update(changedFraction, now) {
    if (changedFraction <= this.threshold) {
      this._since = null;
      return false;
    }
    if (this._since === null) this._since = now;
    if (now - this._since >= this.hold) {
      this._since = null;
      return true;
    }
    return false;
  }

  get changing() {
    return this._since !== null;
  }
}
