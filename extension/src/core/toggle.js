// Open-palm toggle: hold to toggle, cooldown, release-before-retrigger,
// tolerance for brief detection dropouts. Port of ghostframe/gesture.py PalmToggle.
// Time is passed in (seconds) so the logic is unit-testable.

export class PalmToggle {
  constructor({ holdTime = 0.5, cooldown = 1.5, gapTolerance = 0.25 } = {}) {
    this.holdTime = holdTime;
    this.cooldown = cooldown;
    this.gapTolerance = gapTolerance;
    this.active = false;
    this._holdStart = null;
    this._lastSeen = -1e9;
    this._cooldownUntil = -1e9;
    this._needsRelease = false;
  }

  /** Feed one observation. Returns true if the state toggled on this call. */
  update(palmVisible, now) {
    if (palmVisible) this._lastSeen = now;
    else if (now - this._lastSeen > this.gapTolerance) {
      this._holdStart = null;
      this._needsRelease = false;
    }
    if (this._needsRelease || now < this._cooldownUntil) {
      this._holdStart = null;
      return false;
    }
    if (palmVisible && this._holdStart === null) this._holdStart = now;
    // Only fire on a frame where the palm is actually seen, not during a dropout.
    if (palmVisible && this._holdStart !== null && now - this._holdStart >= this.holdTime) return this._fire(now);
    return false;
  }

  /** Manual toggle (keyboard shortcut / popup); also starts the cooldown. */
  toggle(now) {
    this._fire(now);
  }

  set(active, now) {
    if (this.active !== active) this._fire(now);
  }

  _fire(now) {
    this.active = !this.active;
    this._holdStart = null;
    this._cooldownUntil = now + this.cooldown;
    this._needsRelease = now - this._lastSeen <= this.gapTolerance;
    return true;
  }

  progress(now) {
    return this._holdStart === null ? 0 : Math.min(1, (now - this._holdStart) / this.holdTime);
  }
}
