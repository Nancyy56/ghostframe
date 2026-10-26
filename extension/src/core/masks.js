// Low-resolution mask operations on plain typed arrays (no DOM), used at the
// processing resolution (320x180 by default). Ports of the OpenCV operations
// in the Python app (ghostframe/segment.py, plate.py).
//
// Conventions: masks are Float32Array (0..1) or Uint8Array (0/255), row-major,
// with explicit width/height. Images are RGBA Uint8ClampedArray (ImageData).

// 1-D running max/min over a window of 2r+1 (van Herk / Gil-Werman): O(n)
// regardless of r. `line` holds the input padded by r samples of `edge` on
// both sides (length len + 2r); result goes to `res[0..len)`.
function lineFilter(line, len, r, isMax, g, hb, res) {
  const k = 2 * r + 1, total = len + 2 * r;
  for (let b = 0; b < total; b += k) {
    const e = Math.min(total, b + k);
    let m = line[b];
    g[b] = m;
    for (let i = b + 1; i < e; i++) { const v = line[i]; m = isMax ? (v > m ? v : m) : (v < m ? v : m); g[i] = m; }
    m = line[e - 1];
    hb[e - 1] = m;
    for (let i = e - 2; i >= b; i--) { const v = line[i]; m = isMax ? (v > m ? v : m) : (v < m ? v : m); hb[i] = m; }
  }
  for (let x = 0; x < len; x++) {
    const a = hb[x], c = g[x + 2 * r];
    res[x] = isMax ? (a > c ? a : c) : (a < c ? a : c);
  }
}

function morph(src, w, h, r, isMax) {
  if (r <= 0) return src.slice();
  const edge = isMax ? -Infinity : (src instanceof Uint8Array ? 255 : Infinity);
  const tmp = new src.constructor(src.length);
  const out = new src.constructor(src.length);
  const n = Math.max(w, h) + 2 * r;
  const line = new Float64Array(n), g = new Float64Array(n), hb = new Float64Array(n), res = new Float64Array(n);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    line.fill(edge, 0, r);
    for (let x = 0; x < w; x++) line[r + x] = src[row + x];
    line.fill(edge, r + w, w + 2 * r);
    lineFilter(line, w, r, isMax, g, hb, res);
    for (let x = 0; x < w; x++) tmp[row + x] = res[x];
  }
  for (let x = 0; x < w; x++) {
    line.fill(edge, 0, r);
    for (let y = 0; y < h; y++) line[r + y] = tmp[y * w + x];
    line.fill(edge, r + h, h + 2 * r);
    lineFilter(line, h, r, isMax, g, hb, res);
    for (let y = 0; y < h; y++) out[y * w + x] = res[y];
  }
  return out;
}

/** Separable square max filter (dilation) with radius r. Works on Float32 or Uint8. */
export function dilate(src, w, h, r) {
  return morph(src, w, h, r, true);
}

/** Separable square min filter (erosion); pixels outside the image count as max (like OpenCV). */
export function erode(src, w, h, r) {
  return morph(src, w, h, r, false);
}

/** Box blur with radius r (running sums, O(n)), applied `passes` times (~Gaussian). Float32 in/out. */
export function blur(src, w, h, r, passes = 2) {
  const a = Float32Array.from(src);
  if (r <= 0) return a;
  const b = new Float32Array(a.length);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let s = 0;
      for (let k = 0; k <= Math.min(w - 1, r); k++) s += a[row + k];
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
        b[row + x] = s / (x1 - x0 + 1);
        if (x + r + 1 < w) s += a[row + x + r + 1];
        if (x - r >= 0) s -= a[row + x - r];
      }
    }
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = 0; k <= Math.min(h - 1, r); k++) s += b[k * w + x];
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
        a[y * w + x] = s / (y1 - y0 + 1);
        if (y + r + 1 < h) s += b[(y + r + 1) * w + x];
        if (y - r >= 0) s -= b[(y - r) * w + x];
      }
    }
  }
  return a;
}

/**
 * Extend a mask straight down to the bottom of the frame: a person continues
 * below whatever part is detected (a chin sits on a torso). In place on a copy.
 */
export function fillDown(src, w, h) {
  const out = src.slice();
  for (let x = 0; x < w; x++) {
    let m = 0;
    for (let y = 0; y < h; y++) {
      const i = y * w + x;
      if (out[i] > m) m = out[i];
      out[i] = m;
    }
  }
  return out;
}

/**
 * Uint8 mask (255) of pixels that are safely NOT the person, for learning the
 * background: person-like pixels, everything below them, plus a margin are excluded.
 */
/**
 * fillDown, widening as it goes down (a "body cone"): a seated person's
 * shoulders, arms and clothing extend sideways beyond the detected head/torso
 * columns. Each column's coverage spreads sideways by `slope` px per row
 * below where the person starts in that column.
 */
export function fillDownCone(src, w, h, slope = 0.3) {
  const out = new src.constructor(src.length);
  const top = new Int32Array(w).fill(-1); // first person row per column
  for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) if (src[y * w + x]) { top[x] = y; break; }
  for (let x = 0; x < w; x++) {
    if (top[x] < 0) continue;
    const v = 255;
    for (let y = top[x]; y < h; y++) {
      const r = Math.floor((y - top[x]) * slope);
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r), row = y * w;
      for (let k = x0; k <= x1; k++) out[row + k] = v;
    }
  }
  return out;
}

export function backgroundMask(person, w, h, { threshold = 0.1, margin = 10, down = true, slope = 0.3 } = {}) {
  let m = new Uint8Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = person[i] > threshold ? 255 : 0;
  if (down) m = slope > 0 ? fillDownCone(m, w, h, slope) : fillDown(m, w, h);
  m = dilate(m, w, h, margin);
  for (let i = 0; i < m.length; i++) m[i] = 255 - m[i];
  return m;
}

// ---- hand hull ----------------------------------------------------------------

function convexHull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const L = dx * dx + dy * dy;
  let t = L > 0 ? ((px - ax) * dx + (py - ay) * dy) / L : 0;
  t = Math.max(0, Math.min(1, t));
  const qx = ax + t * dx - px, qy = ay + t * dy - py;
  return qx * qx + qy * qy;
}

/**
 * Union `mask` (Float32) with the convex hull of hand landmarks (normalized
 * [x, y] pairs), grown by `grow` x the wrist-to-middle-knuckle distance so it
 * reaches fingertips. The selfie segmenter often misses raised fingers.
 */
export function addHand(mask, w, h, hand, grow = 0.45) {
  if (!hand || hand.length < 10) return mask;
  const pts = hand.map(([x, y]) => [x * w, y * h]);
  const g = Math.max(2, grow * Math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]));
  const hull = convexHull(pts);
  const xs = hull.map((q) => q[0]), ys = hull.map((q) => q[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - g)), x1 = Math.min(w - 1, Math.ceil(Math.max(...xs) + g));
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - g)), y1 = Math.min(h - 1, Math.ceil(Math.max(...ys) + g));
  const out = mask.slice();
  const g2 = g * g;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const px = x + 0.5, py = y + 0.5;
      let inside = hull.length >= 3;
      let best = Infinity;
      for (let i = 0; i < hull.length; i++) {
        const a = hull[i], b = hull[(i + 1) % hull.length];
        if ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]) < 0) inside = false;
        const d = segDist2(px, py, a[0], a[1], b[0], b[1]);
        if (d < best) best = d;
      }
      if (inside || best <= g2) out[y * w + x] = 1;
    }
  }
  return out;
}

// ---- temporal refinement ----------------------------------------------------

/**
 * Raw per-frame masks -> stable, soft-edged mask. Biased toward covering:
 * soft threshold, grow-instantly/shrink-slowly temporal filter, dilation,
 * feather. Port of MaskRefiner.
 */
export class MaskRefiner {
  constructor({ low = 0.2, high = 0.6, decay = 0.8, dilatePx = 3, featherPx = 3 } = {}) {
    Object.assign(this, { low, high, decay, dilatePx, featherPx });
    this._prev = null;
  }

  reset() {
    this._prev = null;
  }

  refine(raw, w, h) {
    const m = new Float32Array(raw.length);
    const k = 1 / (this.high - this.low);
    for (let i = 0; i < m.length; i++) m[i] = Math.min(1, Math.max(0, (raw[i] - this.low) * k));
    if (this._prev && this._prev.length === m.length) {
      for (let i = 0; i < m.length; i++) m[i] = Math.max(m[i], this._prev[i] * this.decay);
    }
    this._prev = m;
    let out = dilate(m, w, h, this.dilatePx);
    out = blur(out, w, h, this.featherPx);
    return out;
  }
}

// ---- difference from the known room ---------------------------------------------

function median256(hist, n) {
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc * 2 >= n) return v;
  }
  return 255;
}

/**
 * Soft mask (0..1) of pixels that differ from the known background, and the
 * fraction of `region` pixels (default: known) that changed. Catches the person
 * regardless of pose or detector failure. A per-channel gain from medians over
 * known pixels compensates global brightness shifts (webcam auto-exposure).
 *
 * frame: RGBA Uint8ClampedArray, plate: Float32Array RGB (3 per pixel),
 * known: Uint8Array (255 = seen).
 */
export function differenceMask(frame, plate, known, w, h, { low = 18, high = 40, region = null } = {}) {
  const n = w * h;
  const hf = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  const hp = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  let nk = 0;
  for (let i = 0; i < n; i++) {
    if (!known[i]) continue;
    nk++;
    for (let c = 0; c < 3; c++) {
      hf[c][frame[i * 4 + c]]++;
      hp[c][Math.min(255, plate[i * 3 + c] | 0)]++;
    }
  }
  const out = new Float32Array(n);
  if (nk < 100) return { mask: out, changed: 0 };
  const gain = [0, 1, 2].map((c) => Math.min(2, Math.max(0.5, (median256(hf[c], nk) + 1) / (median256(hp[c], nk) + 1))));
  const k = 1 / (high - low);
  for (let i = 0; i < n; i++) {
    if (!known[i]) continue;
    let d = 0;
    for (let c = 0; c < 3; c++) {
      const v = Math.abs(frame[i * 4 + c] - plate[i * 3 + c] * gain[c]);
      if (v > d) d = v;
    }
    out[i] = Math.min(1, Math.max(0, (d - low) * k));
  }
  // Light smoothing so sensor noise doesn't speckle, then close small gaps.
  let m = blur(out, w, h, 1, 1);
  m = erode(dilate(m, w, h, 2), w, h, 2);
  for (let i = 0; i < n; i++) if (!known[i]) m[i] = 0;
  let changed = 0, nr = 0;
  for (let i = 0; i < n; i++) {
    if (region ? !region[i] : !known[i]) continue;
    nr++;
    if (m[i] > 0.5) changed++;
  }
  return { mask: m, changed: nr > 0 ? changed / nr : 0 };
}

// ---- hole filling -----------------------------------------------------------------

/**
 * Fill unknown pixels of an RGB float image by pull-push over an image
 * pyramid: holes get smooth colour interpolated from their surroundings (the
 * browser counterpart of OpenCV's Telea inpainting used as the basic fill).
 * Returns a new Float32Array RGB.
 */
export function pullPushFill(rgb, known, w, h) {
  const levels = [{ w, h, c: Float32Array.from(rgb), a: Float32Array.from(known, (v) => (v ? 1 : 0)) }];
  while (levels[levels.length - 1].w > 2 && levels[levels.length - 1].h > 2) {
    const p = levels[levels.length - 1];
    const nw = Math.ceil(p.w / 2), nh = Math.ceil(p.h / 2);
    const c = new Float32Array(nw * nh * 3), a = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let sa = 0, s0 = 0, s1 = 0, s2 = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const sx = Math.min(p.w - 1, 2 * x + dx), sy = Math.min(p.h - 1, 2 * y + dy);
            const j = sy * p.w + sx, wt = p.a[j];
            sa += wt; s0 += p.c[j * 3] * wt; s1 += p.c[j * 3 + 1] * wt; s2 += p.c[j * 3 + 2] * wt;
          }
        }
        const i = y * nw + x;
        if (sa > 0) { c[i * 3] = s0 / sa; c[i * 3 + 1] = s1 / sa; c[i * 3 + 2] = s2 / sa; }
        a[i] = Math.min(1, sa);
      }
    }
    levels.push({ w: nw, h: nh, c, a });
  }
  for (let l = levels.length - 2; l >= 0; l--) {
    const p = levels[l], q = levels[l + 1];
    for (let y = 0; y < p.h; y++) {
      for (let x = 0; x < p.w; x++) {
        const i = y * p.w + x;
        const wt = p.a[i];
        if (wt >= 1) continue;
        // bilinear sample of the coarser level
        const fx = Math.min(q.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5)), fy = Math.min(q.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
        const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(q.w - 1, x0 + 1), y1 = Math.min(q.h - 1, y0 + 1);
        const tx = fx - x0, ty = fy - y0;
        for (let ch = 0; ch < 3; ch++) {
          const v = (q.c[(y0 * q.w + x0) * 3 + ch] * (1 - tx) + q.c[(y0 * q.w + x1) * 3 + ch] * tx) * (1 - ty) +
            (q.c[(y1 * q.w + x0) * 3 + ch] * (1 - tx) + q.c[(y1 * q.w + x1) * 3 + ch] * tx) * ty;
          p.c[i * 3 + ch] = p.c[i * 3 + ch] * wt + v * (1 - wt);
        }
        p.a[i] = 1;
      }
    }
  }
  return levels[0].c;
}

/** Blend weight of the real plate vs the fill: known eroded past the blur reach, then blurred. */
export function softKnown(known, w, h, seam = 3) {
  const e = erode(known, w, h, seam + 1);
  const f = new Float32Array(e.length);
  for (let i = 0; i < f.length; i++) f[i] = e[i] / 255;
  return blur(f, w, h, Math.max(1, Math.round(seam / 2)));
}

/**
 * differenceMask computed at half resolution (4x less work; the difference is
 * a coarse signal and gets dilated/feathered later anyway), mask returned at
 * full resolution (bilinear). Same arguments as differenceMask.
 */
export function differenceMaskHalf(frame, plate, known, w, h, opts = {}) {
  const hw = w >> 1, hh = h >> 1, n = hw * hh;
  const f = new Uint8ClampedArray(n * 4), p = new Float32Array(n * 3), k = new Uint8Array(n);
  const region = opts.region ? new Uint8Array(n) : null;
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const i = y * hw + x;
      const a = 2 * y * w + 2 * x, b = a + 1, c = a + w, d = c + 1;
      for (let ch = 0; ch < 3; ch++) {
        f[i * 4 + ch] = (frame[a * 4 + ch] + frame[b * 4 + ch] + frame[c * 4 + ch] + frame[d * 4 + ch]) >> 2;
        p[i * 3 + ch] = (plate[a * 3 + ch] + plate[b * 3 + ch] + plate[c * 3 + ch] + plate[d * 3 + ch]) * 0.25;
      }
      // Known only if all four are known (unknown plate pixels are meaningless).
      k[i] = known[a] && known[b] && known[c] && known[d] ? 255 : 0;
      if (region) region[i] = opts.region[a] && opts.region[b] && opts.region[c] && opts.region[d] ? 1 : 0;
    }
  }
  const { mask, changed } = differenceMask(f, p, k, hw, hh, { ...opts, region });
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(hh - 1, Math.max(0, (y + 0.5) / 2 - 0.5)), y0 = Math.floor(fy), y1 = Math.min(hh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(hw - 1, Math.max(0, (x + 0.5) / 2 - 0.5)), x0 = Math.floor(fx), x1 = Math.min(hw - 1, x0 + 1), tx = fx - x0;
      out[y * w + x] = known[y * w + x]
        ? (mask[y0 * hw + x0] * (1 - tx) + mask[y0 * hw + x1] * tx) * (1 - ty) + (mask[y1 * hw + x0] * (1 - tx) + mask[y1 * hw + x1] * tx) * ty
        : 0;
    }
  }
  return { mask: out, changed };
}

/**
 * Exposure/colour correction map for the background: per-channel ratio of the
 * live frame to the learned plate, measured where the room is visible right
 * now (known and not the person), extended smoothly into the area behind the
 * person (pull-push), blurred to low frequency and clamped.
 *
 * Webcams keep re-adjusting exposure and white balance (e.g. when a person in
 * dark clothes fills the frame). Visible room pixels follow, but the plate
 * behind the person keeps its old exposure, which shows as a lighter or
 * darker "shadow" patch. Multiplying the plate/fill by this map removes it.
 *
 * Also used to keep a fill current: frame = plate now (frameStride 3),
 * plate = plate when the fill was made, person = null.
 *
 * Returns Float32Array RGB at (w/cell x h/cell).
 */
export function exposureGain(frame, plate, known, person, w, h, { cell = 4, minSamples = 4, clamp = [0.5, 2], frameStride = 4 } = {}) {
  const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell), n = gw * gh;
  const sf = new Float32Array(n * 3), sp = new Float32Array(n * 3), cnt = new Uint16Array(n);
  for (let y = 0; y < h; y++) {
    const gy = (y / cell) | 0;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!known[i] || (person && person[i] > 0.1)) continue;
      const g = gy * gw + ((x / cell) | 0);
      cnt[g]++;
      for (let c = 0; c < 3; c++) { sf[g * 3 + c] += frame[i * frameStride + c]; sp[g * 3 + c] += plate[i * 3 + c]; }
    }
  }
  const ratio = new Float32Array(n * 3), ok = new Uint8Array(n);
  let any = false;
  for (let g = 0; g < n; g++) {
    if (cnt[g] < minSamples) continue;
    ok[g] = 255;
    any = true;
    const bias = 8 * cnt[g]; // keeps near-black pixels from producing huge ratios
    for (let c = 0; c < 3; c++) ratio[g * 3 + c] = (sf[g * 3 + c] + bias) / (sp[g * 3 + c] + bias);
  }
  if (!any) return new Float32Array(n * 3).fill(1);
  const filled = pullPushFill(ratio, ok, gw, gh);
  const out = new Float32Array(n * 3);
  for (let c = 0; c < 3; c++) {
    const ch = new Float32Array(n);
    for (let g = 0; g < n; g++) ch[g] = filled[g * 3 + c];
    const b = blur(ch, gw, gh, 3);
    for (let g = 0; g < n; g++) out[g * 3 + c] = Math.min(clamp[1], Math.max(clamp[0], b[g]));
  }
  return out;
}

/** Bilinear upsampling of a low-res RGB gain map (gw x gh) to w x h (RGB Float32). */
export function upsampleGain(gain, gw, gh, w, h) {
  const out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(gh - 1, Math.max(0, ((y + 0.5) * gh) / h - 0.5)), y0 = Math.floor(fy), y1 = Math.min(gh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(gw - 1, Math.max(0, ((x + 0.5) * gw) / w - 0.5)), x0 = Math.floor(fx), x1 = Math.min(gw - 1, x0 + 1), tx = fx - x0;
      for (let c = 0; c < 3; c++) {
        out[(y * w + x) * 3 + c] =
          (gain[(y0 * gw + x0) * 3 + c] * (1 - tx) + gain[(y0 * gw + x1) * 3 + c] * tx) * (1 - ty) +
          (gain[(y1 * gw + x0) * 3 + c] * (1 - tx) + gain[(y1 * gw + x1) * 3 + c] * tx) * ty;
      }
    }
  }
  return out;
}
