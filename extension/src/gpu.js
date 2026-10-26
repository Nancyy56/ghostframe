// Full-resolution GPU work (WebGL2 on an OffscreenCanvas):
//  - the learned background plate (half-float when available, so the slow
//    running average doesn't stall on 8-bit rounding),
//  - applying the low-resolution learning masks from BackgroundModel,
//  - compositing: frame / plate / fill blended by the person mask.
// Low-res masks are uploaded as R8 textures and upsampled with LINEAR
// filtering (smooth edges instead of blocks).

const VS = `#version 300 es
in vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const LEARN_FS = `#version 300 es
precision highp float;
uniform sampler2D uFrame, uPlate, uNew, uUpd, uGain; uniform vec2 uSize; uniform float uRate;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  vec4 p = texture(uPlate, uv);
  // Learn at the plate's reference exposure: undo the camera's current gain.
  vec4 f = vec4(texture(uFrame, uv).rgb / max(texture(uGain, uv).rgb * 2.0, vec3(0.05)), 1.0);
  if (texture(uNew, uv).r > 0.5) o = f;
  else if (texture(uUpd, uv).r > 0.5) o = mix(p, f, uRate);
  else o = p;
}`;

const COMPOSITE_FS = `#version 300 es
precision highp float;
uniform sampler2D uFrame, uPlate, uFill, uSoft, uMask, uGain; uniform vec2 uSize; uniform float uStrength;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uSize; uv.y = 1.0 - uv.y;   // canvas is bottom-up, textures top-down
  vec4 f = texture(uFrame, uv);
  // Plate and fill are stored at a reference exposure; show them at the
  // camera's current exposure / white balance.
  vec3 base = mix(texture(uFill, uv).rgb, texture(uPlate, uv).rgb, texture(uSoft, uv).r);
  base = clamp(base * texture(uGain, uv).rgb * 2.0, 0.0, 1.0);
  float m = clamp(texture(uMask, uv).r * uStrength, 0.0, 1.0);
  o = vec4(mix(f.rgb, base, m), 1.0);
}`;

const COPY_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc; uniform vec2 uSize; out vec4 o;
void main() { o = texture(uSrc, gl_FragCoord.xy / uSize); }`;

export class Compositor {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.canvas = new OffscreenCanvas(width, height);
    const gl = this.canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    this.float = !!gl.getExtension("EXT_color_buffer_float");
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.learnProg = this._program(LEARN_FS);
    this.compProg = this._program(COMPOSITE_FS);
    this.copyProg = this._program(COPY_FS);

    this.frameTex = this._tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, width, height);
    const pf = this.float ? [gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT] : [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE];
    this.plate = [0, 1].map(() => {
      const tex = this._tex(pf[0], pf[1], pf[2], width, height);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fb };
    });
    this.cur = 0;
    this.readTex = this._tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, width, height);
    this.readFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.readFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.readTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.newTex = this._tex(gl.R8, gl.RED, gl.UNSIGNED_BYTE, 1, 1);
    this.updTex = this._tex(gl.R8, gl.RED, gl.UNSIGNED_BYTE, 1, 1);
    this.maskTex = this._tex(gl.R8, gl.RED, gl.UNSIGNED_BYTE, 1, 1);
    this.softTex = this._tex(gl.R8, gl.RED, gl.UNSIGNED_BYTE, 1, 1);
    this.fillTex = this._tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, 1, 1);
    this.gainTex = this._tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, 1, 1);
    this.setGain(new Float32Array([1, 1, 1]), 1, 1);
  }

  _program(fs) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "p");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
  }

  _tex(internal, format, type, w, h) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  _uploadR8(tex, data, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  }

  _draw(prog, textures, uniforms, fb) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(prog.p);
    let unit = 0;
    for (const [name, tex] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(prog.u[name], unit++);
    }
    gl.uniform2f(prog.u.uSize, this.width, this.height);
    for (const [name, v] of Object.entries(uniforms)) gl.uniform1f(prog.u[name], v);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /** Upload the current camera frame (ImageBitmap/VideoFrame/canvas). */
  setFrame(source) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  /** Apply BackgroundModel.learn() masks (low-res Uint8) to the full-res plate. */
  learn(newMask, updateMask, lw, lh, rate) {
    this._uploadR8(this.newTex, newMask, lw, lh);
    this._uploadR8(this.updTex, updateMask, lw, lh);
    const src = this.plate[this.cur], dst = this.plate[1 - this.cur];
    this._draw(this.learnProg, { uFrame: this.frameTex, uPlate: src.tex, uNew: this.newTex, uUpd: this.updTex, uGain: this.gainTex },
      { uRate: rate }, dst.fb);
    this.cur = 1 - this.cur;
  }

  /** Blend weight of plate vs fill (Float32 0..1, low-res). */
  setSoft(soft, lw, lh) {
    this._uploadR8(this.softTex, Uint8Array.from(soft, (v) => Math.round(v * 255)), lw, lh);
  }

  /** Guess for never-seen pixels: low-res RGB float (basic fill) or an ImageBitmap (realistic fill). */
  setFill(fill, lw, lh) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.fillTex);
    if (fill instanceof Float32Array) {
      const rgba = new Uint8Array(lw * lh * 4);
      for (let i = 0; i < lw * lh; i++) {
        rgba[i * 4] = fill[i * 3]; rgba[i * 4 + 1] = fill[i * 3 + 1]; rgba[i * 4 + 2] = fill[i * 3 + 2]; rgba[i * 4 + 3] = 255;
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lw, lh, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, fill);
    }
  }

  /** Camera's current exposure relative to the plate's reference (Float32 RGB gains, low-res; 0..2). */
  setGain(gain, gw, gh) {
    const gl = this.gl, n = gw * gh, rgba = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) rgba[i * 4 + c] = Math.max(0, Math.min(255, Math.round(gain[i * 3 + c] * 127.5)));
      rgba[i * 4 + 3] = 255;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.gainTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gw, gh, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  }

  /** Composite into the canvas and return it as an ImageBitmap. mask: low-res Float32 0..1. */
  composite(mask, lw, lh, strength) {
    this._uploadR8(this.maskTex, Uint8Array.from(mask, (v) => Math.round(v * 255)), lw, lh);
    this._draw(this.compProg, {
      uFrame: this.frameTex, uPlate: this.plate[this.cur].tex, uFill: this.fillTex, uSoft: this.softTex, uMask: this.maskTex,
      uGain: this.gainTex,
    }, { uStrength: strength }, null);
    return this.canvas.transferToImageBitmap();
  }

  /** Full-res plate as RGBA bytes (row 0 = top), for saving. */
  readPlate() {
    const gl = this.gl;
    this._draw(this.copyProg, { uSrc: this.plate[this.cur].tex }, {}, this.readFb);
    const out = new Uint8Array(this.width * this.height * 4);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }

  /** Restore a saved plate (ImageBitmap at full res). */
  loadPlate(bitmap) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.readTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    this._draw(this.copyProg, { uSrc: this.readTex }, {}, this.plate[this.cur].fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose() {
    const ext = this.gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
  }
}
