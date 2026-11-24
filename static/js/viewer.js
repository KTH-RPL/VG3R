/* VG3R project page: a small WebGL point-cloud viewer that runs from file:// (classic script, no modules, no external libraries).
   Point clouds are JavaScript files that register base64 data in window.VG3R_PCD (static/pcd/<id>.js), loaded with script tags.
   One canvas holds one or more panels that share one orbit camera (rotate: left drag, pan: right drag or shift + drag,
   zoom: wheel or pinch, double click: reset). Clouds of the trajectory section also carry camera trajectories (d.tracks), drawn as
   wide lines over the points; each trajectory can be shown or hidden. */
(function () {
  'use strict';

  window.VG3R_PCD = window.VG3R_PCD || {};
  var scripts = {};

  function loadScript(src) {
    if (scripts[src]) return scripts[src];
    scripts[src] = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error('cannot load ' + src)); };
      document.head.appendChild(s);
    });
    return scripts[src];
  }

  function b64(str) {
    var bin = atob(str), n = bin.length, out = new Uint8Array(n);
    for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function getCloud(id) {
    return loadScript('static/pcd/' + id + '.js').then(function () {
      var d = window.VG3R_PCD[id];
      if (!d) throw new Error('no data in ' + id);
      if (!d._pos) {
        d._pos = new Uint16Array(b64(d.pos).buffer);
        d._col = b64(d.col);
        d.pos = d.col = null;                       // free the strings
      }
      return d;
    });
  }

  /* ---------- small vector / matrix helpers (column-major 4x4, as WebGL expects) ---------- */
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { var l = Math.sqrt(dot(a, a)) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

  function lookAt(eye, target, up) {
    var f = norm(sub(target, eye)), s = norm(cross(f, up)), u = cross(s, f);
    return [s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot(s, eye), -dot(u, eye), dot(f, eye), 1];
  }

  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  }

  /* ---------- the orbit camera, shared by the panels of one viewer ---------- */
  function OrbitCamera(view) {
    this.set(view);
  }
  OrbitCamera.prototype.set = function (view) {
    // view: {eye, target, up}; the orbit keeps `up` vertical
    this.home = view;
    this.up = norm(view.up);
    var d = sub(view.eye, view.target);
    this.r = Math.sqrt(dot(d, d));
    var dn = norm(d);
    // a horizontal basis (e1, e2) orthogonal to up
    var h = sub(dn, mul(this.up, dot(dn, this.up)));
    if (dot(h, h) < 1e-8) h = norm(cross(this.up, [1, 0, 0]));
    this.e1 = norm(h);
    this.e2 = norm(cross(this.up, this.e1));
    this.yaw = 0;
    this.pitch = Math.asin(Math.max(-1, Math.min(1, dot(dn, this.up))));
    this.target = view.target.slice();
    this.r0 = this.r;
    // framing of the initial view: horizontal field of view `hfov` on an image of aspect `aspect0` (as the paper's renders)
    this.hfov = (view.hfov || 50) * Math.PI / 180;
    this.aspect0 = view.aspect0 || 4 / 3;
  };
  /* vertical field of view of a panel of aspect a that shows at least the framing of the initial view */
  OrbitCamera.prototype.fovFor = function (a) {
    var t = Math.tan(this.hfov / 2);
    return 2 * Math.atan(Math.max(t / this.aspect0, t / a));
  };
  OrbitCamera.prototype.reset = function () { this.set(this.home); };
  OrbitCamera.prototype.eye = function () {
    var c = Math.cos(this.pitch), dir = add(add(mul(this.e1, c * Math.cos(this.yaw)), mul(this.e2, c * Math.sin(this.yaw))),
      mul(this.up, Math.sin(this.pitch)));
    return add(this.target, mul(dir, this.r));
  };
  OrbitCamera.prototype.rotate = function (dx, dy) {
    this.yaw -= dx * 0.008;
    this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.008));
  };
  OrbitCamera.prototype.pan = function (dx, dy, h) {
    var eye = this.eye(), f = norm(sub(this.target, eye)), s = norm(cross(f, this.up)), u = cross(s, f);
    var k = 2 * this.r * Math.tan(this.fov / 2) / Math.max(h, 1);
    this.target = add(this.target, add(mul(s, -dx * k), mul(u, dy * k)));
  };
  OrbitCamera.prototype.zoom = function (f) { this.r = Math.max(this.r0 * 0.02, Math.min(this.r0 * 20, this.r * f)); };
  OrbitCamera.prototype.fov = 50 * Math.PI / 180;

  /* ---------- shaders ---------- */
  var VS_POINTS = [
    'attribute vec3 aPos; attribute vec3 aCol;',
    'uniform vec3 uMin; uniform vec3 uExt; uniform mat4 uMV; uniform mat4 uP; uniform float uPx; uniform float uRef;',
    'uniform float uFade; uniform float uFlat;',
    'varying vec3 vCol;',
    'void main() {',
    '  vec4 mv = uMV * vec4(uMin + aPos * uExt, 1.0);',
    '  gl_Position = uP * mv;',
    '  float s = uFlat > 0.0 ? uPx * uFlat : uPx * uRef / max(-mv.z, 1e-4);',   // uFlat: one size at every depth, as the figures
    '  gl_PointSize = clamp(s, 1.0, 48.0);',
    '  vCol = mix(aCol, vec3(1.0), uFade);',          // faded scene (trajectories), so that the cameras stand out
    '}'].join('\n');
  /* wide lines: every segment is a quad (two triangles) widened in screen space; aSW = (side +1 / -1, width in CSS pixels) */
  var VS_WIDE = [
    'attribute vec3 aA; attribute vec3 aB; attribute vec3 aCol; attribute vec2 aSW;',
    'uniform mat4 uMV; uniform mat4 uP; uniform vec2 uRes; uniform float uDpr;',
    'varying vec3 vCol;',
    'void main() {',
    '  vec4 a = uP * uMV * vec4(aA, 1.0);',
    '  vec4 b = uP * uMV * vec4(aB, 1.0);',
    '  vCol = aCol;',
    '  if (a.w < 1e-5 || b.w < 1e-5) {',                 // a segment that reaches behind the eye is left out
    '    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);',
    '  } else {',
    '    vec2 d = (b.xy / b.w - a.xy / a.w) * uRes;',
    '    float l = length(d);',
    '    vec2 n = l > 1e-6 ? vec2(-d.y, d.x) / l : vec2(0.0, 1.0);',
    '    gl_Position = a + vec4(n * (aSW.x * aSW.y * uDpr) / uRes * a.w, 0.0, 0.0);',
    '  }',
    '}'].join('\n');
  /* round joints of the wide paths */
  var VS_DOTS = [
    'attribute vec3 aPos; attribute vec3 aCol; attribute float aW; uniform mat4 uMV; uniform mat4 uP; uniform float uDpr;',
    'varying vec3 vCol;',
    'void main() { gl_Position = uP * uMV * vec4(aPos, 1.0); gl_PointSize = aW * uDpr; vCol = aCol; }'].join('\n');
  var FS_POINTS = [
    'precision mediump float; varying vec3 vCol;',
    'void main() { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vCol, 1.0); }'].join('\n');
  var VS_LINES = [
    'attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMV; uniform mat4 uP; varying vec3 vCol;',
    'void main() { gl_Position = uP * uMV * vec4(aPos, 1.0); vCol = aCol; }'].join('\n');
  var FS_LINES = 'precision mediump float; varying vec3 vCol; void main() { gl_FragColor = vec4(vCol, 1.0); }';

  function program(gl, vs, fs) {
    function sh(type, src) {
      var s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    var p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    var loc = {};
    ['aPos', 'aCol', 'aA', 'aB', 'aSW', 'aW'].forEach(function (n) { loc[n] = gl.getAttribLocation(p, n); });   // -1: not in p
    ['uMin', 'uExt', 'uMV', 'uP', 'uPx', 'uRef', 'uFade', 'uFlat', 'uRes', 'uDpr'].forEach(function (n) {
      loc[n] = gl.getUniformLocation(p, n);
    });
    return { p: p, loc: loc };
  }

  function attrib(gl, loc, n, stride, offset) {
    if (loc < 0) return;
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, n, gl.FLOAT, false, stride, offset);
  }

  /* point p transformed by M (16 numbers, row-major 4x4) */
  function tf(M, p) {
    return [M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3], M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7],
      M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11]];
  }

  /* segments [[a, b], ...] as wide lines: 6 vertices per segment of (a, b, color, side, width) = 11 floats */
  function wideBuffer(gl, segs, rgb, width) {
    var arr = new Float32Array(segs.length * 66), o = 0;
    function v(A, B, side) {
      arr[o++] = A[0]; arr[o++] = A[1]; arr[o++] = A[2]; arr[o++] = B[0]; arr[o++] = B[1]; arr[o++] = B[2];
      arr[o++] = rgb[0]; arr[o++] = rgb[1]; arr[o++] = rgb[2]; arr[o++] = side; arr[o++] = width;
    }
    segs.forEach(function (s) {
      var a = s[0], b = s[1];                              // corners a+, a-, b-, b+ (at b the normal is reversed)
      v(a, b, 1); v(a, b, -1); v(b, a, 1);
      v(a, b, 1); v(b, a, 1); v(b, a, -1);
    });
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    return { b: buf, n: segs.length * 6 };
  }

  /* round dots of diameter `width` at the points: (position, color, width) = 7 floats */
  function dotBuffer(gl, pts, rgb, width) {
    var arr = new Float32Array(pts.length * 7), o = 0;
    pts.forEach(function (q) {
      arr[o++] = q[0]; arr[o++] = q[1]; arr[o++] = q[2]; arr[o++] = rgb[0]; arr[o++] = rgb[1]; arr[o++] = rgb[2]; arr[o++] = width;
    });
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    return { b: buf, n: pts.length };
  }

  function hexRgb(h) {
    h = h.replace('#', '');
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  }

  /* camera frustums (and the camera path) as line segments: cams = [{c2w: 16 numbers, row-major 4x4}], K = [fx, fy, cx, cy, w, h] */
  function frustumLines(cams, K, scale, colors, path) {
    var pos = [], col = [];
    var w = K[4], h = K[5];
    var corners = [[0, 0], [w, 0], [w, h], [0, h]].map(function (p) {
      return [(p[0] - K[2]) / K[0], (p[1] - K[3]) / K[1], 1];
    });
    function seg(a, b, c) { pos.push(a[0], a[1], a[2], b[0], b[1], b[2]); col.push(c[0], c[1], c[2], c[0], c[1], c[2]); }
    cams.forEach(function (cam, i) {
      if (!cam.show) return;
      var M = cam.c2w, c = colors[i], o = tf(M, [0, 0, 0]);
      var q = corners.map(function (p) { return tf(M, [p[0] * scale, p[1] * scale, p[2] * scale]); });
      for (var k = 0; k < 4; k++) { seg(o, q[k], c); seg(q[k], q[(k + 1) % 4], c); }
    });
    if (path) {
      for (var i = 1; i < cams.length; i++) {
        var a = tf(cams[i - 1].c2w, [0, 0, 0]), b = tf(cams[i].c2w, [0, 0, 0]);
        seg(a, b, colors[i]);
      }
    }
    return { pos: new Float32Array(pos), col: new Float32Array(col) };
  }

  /* time ramp of the paper's figures (dark red at the first frame, yellow at the last) */
  var TIME = ['#7E1F14', '#B23A1E', '#D8692A', '#E99A34', '#F2C23E'].map(hexRgb);
  function timeColor(t) {
    var x = Math.max(0, Math.min(1, t)) * (TIME.length - 1), i = Math.min(Math.floor(x), TIME.length - 2), f = x - i;
    return [0, 1, 2].map(function (k) { return TIME[i][k] * (1 - f) + TIME[i + 1][k] * f; });
  }

  /* ---------- the viewer ---------- */
  function Viewer(container, opts) {
    opts = opts || {};
    this.container = container;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'pcd-canvas';
    container.appendChild(this.canvas);
    this.overlay = document.createElement('div');
    this.overlay.className = 'pcd-overlay';
    container.appendChild(this.overlay);
    this.cols = opts.cols || 1;
    this.bg = hexRgb(opts.bg || '#ffffff');
    this.px = opts.px || 2.0;
    this.panels = [];
    this.cam = null;
    this.dirty = true;
    this.anim = null;
    var gl = this.canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: false }) ||
      this.canvas.getContext('experimental-webgl');
    if (!gl) {
      container.classList.add('no-webgl');
      var msg = document.createElement('div');
      msg.className = 'pcd-msg';
      msg.textContent = 'WebGL is not available in this browser, so the interactive point clouds cannot be shown.';
      container.appendChild(msg);
      this.gl = null;
      return;
    }
    this.gl = gl;
    this.progPts = program(gl, VS_POINTS, FS_POINTS);
    this.progLines = program(gl, VS_LINES, FS_LINES);
    this.progWide = program(gl, VS_WIDE, FS_LINES);
    this.progDots = program(gl, VS_DOTS, FS_POINTS);
    this.frusta = true;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.SCISSOR_TEST);
    this._events();
    var self = this;
    if (window.ResizeObserver) new ResizeObserver(function () { self.dirty = true; }).observe(container);
    window.addEventListener('resize', function () { self.dirty = true; });
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  Viewer.prototype._events = function () {
    var self = this, c = this.canvas, ptrs = {}, last = null, pinch = null;
    c.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    c.addEventListener('pointerdown', function (e) {
      c.setPointerCapture(e.pointerId);
      ptrs[e.pointerId] = { x: e.clientX, y: e.clientY, b: e.button, shift: e.shiftKey };
      self.stopAnim(true);
    });
    c.addEventListener('pointermove', function (e) {
      var p = ptrs[e.pointerId];
      if (!p || !self.cam) return;
      var ids = Object.keys(ptrs);
      if (ids.length === 2) {
        var a = ptrs[ids[0]], b = ptrs[ids[1]];
        var ox = (a.x + b.x) / 2, oy = (a.y + b.y) / 2, od = Math.hypot(a.x - b.x, a.y - b.y);
        p.x = e.clientX; p.y = e.clientY;
        var nx = (a.x + b.x) / 2, ny = (a.y + b.y) / 2, nd = Math.hypot(a.x - b.x, a.y - b.y);
        if (od > 0) self.cam.zoom(od / Math.max(nd, 1));
        self.cam.pan(nx - ox, ny - oy, self._panelH());
      } else {
        var dx = e.clientX - p.x, dy = e.clientY - p.y;
        p.x = e.clientX; p.y = e.clientY;
        if (p.b === 2 || p.b === 1 || p.shift || e.shiftKey) self.cam.pan(dx, dy, self._panelH());
        else self.cam.rotate(dx, dy);
      }
      self.dirty = true;
      self._sync();
    });
    function up(e) { delete ptrs[e.pointerId]; }
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('wheel', function (e) {
      if (!self.cam) return;
      e.preventDefault();
      self.cam.zoom(Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
      self.dirty = true;
      self._sync();
    }, { passive: false });
    c.addEventListener('dblclick', function () { self.resetView(); });
  };

  Viewer.prototype._panelH = function () {
    var rows = Math.ceil(Math.max(this.panels.length, 1) / this.cols);
    return this.canvas.clientHeight / rows;
  };

  /* viewers linked to this one follow its camera (comparison grids built of several viewers) */
  Viewer.prototype.link = function (others) { this.linked = others; };
  Viewer.prototype._sync = function () {
    if (!this.linked) return;
    var cam = this.cam;
    this.linked.forEach(function (v) {
      if (v === this || !v.cam) return;
      v.cam.target = cam.target.slice(); v.cam.r = cam.r; v.cam.yaw = cam.yaw; v.cam.pitch = cam.pitch;
      v.dirty = true;
    }, this);
  };

  Viewer.prototype.resetView = function () {
    if (this.cam) { this.cam.reset(); this.dirty = true; this._sync(); }
  };

  Viewer.prototype.setPointSize = function (px) { this.px = px; this.dirty = true; };

  Viewer.prototype.clear = function () {
    var gl = this.gl;
    this.panels.forEach(function (p) {
      if (!gl) return;
      if (p.bPos) gl.deleteBuffer(p.bPos);
      if (p.bCol) gl.deleteBuffer(p.bCol);
      if (p.lPos) gl.deleteBuffer(p.lPos);
      if (p.lCol) gl.deleteBuffer(p.lCol);
      (p.tracks || []).forEach(function (t) {
        [t.path, t.fr, t.dots].forEach(function (g) { if (g && g.b) gl.deleteBuffer(g.b); });
      });
    });
    this.panels = [];
    this.overlay.innerHTML = '';
    this.dirty = true;
  };

  /* clouds: [{id, label, ours}] in panel order; options: {cols, view (else the view of the first cloud), frustums: true} */
  Viewer.prototype.show = function (clouds, options) {
    var self = this;
    options = options || {};
    if (options.cols) this.cols = options.cols;
    this.token = (this.token || 0) + 1;
    var token = this.token;
    this.container.classList.add('loading');
    return Promise.all(clouds.map(function (c) { return getCloud(c.id); })).then(function (data) {
      if (token !== self.token) return;                     // a later call replaced this one
      self.clear();
      var gl = self.gl;
      data.forEach(function (d, i) {
        var p = { d: d, label: clouds[i].label, ours: !!clouds[i].ours, count: d.n, frames: d.frames || null };
        if (gl) {
          p.bPos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, p.bPos); gl.bufferData(gl.ARRAY_BUFFER, d._pos, gl.STATIC_DRAW);
          p.bCol = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, p.bCol); gl.bufferData(gl.ARRAY_BUFFER, d._col, gl.STATIC_DRAW);
          if (d.cams && d.cams.length && options.frustums !== false) self._camBuffers(p, d.cams.length);
          if (d.tracks) self._trackBuffers(p);
        }
        self.panels.push(p);
        var lab = document.createElement('div');
        lab.className = 'pcd-label' + (p.ours ? ' ours' : '');
        lab.textContent = p.label || '';
        self.overlay.appendChild(lab);
        p.labelEl = lab;
      });
      var v = options.view || data[0].view;
      if (!self.cam || options.keepView !== true) self.cam = new OrbitCamera(v);
      self.container.classList.remove('loading');
      self.dirty = true;
      if (options.animate) self.play();
      self._layoutLabels();
    }).catch(function (err) {
      self.container.classList.remove('loading');
      var msg = document.createElement('div');
      msg.className = 'pcd-msg';
      msg.textContent = String(err);
      self.overlay.appendChild(msg);
    });
  };

  Viewer.prototype._camBuffers = function (p, upto) {
    var gl = this.gl, d = p.d, n = d.cams.length;
    var cams = d.cams.map(function (M, i) { return { c2w: M, show: i < upto && (i % (d.camEvery || 6) === 0 || i === upto - 1) }; });
    var colors = d.cams.map(function (M, i) { return d.camColor ? hexRgb(d.camColor) : timeColor(i / Math.max(n - 1, 1)); });
    var L = frustumLines(cams.slice(0, upto), d.K, d.camScale, colors.slice(0, upto), true);
    if (!p.lPos) { p.lPos = gl.createBuffer(); p.lCol = gl.createBuffer(); }
    gl.bindBuffer(gl.ARRAY_BUFFER, p.lPos); gl.bufferData(gl.ARRAY_BUFFER, L.pos, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, p.lCol); gl.bufferData(gl.ARRAY_BUFFER, L.col, gl.DYNAMIC_DRAW);
    p.lCount = L.pos.length / 3;
  };

  /* camera trajectories of a cloud (d.tracks: [{name, color, c2w: one row-major 4x4 per frame, ours, gt}], in drawing order): the
     path through the camera centers and the frustums of the frames d.frusta (depth d.camScale, intrinsics d.K), as wide lines over
     the points; widths in CSS pixels, in the proportions of the paper's figures (VG3R 4.5, the other methods 3.2, ground truth 2) */
  Viewer.prototype._trackBuffers = function (p) {
    var gl = this.gl, d = p.d, K = d.K, s = d.camScale;
    var corners = [[0, 0], [K[4], 0], [K[4], K[5]], [0, K[5]]].map(function (q) {
      return [(q[0] - K[2]) / K[0] * s, (q[1] - K[3]) / K[1] * s, s];
    });
    p.tracks = d.tracks.map(function (t) {
      var rgb = hexRgb(t.color), wPath = t.ours ? 4.5 : (t.gt ? 2.0 : 3.2), wFr = 1.6;
      var cen = t.c2w.map(function (M) { return tf(M, [0, 0, 0]); }), segs = [], fr = [];
      for (var i = 1; i < cen.length; i++) segs.push([cen[i - 1], cen[i]]);
      (d.frusta || []).forEach(function (i) {
        if (i >= t.c2w.length) return;
        var o = cen[i], q = corners.map(function (c) { return tf(t.c2w[i], c); });
        for (var k = 0; k < 4; k++) { fr.push([o, q[k]]); fr.push([q[k], q[(k + 1) % 4]]); }
      });
      return { name: t.name, visible: true, path: wideBuffer(gl, segs, rgb, wPath), fr: wideBuffer(gl, fr, rgb, wFr),
        dots: dotBuffer(gl, cen, rgb, wPath) };
    });
  };

  Viewer.prototype.setTrackVisible = function (name, on) {
    this.panels.forEach(function (p) { (p.tracks || []).forEach(function (t) { if (t.name === name) t.visible = !!on; }); });
    this.dirty = true;
  };

  Viewer.prototype.setFrusta = function (on) { this.frusta = !!on; this.dirty = true; };

  /* the trajectories over the points (no depth test, so that the scene never hides them), in order: the ground truth last */
  Viewer.prototype._drawTracks = function (p, MV, P, w, h, dpr) {
    var gl = this.gl, pw = this.progWide, L = pw.loc, pd = this.progDots, LD = pd.loc, self = this;
    gl.disable(gl.DEPTH_TEST);
    p.tracks.forEach(function (t) {
      if (!t.visible) return;
      gl.useProgram(pw.p);
      gl.uniformMatrix4fv(L.uMV, false, MV);
      gl.uniformMatrix4fv(L.uP, false, P);
      gl.uniform2f(L.uRes, w, h);
      gl.uniform1f(L.uDpr, dpr);
      [self.frusta ? t.fr : null, t.path].forEach(function (g) {
        if (!g || !g.n) return;
        gl.bindBuffer(gl.ARRAY_BUFFER, g.b);
        attrib(gl, L.aA, 3, 44, 0); attrib(gl, L.aB, 3, 44, 12); attrib(gl, L.aCol, 3, 44, 24); attrib(gl, L.aSW, 2, 44, 36);
        gl.drawArrays(gl.TRIANGLES, 0, g.n);
      });
      if (t.dots.n) {
        gl.useProgram(pd.p);
        gl.uniformMatrix4fv(LD.uMV, false, MV);
        gl.uniformMatrix4fv(LD.uP, false, P);
        gl.uniform1f(LD.uDpr, dpr);
        gl.bindBuffer(gl.ARRAY_BUFFER, t.dots.b);
        attrib(gl, LD.aPos, 3, 28, 0); attrib(gl, LD.aCol, 3, 28, 12); attrib(gl, LD.aW, 1, 28, 24);
        gl.drawArrays(gl.POINTS, 0, t.dots.n);
      }
    });
    [L.aA, L.aB, L.aCol, L.aSW, LD.aPos, LD.aCol, LD.aW].forEach(function (l) { if (l >= 0) gl.disableVertexAttribArray(l); });
    gl.enable(gl.DEPTH_TEST);
  };

  /* reveal the frames one by one (clouds whose points are sorted by frame carry `frames`, the cumulative counts) */
  Viewer.prototype.play = function (seconds) {
    var self = this, any = this.panels.some(function (p) { return p.frames; });
    if (!any) return;
    var T = (seconds || 4.0) * 1000, t0 = performance.now();
    this.anim = function (now) {
      var t = Math.min(1, (now - t0) / T);
      self.panels.forEach(function (p) {
        if (!p.frames) return;
        var nf = p.frames.length, k = Math.max(1, Math.round(t * nf));
        p.count = p.frames[k - 1];
        if (p.d.cams && self.gl) self._camBuffers(p, Math.min(k, p.d.cams.length));
      });
      self.dirty = true;
      if (t >= 1) self.anim = null;
    };
  };

  Viewer.prototype.stopAnim = function (complete) {
    if (!this.anim) return;
    this.anim = null;
    if (complete) {
      var self = this;
      this.panels.forEach(function (p) {
        p.count = p.d.n;
        if (p.d.cams && self.gl) self._camBuffers(p, p.d.cams.length);
      });
      this.dirty = true;
    }
  };

  Viewer.prototype._layoutLabels = function () {
    var n = this.panels.length, cols = Math.min(this.cols, Math.max(n, 1)), rows = Math.ceil(n / cols);
    this.panels.forEach(function (p, i) {
      var cx = i % cols, cy = Math.floor(i / cols);
      p.labelEl.style.left = (100 * cx / cols) + '%';
      p.labelEl.style.top = (100 * cy / rows) + '%';
      p.labelEl.style.width = (100 / cols) + '%';
    });
  };

  Viewer.prototype._loop = function (now) {
    requestAnimationFrame(this._loop);
    if (this.anim) this.anim(now);
    if (!this.dirty || !this.gl) return;
    this.dirty = false;
    this._draw();
  };

  Viewer.prototype._draw = function () {
    var gl = this.gl, c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = Math.max(1, Math.round(c.clientWidth * dpr)), H = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    gl.viewport(0, 0, W, H);
    gl.scissor(0, 0, W, H);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.cam || !this.panels.length) return;
    var n = this.panels.length, cols = Math.min(this.cols, n), rows = Math.ceil(n / cols);
    var pw = W / cols, ph = H / rows;
    var eye = this.cam.eye(), MV = lookAt(eye, this.cam.target, this.cam.up);
    this.cam.fov = this.cam.fovFor(pw / ph);
    var P = perspective(this.cam.fov, pw / ph, this.cam.r * 0.01, this.cam.r * 100);
    var ref = this.cam.r0 * (ph / 600);                       // point size in device pixels at the initial distance, per 600 px of panel
    for (var i = 0; i < n; i++) {
      var p = this.panels[i], x = Math.round((i % cols) * pw), y = Math.round(H - (Math.floor(i / cols) + 1) * ph);
      gl.viewport(x, y, Math.round(pw), Math.round(ph));
      gl.scissor(x, y, Math.round(pw), Math.round(ph));
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      var pr = this.progPts, L = pr.loc, d = p.d;
      gl.useProgram(pr.p);
      gl.uniform3fv(L.uMin, d.bmin);
      gl.uniform3fv(L.uExt, [d.bmax[0] - d.bmin[0], d.bmax[1] - d.bmin[1], d.bmax[2] - d.bmin[2]]);
      gl.uniformMatrix4fv(L.uMV, false, MV);
      gl.uniformMatrix4fv(L.uP, false, P);
      gl.uniform1f(L.uPx, this.px * (d.px || 1));
      gl.uniform1f(L.uRef, ref);
      gl.uniform1f(L.uFade, d.fade || 0);
      gl.uniform1f(L.uFlat, d.flat ? d.flat * dpr : 0);     // d.flat: CSS pixels per unit of the point-size slider
      gl.bindBuffer(gl.ARRAY_BUFFER, p.bPos);
      gl.enableVertexAttribArray(L.aPos);
      gl.vertexAttribPointer(L.aPos, 3, gl.UNSIGNED_SHORT, true, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, p.bCol);
      gl.enableVertexAttribArray(L.aCol);
      gl.vertexAttribPointer(L.aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.drawArrays(gl.POINTS, 0, p.count);
      if (p.lPos && p.lCount) {
        var pl = this.progLines, LL = pl.loc;
        gl.useProgram(pl.p);
        gl.uniformMatrix4fv(LL.uMV, false, MV);
        gl.uniformMatrix4fv(LL.uP, false, P);
        gl.bindBuffer(gl.ARRAY_BUFFER, p.lPos);
        gl.enableVertexAttribArray(LL.aPos);
        gl.vertexAttribPointer(LL.aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, p.lCol);
        gl.enableVertexAttribArray(LL.aCol);
        gl.vertexAttribPointer(LL.aCol, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, p.lCount);
      }
      if (p.tracks) this._drawTracks(p, MV, P, Math.round(pw), Math.round(ph), dpr);
    }
  };

  window.VG3RViewer = Viewer;
  window.VG3RLoadScript = loadScript;
})();
