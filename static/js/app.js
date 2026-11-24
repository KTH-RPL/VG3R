/* VG3R project page: builds the gallery, the comparison tabs, the video-depth tabs and the trajectory figures from
   window.VG3R_MANIFEST (static/data/manifest.js, written by build_project_page.py). */
(function () {
  'use strict';
  var M = window.VG3R_MANIFEST || {};

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* abstract */
  if (M.abstract) document.getElementById('abstract').innerHTML = M.abstract;

  /* ---------- gallery ---------- */
  var gv = new window.VG3RViewer(document.getElementById('gallery-viewer'), { cols: 1, px: 3 });
  var thumbs = document.getElementById('gallery-thumbs');
  var gInfo = document.getElementById('gallery-info');
  function openGallery(i) {
    var g = M.gallery[i];
    Array.prototype.forEach.call(thumbs.children, function (t, k) { t.classList.toggle('active', k === i); });
    gInfo.textContent = g.title + (g.note ? ' · ' + g.note : '');
    gv.show([{ id: g.id, label: '' }], { animate: true });
  }
  (M.gallery || []).forEach(function (g, i) {
    var b = el('button', 'thumb');
    b.type = 'button';
    var im = el('img');
    im.src = g.thumb;
    im.alt = g.title;
    im.loading = 'lazy';
    b.appendChild(im);
    b.appendChild(el('span', null, g.title));
    b.addEventListener('click', function () { openGallery(i); });
    thumbs.appendChild(b);
  });
  document.getElementById('gallery-play').addEventListener('click', function () { gv.stopAnim(false); gv.play(); });
  document.getElementById('gallery-reset').addEventListener('click', function () { gv.resetView(); });
  document.getElementById('gallery-size').addEventListener('input', function (e) { gv.setPointSize(parseFloat(e.target.value)); });

  /* ---------- comparisons ---------- */
  var cv = new window.VG3RViewer(document.getElementById('compare-viewer'), { cols: 4, px: 3 });
  var groupsEl = document.getElementById('compare-groups'), clipsEl = document.getElementById('compare-clips');
  var cInfo = document.getElementById('compare-info');
  var curGroup = 0;
  function openClip(gi, ci) {
    var grp = M.compare[gi], clip = grp.clips[ci];
    Array.prototype.forEach.call(clipsEl.children, function (t, k) { t.classList.toggle('active', k === ci); });
    cInfo.textContent = clip.note || '';
    var cols = grp.cols || Math.min(4, clip.panels.length);
    cv.show(clip.panels, { cols: cols, view: clip.view });
  }
  function openGroup(gi) {
    curGroup = gi;
    Array.prototype.forEach.call(groupsEl.children, function (t, k) { t.classList.toggle('active', k === gi); });
    clipsEl.innerHTML = '';
    M.compare[gi].clips.forEach(function (c, ci) {
      var b = el('button', 'tab', c.title);
      b.type = 'button';
      b.addEventListener('click', function () { openClip(gi, ci); });
      clipsEl.appendChild(b);
    });
    openClip(gi, 0);
  }
  (M.compare || []).forEach(function (g, gi) {
    var b = el('button', 'tab', g.title);
    b.type = 'button';
    b.addEventListener('click', function () { openGroup(gi); });
    groupsEl.appendChild(b);
  });
  document.getElementById('compare-reset').addEventListener('click', function () { cv.resetView(); });
  document.getElementById('compare-size').addEventListener('input', function (e) { cv.setPointSize(parseFloat(e.target.value)); });

  /* ---------- video depth ---------- */
  var vdTabs = document.getElementById('vd-tabs'), vdVideo = document.getElementById('vd-video'), vdCap = document.getElementById('vd-caption');
  function openVd(i) {
    var v = M.videodepth[i];
    Array.prototype.forEach.call(vdTabs.children, function (t, k) { t.classList.toggle('active', k === i); });
    vdVideo.src = v.src;
    if (v.poster) vdVideo.poster = v.poster;
    vdCap.textContent = v.caption || '';
    var p = vdVideo.play();
    if (p && p.catch) p.catch(function () {});
  }
  (M.videodepth || []).forEach(function (v, i) {
    var b = el('button', 'tab', v.title);
    b.type = 'button';
    b.addEventListener('click', function () { openVd(i); });
    vdTabs.appendChild(b);
  });

  /* ---------- trajectories ---------- */
  /* interactive: one sequence at a time; every trajectory can be shown or hidden (the choice is kept across sequences) */
  var T3 = M.traj3d || [], tvEl = document.getElementById('traj-viewer'), openTrajGroup = null;
  if (T3.length && tvEl) {
    var tv = new window.VG3RViewer(tvEl, { cols: 1, px: 2.5 });
    var tGroupsEl = document.getElementById('traj-groups'), tClipsEl = document.getElementById('traj-clips');
    var tChipsEl = document.getElementById('traj-chips'), tInfo = document.getElementById('traj-info');
    var hiddenTracks = {};
    var trajChips = function (clip) {
      tChipsEl.innerHTML = '';
      clip.tracks.forEach(function (t) {
        var b = el('button', 'chip' + (t.ours ? ' ours' : ''));
        b.type = 'button';
        var sw = el('span', 'sw');
        sw.style.background = t.color;
        b.appendChild(sw);
        b.appendChild(el('span', null, t.name));
        if (t.ate != null) b.appendChild(el('span', 'ate', 'ATE ' + t.ate.toFixed(3)));
        var sync = function () {
          var on = !hiddenTracks[t.name];
          b.classList.toggle('off', !on);
          b.setAttribute('aria-pressed', on ? 'true' : 'false');
          b.title = (on ? 'Hide ' : 'Show ') + t.name;
        };
        sync();
        b.addEventListener('click', function () {
          hiddenTracks[t.name] = !hiddenTracks[t.name];
          sync();
          tv.setTrackVisible(t.name, !hiddenTracks[t.name]);
        });
        tChipsEl.appendChild(b);
      });
    };
    var openTrajClip = function (gi, ci) {
      var clip = T3[gi].clips[ci];
      Array.prototype.forEach.call(tClipsEl.children, function (t, k) { t.classList.toggle('active', k === ci); });
      trajChips(clip);
      tInfo.textContent = T3[gi].title + ' ' + clip.title + ' · ' + clip.note;
      tv.show([{ id: clip.id, label: '' }]).then(function () {
        clip.tracks.forEach(function (t) { tv.setTrackVisible(t.name, !hiddenTracks[t.name]); });
      });
    };
    openTrajGroup = function (gi) {
      Array.prototype.forEach.call(tGroupsEl.children, function (t, k) { t.classList.toggle('active', k === gi); });
      tClipsEl.innerHTML = '';
      T3[gi].clips.forEach(function (c, ci) {
        var b = el('button', 'tab', c.title);
        b.type = 'button';
        b.addEventListener('click', function () { openTrajClip(gi, ci); });
        tClipsEl.appendChild(b);
      });
      openTrajClip(gi, 0);
    };
    T3.forEach(function (g, gi) {
      var b = el('button', 'tab', g.title);
      b.type = 'button';
      b.addEventListener('click', function () { openTrajGroup(gi); });
      tGroupsEl.appendChild(b);
    });
    document.getElementById('traj-frusta').addEventListener('change', function (e) { tv.setFrusta(e.target.checked); });
    document.getElementById('traj-reset').addEventListener('click', function () { tv.resetView(); });
    document.getElementById('traj-size').addEventListener('input', function (e) { tv.setPointSize(parseFloat(e.target.value)); });
  }

  /* the figures of the paper */
  var tr = document.getElementById('traj-imgs');
  (M.trajectories || []).forEach(function (t) {
    var f = el('figure', 'traj');
    var im = el('img');
    im.src = t.src;
    im.alt = t.caption || 'camera trajectories';
    im.loading = 'lazy';
    f.appendChild(im);
    if (t.caption) f.appendChild(el('figcaption', 'caption', t.caption));
    tr.appendChild(f);
  });

  /* ---------- figures: a magnifier on hover, a click enlarges them (scroll or pinch to zoom further, drag to pan, click the image
     to zoom in or out; Esc, the close button or a click outside the image closes the enlarged view) ---------- */
  var LB = null;
  var lightbox = function () {
    if (LB) return LB;
    var box = el('div', 'lb'), stage = el('div', 'lb-stage'), img = el('img', 'lb-img'), close = el('button', 'lb-close', '\u00d7');
    var hint = el('div', 'lb-hint', 'Scroll or pinch to zoom, drag to pan, click the image to zoom in or out. ' +
      'Esc or a click outside the image closes this view.');
    box.hidden = true;
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');
    close.type = 'button';
    close.setAttribute('aria-label', 'Close the enlarged view');
    img.alt = '';
    img.draggable = false;
    stage.appendChild(img);
    box.appendChild(stage);
    box.appendChild(close);
    box.appendChild(hint);
    document.body.appendChild(box);
    var s = 1, tx = 0, ty = 0, ptrs = {}, moved = 0, downOn = null, opener = null, zin = 2.5;
    var apply = function () {                              // the pan is clamped so that the image stays over the stage
      var r = stage.getBoundingClientRect();
      var mx = Math.max(0, (img.offsetWidth * s - r.width) / 2), my = Math.max(0, (img.offsetHeight * s - r.height) / 2);
      tx = Math.max(-mx, Math.min(mx, tx));
      ty = Math.max(-my, Math.min(my, ty));
      img.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + s + ')';
      box.classList.toggle('zoomed', s > 1.001);
    };
    var zoomAt = function (ns, cx, cy) {                   // keeps the point under (cx, cy) in place
      ns = Math.max(1, Math.min(8, ns));
      var r = stage.getBoundingClientRect(), ox = cx - (r.left + r.width / 2), oy = cy - (r.top + r.height / 2);
      tx = ox - (ox - tx) * ns / s;
      ty = oy - (oy - ty) * ns / s;
      s = ns;
      apply();
    };
    var hide = function () {
      box.hidden = true;
      ptrs = {};
      document.documentElement.classList.remove('lb-open');
      img.removeAttribute('src');
      if (opener) opener.focus();
    };
    close.addEventListener('click', hide);
    document.addEventListener('keydown', function (e) { if (!box.hidden && e.key === 'Escape') hide(); });
    stage.addEventListener('wheel', function (e) {
      e.preventDefault();
      zoomAt(s * Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.002)), e.clientX, e.clientY);
    }, { passive: false });
    stage.addEventListener('pointerdown', function (e) {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      e.preventDefault();
      ptrs[e.pointerId] = { x: e.clientX, y: e.clientY };
      if (Object.keys(ptrs).length === 1) { moved = 0; downOn = e.target; }
    });
    window.addEventListener('pointermove', function (e) {
      var p = ptrs[e.pointerId];
      if (!p || box.hidden) return;
      var ids = Object.keys(ptrs);
      if (ids.length === 2) {                               // pinch
        var a = ptrs[ids[0]], b = ptrs[ids[1]], d0 = Math.hypot(a.x - b.x, a.y - b.y);
        p.x = e.clientX; p.y = e.clientY;
        var d1 = Math.hypot(a.x - b.x, a.y - b.y);
        if (d0 > 0) zoomAt(s * d1 / d0, (a.x + b.x) / 2, (a.y + b.y) / 2);
        moved += 10;
        return;
      }
      var dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      if (s > 1.001) { tx += dx; ty += dy; apply(); }
    });
    var up = function (e) {
      if (!ptrs[e.pointerId]) return;
      delete ptrs[e.pointerId];
      if (Object.keys(ptrs).length || moved > 5 || box.hidden) return;   // a drag or a pinch, not a click
      if (downOn === img) zoomAt(s > 1.001 ? 1 : zin, e.clientX, e.clientY);
      else hide();
    };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', function (e) { delete ptrs[e.pointerId]; });
    window.addEventListener('blur', function () { ptrs = {}; });     // a button released outside the window
    /* rel: the clicked point of the figure (fractions of its width and height). A wide strip (the trajectory figures) opens zoomed
       on that point, at its own resolution (or so that it fills three quarters of the screen height); other figures open whole. */
    var first = function (rel) {
      var r = stage.getBoundingClientRect(), fw = img.offsetWidth, fh = img.offsetHeight;
      if (!fw || !fh) return;
      var wide = img.naturalWidth / img.naturalHeight > 5;
      zin = wide ? Math.max(1.5, Math.min(8, img.naturalWidth / fw, 0.75 * r.height / fh)) : 2.5;
      if (wide && rel) {
        s = zin;
        tx = -(rel[0] - 0.5) * fw * s;
        ty = -(rel[1] - 0.5) * fh * s;
        apply();
      }
    };
    LB = {
      open: function (src, alt, from, rel) {
        opener = from || null;
        s = 1; tx = 0; ty = 0; ptrs = {}; zin = 2.5;
        img.style.transform = '';
        box.classList.remove('zoomed');
        box.hidden = false;
        document.documentElement.classList.add('lb-open');
        img.onload = function () { first(rel); };
        img.alt = alt || '';
        img.src = src;
        if (img.complete && img.naturalWidth) first(rel);
        close.focus();
      }
    };
    return LB;
  };
  var zoomable = function (img) {
    var w = el('div', 'zoomable');
    img.parentNode.insertBefore(w, img);
    w.appendChild(img);
    var badge = el('span', 'zoom-badge');
    badge.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"></circle>' +
      '<path d="M15.4 15.4 L21 21 M10.5 7.6 V13.4 M7.6 10.5 H13.4"></path></svg>';
    w.appendChild(badge);
    w.tabIndex = 0;
    w.setAttribute('role', 'button');
    w.setAttribute('aria-label', 'Enlarge: ' + (img.alt || 'figure'));
    w.title = 'Click to enlarge';
    var open = function (rel) { lightbox().open(img.currentSrc || img.src, img.alt, w, rel); };
    w.addEventListener('click', function (e) {
      var r = img.getBoundingClientRect();
      open(r.width && r.height ? [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] : null);
    });
    w.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(null); } });
  };
  Array.prototype.forEach.call(document.querySelectorAll('img.figure, .traj img'), zoomable);

  /* open the first entries; the comparison and trajectory viewers load when they scroll into view, so the page opens quickly */
  if (M.gallery && M.gallery.length) openGallery(0);
  if (M.videodepth && M.videodepth.length) openVd(0);
  var whenVisible = function (target, start) {
    var started = false, go = function () { if (!started) { started = true; start(); } };
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) go(); }); },
        { rootMargin: '400px' }).observe(target);
    } else go();
  };
  if (M.compare && M.compare.length) whenVisible(document.getElementById('compare'), function () { openGroup(0); });
  if (openTrajGroup) whenVisible(document.getElementById('trajectories'), function () { openTrajGroup(0); });
})();

// public page: copy the BibTeX (button at the top right of the block, as on the Any3R page)
(function () {
  function init() {
    var btn = document.getElementById("copy-bibtex"), pre = document.getElementById("bibtex-text");
    if (!btn || !pre) return;
    function done(ok) {
      btn.textContent = ok ? "Copied" : "Press Ctrl+C";
      setTimeout(function () { btn.textContent = "Copy"; }, 1600);
    }
    function fallback() {                       // no clipboard API: select the text and copy it, or leave it selected
      var r = document.createRange(), sel = window.getSelection(), ok = false;
      r.selectNodeContents(pre); sel.removeAllRanges(); sel.addRange(r);
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      if (ok) sel.removeAllRanges();
      done(ok);
    }
    btn.addEventListener("click", function () {
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(pre.textContent).then(function () { done(true); }, fallback);
      } else {
        fallback();
      }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
