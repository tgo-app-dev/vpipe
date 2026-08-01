// Mask Editor panel -- the GUI view the "create-mask" stage ships with
// itself.
//
// Embedded in libvpipe beside the stage's C++ and served by whatever
// front end is attached; it imports nothing from that front end. The
// widget helpers arrive through mount()'s ctx, and the channel to
// create-mask-view-backend.cc through ctx.openChannel().
//
// THE MASK IS A BYTE PER SAMPLE, and this paints into that array
// directly rather than into a canvas. A canvas would force every stroke
// through RGBA compositing rules, and none of the three mask kinds
// actually composites that way: a two-state mask has no partial
// coverage to blend, a class index must be ASSIGNED and never averaged
// with its neighbour, and only alpha wants a soft edge at all. Owning
// the buffer makes each of those one line in the dab loop, and makes
// the commit exact -- what leaves here is the array, not a rendering of
// it.
//
// TWO COORDINATE SPACES, because the mask canvas need not match the
// reference image. The view box is the REFERENCE image's size when
// there is one (so the picture is never distorted) and the mask is
// stretched over it, which is what makes a configured width x height
// register with the image it is a mask for. Screen -> box -> mask, and
// the brush radius is in BOX pixels so the round tip stays round where
// the user sees it -- an ellipse in mask space is the correct footprint
// of a circle drawn on a stretched canvas.
//
// LEFT BUTTON PAINTS, RIGHT BUTTON PANS. The pan-zoom controller takes
// panButton: 2 for exactly that, which leaves the wheel for zoom and
// the whole left button for the brush.
//
// COMMIT IS THE ONLY THING THAT SPEAKS TO THE STAGE. Strokes are local
// until the button is released; then the buffer goes up as a PNG and
// the stage emits one beat. Nothing is streamed while painting -- the
// user, not the pipeline, decides when a mask is finished.

import { makePanZoom } from '../../sdk/pan-zoom.js';

export { strings } from './create-mask-strings.js';

const MIN_RADIUS = 1;
const MAX_RADIUS = 512;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// "#rrggbb" -> [r, g, b]. Anything unparseable reads as mid grey, so a
// malformed palette entry is visible rather than invisible.
function parseColor(s) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(s || ''));
  if (!m) { return [128, 128, 128]; }
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

// Chunked so a large mask doesn't blow the argument limit of apply().
function toBase64(bytes) {
  let s = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

export function mount(body, actions, ctx) {
  const { el, clear, makeIcon, t } = ctx.host;
  const cfg = ctx.config || {};
  const onTitle = ctx.onTitle || (() => {});
  const saveConfig = ctx.onConfigChange || (() => {});

  let destroyed = false;
  let screen = null;                  // 'picker' | 'waiting' | 'editing'
  let editor = null;                  // the canvas editor, when editing
  let designation = cfg.designation || null;

  // ---- channel ------------------------------------------------------
  const channel = ctx.openChannel({
    onOpen() {
      if (destroyed) { return; }
      if (designation) {
        channel.send({ m: 'watch', pipeline: designation.pipeline,
                       stage: designation.stage });
      } else {
        channel.send({ m: 'list' });
      }
    },
    onMessage(msg) { if (!destroyed) { handleMessage(msg); } },
    onBinary(header, payload) {
      if (!destroyed && editor) { editor.onBinary(header, payload); }
    },
    onClose() {
      if (!destroyed && editor) {
        editor.setStatus(t('mask.reconnecting'));
      }
    },
  });

  function handleMessage(msg) {
    switch (msg && msg.m) {
      case 'stages':
        if (screen === 'picker') { renderStageList(msg.list || []); }
        break;
      case 'waiting': showWaiting(msg.pipeline, msg.stage); break;
      case 'playing': showEditor(msg.title || msg.stage); break;
      case 'gone':    setDesignation(null); break;
      case 'frame':   if (editor) { editor.onFrame(msg); } break;
      case 'committed':
        if (editor) { editor.onCommitted(msg.seq); }
        break;
      default: break;
    }
  }

  function setDesignation(d) {
    designation = d;
    cfg.designation = d;
    saveConfig();
    if (d) {
      channel.send({ m: 'watch', pipeline: d.pipeline, stage: d.stage });
    } else {
      channel.send({ m: 'unwatch' });
      showPicker();
    }
  }

  function teardownEditor() {
    if (editor) {
      try { editor.destroy(); } catch (e) {}
      editor = null;
    }
  }

  function changeBtn() {
    return el('button', { class: 'btn ghost',
      onclick: () => setDesignation(null) }, t('mask.change'));
  }

  // ---- picker -------------------------------------------------------
  let listEl = null;

  function showPicker() {
    teardownEditor();
    screen = 'picker';
    onTitle(t('mask.panel'));
    clear(body);
    if (actions) { clear(actions); }
    listEl = el('div', { class: 'cmk-list' },
      el('div', { class: 'cmk-hint' }, t('mask.loading')));
    body.append(el('div', { class: 'cmk-picker' },
      el('div', { class: 'cmk-picker-title' }, t('mask.select')),
      listEl));
    if (actions) {
      actions.append(el('button', { class: 'btn ghost',
        onclick: () => channel.send({ m: 'list' }) }, t('mask.refresh')));
    }
    channel.send({ m: 'list' });
  }

  function renderStageList(stages) {
    if (!listEl) { return; }
    clear(listEl);
    if (stages.length === 0) {
      listEl.append(el('div', { class: 'cmk-hint' }, t('mask.no_stages')));
      return;
    }
    for (const st of stages) {
      const sub = [st.pipeline + ' / ' + st.stage,
        st.live ? t('mask.state_live') : (st.state || '')]
        .filter(Boolean).join('  ·  ');
      listEl.append(el('button',
        { class: 'cmk-item' + (st.live ? ' live' : ''),
          onclick: () => setDesignation(
            { pipeline: st.pipeline, stage: st.stage }) },
        el('span', { class: 'cmk-item-name' }, st.title || st.stage),
        el('span', { class: 'cmk-item-sub' }, sub)));
    }
  }

  // ---- waiting ------------------------------------------------------
  function showWaiting(pipeline, stage) {
    teardownEditor();
    listEl = null;
    screen = 'waiting';
    clear(body);
    if (actions) { clear(actions); }
    onTitle(t('mask.waiting_title', { stage }));
    body.append(el('div', { class: 'cmk-picker' },
      el('div', { class: 'cmk-waiting' }, makeIcon('hourglass', 'sm')),
      el('div', { class: 'cmk-picker-title' },
        t('mask.waiting_title', { stage })),
      el('div', { class: 'cmk-hint' },
        t('mask.waiting', { pipeline, stage }))));
    if (actions) { actions.append(changeBtn()); }
  }

  // ---- the editor ---------------------------------------------------
  function showEditor(title) {
    teardownEditor();
    listEl = null;
    screen = 'editing';
    onTitle(t('mask.editing', { stage: title }));
    clear(body);
    if (actions) { clear(actions); }

    const canvas = el('canvas', { class: 'cmk-canvas' });
    const status = el('div', { class: 'cmk-status' });
    const viewport = el('div', { class: 'cmk-viewport no-view' }, canvas);

    // The box the content occupies: the reference image's size when
    // there is one, so the picture keeps its own aspect ratio and the
    // mask is what stretches.
    let boxW = 0, boxH = 0;
    // The mask canvas -- the resolution the stage actually authors at.
    let mw = 0, mh = 0;
    let maskBuf = null;                // Uint8Array, one byte per sample
    let maskCv = null, maskCtx = null, maskImg = null;
    let bgBmp = null;
    let seedPng = null;                // last mask PNG the stage sent
    let version = 0;
    let settings = { mode: 'binary', classes: 2, colors: ['#ff3b30'],
                     overlay_opacity: 0.5, interactive: true };
    let palette = [[255, 59, 48]];

    // Brush + tool state, persisted so a saved layout keeps the feel.
    let radius   = clamp(cfg.radius || 24, MIN_RADIUS, MAX_RADIUS);
    let hardness = typeof cfg.hardness === 'number' ? cfg.hardness : 0.5;
    // typeof, not `|| 1`: class 0 is the background brush and a real
    // choice, and truthiness would silently promote it to class 1.
    let curClass = typeof cfg.curClass === 'number' ? cfg.curClass : 1;
    let erasing  = !!cfg.erasing;
    let showBg   = cfg.showBg === undefined ? true : !!cfg.showBg;
    let collapsed = !!cfg.collapsed;
    // Where the pointer is, in box coords, for the brush ring.
    let hover = null;

    const pz = makePanZoom({
      panButton: 2,                    // left stays free to paint
      onApply: () => draw(),
    });

    const setStatus = (m) => { status.textContent = m; };
    setStatus(t('mask.connecting'));

    // ---- controls ----------------------------------------------------
    const ctlBtn = (label, tip, fn) =>
      el('button', { class: 'cmk-ctl', title: tip,
        onclick: (e) => { e.stopPropagation(); fn(); } }, label);

    const radiusOut = el('span', { class: 'cmk-read' }, String(radius));
    const hardOut = el('span', { class: 'cmk-read' },
                       Math.round(hardness * 100) + '%');
    const bgBtn = ctlBtn('◉', '', () => setShowBg());
    const eraseBtn = ctlBtn('⌫', '', () => setErasing());
    const softBtn = ctlBtn('◌', t('mask.softer'),
                           () => setHardness(hardness - 0.1));
    const hardBtn = ctlBtn('●', t('mask.harder'),
                           () => setHardness(hardness + 0.1));
    const classRow = el('div', { class: 'cmk-classes' });
    const classBtns = [];
    const commitBtn = el('button',
      { class: 'cmk-commit', title: t('mask.commit_tip'),
        // On RELEASE, as the stage's contract says: one button-up, one
        // beat. Using pointerup rather than click also means a drag that
        // began on the canvas and ended here cannot commit by accident.
        onpointerup: (e) => { e.stopPropagation(); commit(); } },
      t('mask.commit'));

    const ctlGroup = el('div', { class: 'cmk-ctl-group' },
      ctlBtn('−', t('mask.zoom_out'), () => pz.zoomBy(1 / 1.2)),
      ctlBtn('+', t('mask.zoom_in'), () => pz.zoomBy(1.2)),
      ctlBtn('1:1', t('mask.actual_size'), () => pz.actual()),
      ctlBtn('Fit', t('mask.fit'), () => pz.fit()),
      ctlBtn('⊙', t('mask.center'), () => pz.center()),
      bgBtn,
      el('span', { class: 'cmk-sep' }),
      ctlBtn('[', t('mask.brush_smaller'),
             () => setRadius(Math.round(radius / 1.3))),
      radiusOut,
      ctlBtn(']', t('mask.brush_bigger'),
             () => setRadius(Math.round(radius * 1.3) + 1)),
      softBtn, hardOut, hardBtn,
      eraseBtn,
      classRow,
      el('span', { class: 'cmk-sep' }),
      ctlBtn('⟲', t('mask.reset'), () => resetMask()));
    const collapseBtn = ctlBtn('«', '', () => setCollapsed());
    collapseBtn.classList.add('cmk-toggle');
    // Commit sits OUTSIDE the collapsible group: hiding the controls is
    // for looking at the mask unobstructed, and the one button you must
    // never have to go looking for is the one that emits.
    const controls = el('div', { class: 'cmk-controls' },
      collapseBtn, ctlGroup, commitBtn);
    viewport.append(controls);
    body.append(el('div', { class: 'cmk-view' }, viewport, status));
    if (actions) { actions.append(changeBtn()); }

    pz.attach(viewport);
    // Right-drag is the pan gesture, so the menu it would normally open
    // has to go.
    viewport.addEventListener('contextmenu', (e) => e.preventDefault());

    // ---- mask buffer -------------------------------------------------
    // True while the buffer is still exactly what the stage last sent.
    // Declared here because allocMask resets it.
    let pristine = true;

    function allocMask(w, h) {
      mw = w; mh = h;
      maskBuf = new Uint8Array(w * h);
      maskCv = document.createElement('canvas');
      maskCv.width = w; maskCv.height = h;
      maskCtx = maskCv.getContext('2d');
      maskImg = maskCtx.createImageData(w, h);
      // A fresh canvas has no local work on it, so the mask that comes
      // with this frame is free to seed it.
      pristine = true;
      repaintMask(0, 0, w - 1, h - 1);
    }

    // Rebuild the RGBA the overlay is drawn from, over one dirty box.
    // The per-sample alpha is the mask's own coverage; the stage's
    // overlay_opacity is applied once, globally, at draw time -- which
    // is exactly how the C++ side composes it.
    function repaintMask(x0, y0, x1, y1) {
      if (!maskBuf) { return; }
      const d = maskImg.data;
      const cls = settings.mode === 'class';
      const alpha = settings.mode === 'alpha';
      for (let y = y0; y <= y1; ++y) {
        for (let x = x0; x <= x1; ++x) {
          const i = y * mw + x;
          const v = maskBuf[i];
          const o = i * 4;
          if (v === 0) { d[o + 3] = 0; continue; }
          const c = cls ? (palette[v] || palette[0] || [128, 128, 128])
                        : palette[0];
          d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
          d[o + 3] = alpha ? v : 255;
        }
      }
      maskCtx.putImageData(maskImg, 0, 0,
                           x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    // ---- geometry ----------------------------------------------------
    // Screen point (client coords) -> box coords, or null when there is
    // no content to hit.
    function toBox(clientX, clientY) {
      const s = pz.state();
      if (!s.enabled || !boxW || !boxH) { return null; }
      const r = viewport.getBoundingClientRect();
      const ox = (s.vw - s.dw) / 2 + s.tx;
      const oy = (s.vh - s.dh) / 2 + s.ty;
      if (!(s.scale > 0)) { return null; }
      return { x: (clientX - r.left - ox) / s.scale,
               y: (clientY - r.top - oy) / s.scale };
    }

    // ---- painting ----------------------------------------------------
    let dirty = null;

    function markDirty(x0, y0, x1, y1) {
      if (!dirty) { dirty = { x0, y0, x1, y1 }; return; }
      dirty.x0 = Math.min(dirty.x0, x0);
      dirty.y0 = Math.min(dirty.y0, y0);
      dirty.x1 = Math.max(dirty.x1, x1);
      dirty.y1 = Math.max(dirty.y1, y1);
    }

    // One brush dab at box coords (bx, by). `radius` is in BOX pixels;
    // in mask space that becomes an ellipse whenever the canvas is
    // stretched, which is the right footprint for a round tip.
    function dab(bx, by) {
      if (!maskBuf || !mw || !mh) { return; }
      const cx = bx * mw / boxW;
      const cy = by * mh / boxH;
      const rx = Math.max(0.5, radius * mw / boxW);
      const ry = Math.max(0.5, radius * mh / boxH);
      const x0 = clamp(Math.floor(cx - rx), 0, mw - 1);
      const x1 = clamp(Math.ceil(cx + rx), 0, mw - 1);
      const y0 = clamp(Math.floor(cy - ry), 0, mh - 1);
      const y1 = clamp(Math.ceil(cy + ry), 0, mh - 1);
      if (x1 < x0 || y1 < y0) { return; }

      const soft = settings.mode === 'alpha' && hardness < 1;
      // Full value out to hardness*r, then a linear ramp to nothing at
      // the rim -- so hardness 1 is a hard disc and 0 is a full falloff.
      const inner = soft ? clamp(hardness, 0, 0.999) : 1;
      const paint = settings.mode === 'class'
        ? (erasing ? 0 : curClass)
        : (erasing ? 0 : 255);

      for (let y = y0; y <= y1; ++y) {
        const dy = (y + 0.5 - cy) / ry;
        for (let x = x0; x <= x1; ++x) {
          const dx = (x + 0.5 - cx) / rx;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d > 1) { continue; }
          const i = y * mw + x;
          if (!soft) {
            maskBuf[i] = paint;
            continue;
          }
          const a = d <= inner ? 1 : 1 - (d - inner) / (1 - inner);
          const v = Math.round(clamp(a, 0, 1) * 255);
          // Within a stroke as well as across strokes, coverage
          // ACCUMULATES by max: overlapping dabs must not build a
          // brighter core than the brush can produce.
          maskBuf[i] = erasing
            ? Math.min(maskBuf[i], 255 - v)
            : Math.max(maskBuf[i], v);
        }
      }
      markDirty(x0, y0, x1, y1);
    }

    // Dabs along a segment, so a fast stroke is a stroke and not a row
    // of dots. A quarter of the radius per step keeps the overlap dense
    // enough that the rim of one dab never shows inside the next.
    function stroke(from, to) {
      const dx = to.x - from.x, dy = to.y - from.y;
      const dist = Math.hypot(dx, dy);
      const step = Math.max(0.5, radius * 0.25);
      const n = Math.max(1, Math.ceil(dist / step));
      for (let i = 1; i <= n; ++i) {
        dab(from.x + dx * i / n, from.y + dy * i / n);
      }
    }

    let painting = false;
    let last = null;

    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !settings.interactive || !maskBuf) { return; }
      const p = toBox(e.clientX, e.clientY);
      if (!p) { return; }
      e.preventDefault();
      painting = true;
      // The buffer now holds work the stage has never seen, so an
      // incoming frame must stop re-seeding over the top of it.
      pristine = false;
      last = p;
      dab(p.x, p.y);
      flush();
      try { canvas.setPointerCapture(e.pointerId); } catch (x) {}
    });
    canvas.addEventListener('pointermove', (e) => {
      const p = toBox(e.clientX, e.clientY);
      hover = p;
      if (!painting) { draw(); return; }
      if (!p) { return; }
      stroke(last, p);
      last = p;
      flush();
    });
    const endStroke = (e) => {
      if (!painting) { return; }
      painting = false;
      last = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch (x) {}
    };
    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);
    canvas.addEventListener('pointerleave', () => {
      hover = null;
      draw();
    });

    function flush() {
      if (dirty) {
        repaintMask(dirty.x0, dirty.y0, dirty.x1, dirty.y1);
        dirty = null;
      }
      draw();
    }

    // ---- incoming ----------------------------------------------------
    function applySettings(e) {
      if (!e) { return; }
      settings = {
        mode: e.mode || 'binary',
        classes: Math.max(2, e.classes | 0),
        colors: Array.isArray(e.colors) && e.colors.length
          ? e.colors : ['#ff3b30'],
        overlay_opacity: typeof e.overlay_opacity === 'number'
          ? e.overlay_opacity : 0.5,
        interactive: e.interactive !== false,
      };
      palette = settings.colors.map(parseColor);
      curClass = clamp(curClass, 1, Math.max(1, settings.classes - 1));
      syncTools();
    }

    function releaseBg() {
      if (bgBmp && typeof bgBmp.close === 'function') {
        try { bgBmp.close(); } catch (e) {}
      }
      bgBmp = null;
    }

    function onFrame(msg) {
      version = msg.version || 0;
      applySettings(msg.editor);
      const w = msg.width | 0, h = msg.height | 0;
      const bw = msg.bg_width | 0, bh = msg.bg_height | 0;
      // A frame that carries no background means the stage has none, so
      // drop the one we were showing rather than leave a stale picture
      // under the mask -- and hand the bitmap back while doing it.
      if (!msg.has_bg) { releaseBg(); }

      // The view box follows the reference image when there is one.
      const nbw = bw > 0 ? bw : w;
      const nbh = bh > 0 ? bh : h;
      if (nbw !== boxW || nbh !== boxH) {
        boxW = nbw; boxH = nbh;
        pz.setIntrinsic(boxW || 1, boxH || 1);
      }
      // Re-seed the buffer ONLY when the canvas resolution changes (or
      // on the first frame). A stage republishes on every new reference
      // image, and adopting its mask each time would wipe out strokes
      // the user has not committed yet.
      if (w > 0 && h > 0 && (w !== mw || h !== mh)) { allocMask(w, h); }
      pz.setEnabled(!!(boxW && boxH));
      setStatus(boxW && boxH
        ? (settings.interactive ? t('mask.hint') : t('mask.readonly'))
        : t('mask.no_canvas'));
      syncTools();
      draw();
    }

    function onBinary(header, payload) {
      if (header.m !== 'image') { return; }
      const at = version;
      const slot = header.slot === 'mask' ? 'mask' : 'bg';
      if (slot === 'mask') { seedPng = payload; }
      decode(payload).then((bmp) => {
        if (destroyed || !bmp || at !== version) {
          if (bmp && typeof bmp.close === 'function') {
            try { bmp.close(); } catch (e) {}
          }
          return;
        }
        if (slot === 'bg') {
          releaseBg();
          bgBmp = bmp;
          draw();
          return;
        }
        // A mask arrived. Adopt it only if the buffer is still the
        // pristine one allocated for this size -- see onFrame.
        if (adoptSeed(bmp)) { flush(); }
        if (typeof bmp.close === 'function') {
          try { bmp.close(); } catch (e) {}
        }
      }).catch(() => {});
    }

    // Read a mask bitmap back into the byte buffer. The stage encodes
    // GRAY8, so every channel carries the sample; R is as good as any.
    function adoptSeed(bmp) {
      if (!maskBuf || !pristine || !mw || !mh) { return false; }
      const c = document.createElement('canvas');
      c.width = mw; c.height = mh;
      const g = c.getContext('2d');
      if (!g) { return false; }
      g.drawImage(bmp, 0, 0, mw, mh);
      let px = null;
      try { px = g.getImageData(0, 0, mw, mh).data; } catch (e) { return false; }
      for (let i = 0; i < mw * mh; ++i) { maskBuf[i] = px[i * 4]; }
      markDirty(0, 0, mw - 1, mh - 1);
      return true;
    }

    function decode(buf) {
      const blob = new Blob([buf], { type: 'image/png' });
      if (typeof createImageBitmap === 'function') {
        return createImageBitmap(blob);
      }
      return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(blob);
        const im = new Image();
        im.onload = () => { URL.revokeObjectURL(url); resolve(im); };
        im.onerror = () => { URL.revokeObjectURL(url); reject(); };
        im.src = url;
      });
    }

    // ---- commit ------------------------------------------------------
    // The buffer goes up as a grey PNG: R = G = B = the sample, so the
    // stage reads the value back whichever channel the browser's
    // encoder decided to keep.
    async function commit() {
      if (!settings.interactive || !maskBuf || !mw || !mh) { return; }
      const c = document.createElement('canvas');
      c.width = mw; c.height = mh;
      const g = c.getContext('2d');
      if (!g) { return; }
      const img = g.createImageData(mw, mh);
      const d = img.data;
      for (let i = 0; i < mw * mh; ++i) {
        const v = maskBuf[i];
        d[i * 4] = v; d[i * 4 + 1] = v; d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
      if (!blob || destroyed) { return; }
      const bytes = new Uint8Array(await blob.arrayBuffer());
      channel.send({ m: 'commit', png: toBase64(bytes) });
    }

    function onCommitted() {
      setStatus(t('mask.committed'));
      // Back to the usual hint, so the panel doesn't sit claiming a
      // commit that has scrolled out of relevance.
      window.setTimeout(() => {
        if (!destroyed && editor && settings.interactive) {
          setStatus(t('mask.hint'));
        }
      }, 1200);
    }

    // ---- tools -------------------------------------------------------
    function setRadius(r) {
      radius = clamp(r | 0, MIN_RADIUS, MAX_RADIUS);
      cfg.radius = radius;
      saveConfig();
      syncTools();
      draw();
    }
    function setHardness(h) {
      hardness = clamp(Math.round(h * 10) / 10, 0, 1);
      cfg.hardness = hardness;
      saveConfig();
      syncTools();
    }
    function setErasing(v) {
      erasing = (v === undefined) ? !erasing : !!v;
      cfg.erasing = erasing;
      saveConfig();
      syncTools();
    }
    function setShowBg(v) {
      showBg = (v === undefined) ? !showBg : !!v;
      cfg.showBg = showBg;
      saveConfig();
      syncTools();
      draw();
    }
    function setClass(n) {
      curClass = clamp(n | 0, 0, Math.max(1, settings.classes - 1));
      cfg.curClass = curClass;
      saveConfig();
      syncTools();
    }
    function setCollapsed(v) {
      collapsed = (v === undefined) ? !collapsed : !!v;
      cfg.collapsed = collapsed;
      saveConfig();
      syncTools();
    }

    // Reset means "discard my edits": back to the mask the stage last
    // published (the one its in-mask input seeded, if any), and empty
    // when it never published one.
    function resetMask() {
      if (!maskBuf) { return; }
      maskBuf.fill(0);
      pristine = true;
      markDirty(0, 0, mw - 1, mh - 1);
      if (seedPng) {
        decode(seedPng).then((bmp) => {
          if (destroyed || !bmp) { flush(); return; }
          adoptSeed(bmp);
          if (typeof bmp.close === 'function') {
            try { bmp.close(); } catch (e) {}
          }
          flush();
        }).catch(() => flush());
      } else {
        flush();
      }
    }

    function syncTools() {
      radiusOut.textContent = String(radius);
      radiusOut.title = t('mask.brush_radius', { r: radius });
      hardOut.textContent = Math.round(hardness * 100) + '%';
      hardOut.title = t('mask.hardness',
                        { h: Math.round(hardness * 100) });
      // Hardness only shapes an ALPHA brush -- a two-state mask and a
      // class map have no partial value for it to produce. Dim it
      // rather than drop it, so the row keeps its shape.
      const soft = settings.mode === 'alpha';
      softBtn.disabled = !soft;
      hardBtn.disabled = !soft;
      hardOut.classList.toggle('off', !soft);
      eraseBtn.classList.toggle('on', erasing);
      eraseBtn.title = erasing ? t('mask.erase_on') : t('mask.erase_off');
      bgBtn.classList.toggle('on', showBg);
      bgBtn.title = showBg ? t('mask.bg_hide') : t('mask.bg_show');
      collapseBtn.textContent = collapsed ? '»' : '«';
      collapseBtn.title = collapsed
        ? t('mask.controls_show') : t('mask.controls_hide');
      controls.classList.toggle('collapsed', collapsed);
      commitBtn.disabled = !settings.interactive;

      // The class swatches: one per class, coloured with what it paints.
      // Only in class mode -- there is nothing to choose between
      // otherwise.
      const want = settings.mode === 'class' ? settings.classes : 0;
      if (classBtns.length !== want) {
        clear(classRow);
        classBtns.length = 0;
        for (let i = 0; i < want; ++i) {
          const b = el('button', { class: 'cmk-swatch',
            title: i === 0 ? t('mask.class_bg') : t('mask.class', { n: i }),
            onclick: (e) => { e.stopPropagation(); setClass(i); } },
            String(i));
          const c = palette[i] || [128, 128, 128];
          b.style.background = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
          classBtns.push(b);
          classRow.append(b);
        }
      }
      for (let i = 0; i < classBtns.length; ++i) {
        classBtns[i].classList.toggle('on', i === curClass);
      }
    }

    // ---- rendering ---------------------------------------------------
    function draw() {
      const vw = viewport.clientWidth, vh = viewport.clientHeight;
      if (vw <= 0 || vh <= 0) { return; }
      const dpr = window.devicePixelRatio || 1;
      const bw = Math.round(vw * dpr), bh = Math.round(vh * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw; canvas.height = bh;
      }
      const g = canvas.getContext('2d');
      if (!g) { return; }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = '#000';
      g.fillRect(0, 0, vw, vh);
      if (!boxW || !boxH) { return; }

      const s = pz.state();
      const x = (vw - s.dw) / 2 + s.tx;
      const y = (vh - s.dh) / 2 + s.ty;
      // Show the pixels rather than a blur when magnifying: a mask is
      // judged at its edge, and that edge is where the blur would be.
      g.imageSmoothingEnabled = s.scale < 1.5;

      if (bgBmp && showBg) {
        g.drawImage(bgBmp, x, y, s.dw, s.dh);
      }
      if (maskCv && mw && mh) {
        // The mask is STRETCHED across the same box the reference image
        // occupies, which is what registers a configured width x height
        // with the picture it masks.
        //
        // At the stage's overlay_opacity while the reference image is
        // showing, so the panel looks like the overlay the stage would
        // emit -- but FULLY opaque once it is hidden, because then the
        // mask is the subject rather than something laid over a picture.
        g.save();
        g.globalAlpha = (bgBmp && showBg)
          ? clamp(settings.overlay_opacity, 0, 1) : 1;
        g.drawImage(maskCv, x, y, s.dw, s.dh);
        g.restore();
      }
      drawBrushRing(g, x, y, s.scale);
    }

    // The brush footprint, so the radius control means something before
    // a stroke is laid down.
    function drawBrushRing(g, x, y, scale) {
      if (!hover || !settings.interactive || !(scale > 0)) { return; }
      g.save();
      g.beginPath();
      g.arc(x + hover.x * scale, y + hover.y * scale,
            Math.max(1, radius * scale), 0, Math.PI * 2);
      g.strokeStyle = erasing ? 'rgba(255,255,255,.85)'
                              : 'rgba(255,255,255,.6)';
      g.lineWidth = 1;
      g.stroke();
      g.restore();
    }

    syncTools();
    draw();

    editor = {
      onFrame, onBinary, onCommitted, setStatus,
      destroy() {
        releaseBg();
        maskBuf = null;
        maskCv = null;
      },
    };
  }

  // ---- entry --------------------------------------------------------
  if (designation) {
    showWaiting(designation.pipeline, designation.stage);
  } else {
    showPicker();
  }

  return () => {
    destroyed = true;
    teardownEditor();
    channel.close();
  };
}
