// Compare Images panel -- the GUI view the "compare-image" stage ships
// with itself.
//
// Embedded in libvpipe beside the stage's C++ and served by whatever
// front end is attached; it imports nothing from that front end. The
// widget helpers arrive through mount()'s ctx, and the channel to
// compare-image-view-backend.cc through ctx.openChannel().
//
// SIX COMPARISON MODES, on top of the five view controls (zoom out / in
// / 1:1 / fit / centre):
//   A          image A alone
//   B          image B alone
//   A|B        side by side, A left, B right, centre divider
//   A/B        stacked, A top, B bottom, centre divider
//   A◧B        one frame, draggable VERTICAL split: A left, B right
//   A⬒B        one frame, draggable HORIZONTAL split: A top, B bottom
//
// Plus three controls over the presentation itself:
//   ⇄          SWAP: exchange which input plays the A role and which
//              plays B. Purely presentational -- it re-reads the same
//              latched pair, so it costs no round trip and cannot get
//              out of step with an incoming update.
//   │          SEAM: hide the wipe's line, shadow and grab dot, so
//              nothing is painted over the detail being judged -- which
//              is precisely the detail AT the seam. The handle keeps its
//              hit band and its resize cursor, so the seam is still
//              there to grab; the pointer shape is what says so.
//              Disabled outside the wipe modes, which have no seam.
//   «/»        COLLAPSE: hide every other button for an unobstructed
//              look at the images. The toggle itself always stays, and
//              sits FIRST so it does not move when the row collapses.
//
// Everything is drawn into ONE canvas under ONE transform, which is what
// makes "zoom and position always sync between A and B" true by
// construction rather than by keeping two viewers in step. The backend
// publishes both images already padded to a common size, so a pixel at
// (x, y) means the same place in each -- the wipe seam is therefore
// exact, not approximate.
//
// A missing image is simply not drawn: its half (or its side of the
// seam) stays black, and an empty pair is a black frame.

import { makePanZoom } from '../../sdk/pan-zoom.js';

export { strings } from './compare-image-strings.js';

// [id, label key, tooltip key]. Order is the button order.
const MODES = [
  ['a',     'compare.mode_a',      'compare.mode_a_tip'],
  ['b',     'compare.mode_b',      'compare.mode_b_tip'],
  ['lr',    'compare.mode_lr',     'compare.mode_lr_tip'],
  ['tb',    'compare.mode_tb',     'compare.mode_tb_tip'],
  ['wipev', 'compare.mode_wipe_v', 'compare.mode_wipe_v_tip'],
  ['wipeh', 'compare.mode_wipe_h', 'compare.mode_wipe_h_tip'],
];
const isWipe = (m) => m === 'wipev' || m === 'wipeh';
const isSplitPane = (m) => m === 'lr' || m === 'tb';

export function mount(body, actions, ctx) {
  const { el, clear, makeIcon, t } = ctx.host;
  const cfg = ctx.config || {};
  const onTitle = ctx.onTitle || (() => {});
  const saveConfig = ctx.onConfigChange || (() => {});

  let destroyed = false;
  let screen = null;                  // 'picker' | 'waiting' | 'showing'
  let viewer = null;                  // the canvas comparison, when showing
  let designation = cfg.designation || null;
  // Persisted so a saved layout comes back in the same comparison.
  let mode = MODES.some((m) => m[0] === cfg.mode) ? cfg.mode : 'wipev';
  let wipe = typeof cfg.wipe === 'number' ? cfg.wipe : 0.5;
  // A and B exchanged, the control row collapsed, and the wipe seam's
  // line hidden. All view state, so all ride in the panel config.
  let swapped    = !!cfg.swapped;
  let collapsed  = !!cfg.collapsed;
  let seamHidden = !!cfg.seamHidden;

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
      if (!destroyed && viewer) { viewer.onBinary(header, payload); }
    },
    onClose() {
      if (!destroyed && viewer) {
        viewer.setStatus(t('compare.reconnecting'));
      }
    },
  });

  function handleMessage(msg) {
    switch (msg && msg.m) {
      case 'stages':
        if (screen === 'picker') { renderStageList(msg.list || []); }
        break;
      case 'waiting': showWaiting(msg.pipeline, msg.stage); break;
      case 'playing': showViewer(msg.title || msg.stage); break;
      case 'gone':    setDesignation(null); break;
      case 'pair':    if (viewer) { viewer.onPair(msg); } break;
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

  function teardownViewer() {
    if (viewer) {
      try { viewer.destroy(); } catch (e) {}
      viewer = null;
    }
  }

  function changeBtn() {
    return el('button', { class: 'btn ghost',
      onclick: () => setDesignation(null) }, t('compare.change'));
  }

  // ---- picker -------------------------------------------------------
  let listEl = null;

  function showPicker() {
    teardownViewer();
    screen = 'picker';
    onTitle(t('compare.panel'));
    clear(body);
    if (actions) { clear(actions); }
    listEl = el('div', { class: 'cmpi-list' },
      el('div', { class: 'cmpi-hint' }, t('compare.loading')));
    body.append(el('div', { class: 'cmpi-picker' },
      el('div', { class: 'cmpi-picker-title' }, t('compare.select')),
      listEl));
    if (actions) {
      actions.append(el('button', { class: 'btn ghost',
        onclick: () => channel.send({ m: 'list' }) },
        t('compare.refresh')));
    }
    channel.send({ m: 'list' });
  }

  function renderStageList(stages) {
    if (!listEl) { return; }
    clear(listEl);
    if (stages.length === 0) {
      listEl.append(el('div', { class: 'cmpi-hint' },
        t('compare.no_stages')));
      return;
    }
    for (const st of stages) {
      const sub = [st.pipeline + ' / ' + st.stage,
        st.live ? t('compare.state_live') : (st.state || '')]
        .filter(Boolean).join('  ·  ');
      listEl.append(el('button',
        { class: 'cmpi-item' + (st.live ? ' live' : ''),
          onclick: () => setDesignation(
            { pipeline: st.pipeline, stage: st.stage }) },
        el('span', { class: 'cmpi-item-name' }, st.title || st.stage),
        el('span', { class: 'cmpi-item-sub' }, sub)));
    }
  }

  // ---- waiting ------------------------------------------------------
  function showWaiting(pipeline, stage) {
    teardownViewer();
    listEl = null;
    screen = 'waiting';
    clear(body);
    if (actions) { clear(actions); }
    onTitle(t('compare.waiting_title', { stage }));
    body.append(el('div', { class: 'cmpi-picker' },
      el('div', { class: 'cmpi-waiting' }, makeIcon('hourglass', 'sm')),
      el('div', { class: 'cmpi-picker-title' },
        t('compare.waiting_title', { stage })),
      el('div', { class: 'cmpi-hint' },
        t('compare.waiting', { pipeline, stage }))));
    if (actions) { actions.append(changeBtn()); }
  }

  // ---- the comparison -----------------------------------------------
  function showViewer(title) {
    teardownViewer();
    listEl = null;
    screen = 'showing';
    onTitle(t('compare.showing', { stage: title }));
    clear(body);
    if (actions) { clear(actions); }

    const canvas = el('canvas', { class: 'cmpi-canvas' });
    const status = el('div', { class: 'cmpi-status' });
    const handle = el('div', { class: 'cmpi-handle' });
    const viewport = el('div', { class: 'cmpi-viewport no-view' },
      canvas, handle);

    // Zoom/pan is shared by both images because there is only one of it.
    // In the side-by-side modes each image is fitted to its HALF, so
    // "Fit" means what the user sees, not the full element.
    const pz = makePanZoom({
      paneSize: (vw, vh) => {
        if (mode === 'lr') { return { w: vw / 2, h: vh }; }
        if (mode === 'tb') { return { w: vw, h: vh / 2 }; }
        return { w: vw, h: vh };
      },
      onApply: () => draw(),
    });

    const ctlBtn = (label, tip, fn) =>
      el('button', { class: 'cmpi-ctl', title: tip,
        onclick: (e) => { e.stopPropagation(); fn(); } }, label);
    const modeBtns = new Map();
    const modeRow = el('div', { class: 'cmpi-modes' });
    for (const [id, labelKey, tipKey] of MODES) {
      const b = ctlBtn(t(labelKey), t(tipKey), () => setMode(id));
      modeBtns.set(id, b);
      modeRow.append(b);
    }
    const swapBtn = ctlBtn('⇄', t('compare.swap_tip'), () => setSwapped());
    const seamBtn = ctlBtn('│', t('compare.seam_hide'),
                           () => setSeamHidden());
    // Everything the collapse toggle hides lives in one group, so
    // collapsing is a single rule rather than a per-button sweep.
    const ctlGroup = el('div', { class: 'cmpi-ctl-group' },
      ctlBtn('−', t('compare.zoom_out'), () => pz.zoomBy(1 / 1.2)),
      ctlBtn('+', t('compare.zoom_in'), () => pz.zoomBy(1.2)),
      ctlBtn('1:1', t('compare.actual_size'), () => pz.actual()),
      ctlBtn('Fit', t('compare.fit'), () => pz.fit()),
      ctlBtn('⊙', t('compare.center'), () => pz.center()),
      swapBtn,
      seamBtn,
      modeRow);
    // First in the row, so collapsing the rest doesn't move it out from
    // under the pointer that just clicked it.
    const collapseBtn = ctlBtn('«', '', () => setCollapsed());
    collapseBtn.classList.add('cmpi-toggle');
    const controls = el('div', { class: 'cmpi-controls' },
      collapseBtn, ctlGroup);
    viewport.append(controls);
    body.append(el('div', { class: 'cmpi-view' }, viewport, status));
    if (actions) { actions.append(changeBtn()); }

    pz.attach(viewport);

    // ---- image state ------------------------------------------------
    let bmpA = null, bmpB = null;      // decoded bitmaps (or null)
    let iw = 0, ih = 0;                // common size from the backend
    let version = 0;                   // pair version currently expected
    const setStatus = (m) => { status.textContent = m; };
    setStatus(t('compare.connecting'));

    function releaseSlot(slot) {
      const b = slot === 'a' ? bmpA : bmpB;
      if (b && typeof b.close === 'function') {
        try { b.close(); } catch (e) {}
      }
      if (slot === 'a') { bmpA = null; } else { bmpB = null; }
    }

    function onPair(msg) {
      version = msg.version || 0;
      const w = msg.width | 0, h = msg.height | 0;
      if (w > 0 && h > 0 && (w !== iw || h !== ih)) {
        iw = w; ih = h;
        pz.setIntrinsic(iw, ih);
      }
      // Blank whichever slot this pair does not carry, so a stage that
      // loses an input goes black instead of showing a stale image.
      if (!msg.has_a) { releaseSlot('a'); }
      if (!msg.has_b) { releaseSlot('b'); }
      pz.setEnabled(!!(msg.has_a || msg.has_b));
      setStatus((msg.has_a || msg.has_b) ? '' : t('compare.no_images'));
      draw();
    }

    function onBinary(header, payload) {
      if (header.m !== 'image') { return; }
      const slot = header.slot === 'b' ? 'b' : 'a';
      const at = version;
      decode(payload).then((bmp) => {
        if (destroyed || !bmp) { return; }
        // A newer pair landed while this one decoded: drop it rather
        // than show half of an older comparison.
        if (at !== version) {
          if (typeof bmp.close === 'function') {
            try { bmp.close(); } catch (e) {}
          }
          return;
        }
        releaseSlot(slot);
        if (slot === 'a') { bmpA = bmp; } else { bmpB = bmp; }
        draw();
      }).catch(() => {});
    }

    // createImageBitmap decodes off the main thread and yields something
    // canvas can draw directly; the <img> path is the fallback.
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

    // ---- rendering ---------------------------------------------------
    // Content rect for an image centred in the box `pos` at the shared
    // scale + pan.
    function rectIn(s, pos) {
      const dw = iw * s.scale, dh = ih * s.scale;
      return { x: pos.x + (pos.w - dw) / 2 + s.tx,
               y: pos.y + (pos.h - dh) / 2 + s.ty, w: dw, h: dh };
    }

    // POSITION and CLIP are separate on purpose. `pos` is the box the
    // image is centred in; `clip` is the region of the canvas its pixels
    // may occupy. They coincide for the side-by-side modes, where each
    // image really does live in its own half -- but NOT for the wipes,
    // where both images are positioned over the SAME full viewport and
    // the split only chooses which one's pixels show through. Deriving
    // the position from the clip (the obvious shortcut) makes the two
    // images slide as the seam moves, which defeats a wipe: the whole
    // point is that the seam travels across a single stationary picture.
    function blit(g, img, s, pos, clip) {
      if (!img) { return; }             // absent input -> leave it black
      if (clip.w <= 0 || clip.h <= 0) { return; }
      const r = rectIn(s, pos);
      g.save();
      g.beginPath();
      g.rect(clip.x, clip.y, clip.w, clip.h);
      g.clip();
      g.drawImage(img, r.x, r.y, r.w, r.h);
      g.restore();
    }

    // Where the wipe seam falls, snapped to a whole DEVICE pixel.
    // A clip edge landing mid-pixel is ANTIALIASED, and two abutting
    // clips then each cover the boundary pixel only partly -- the two
    // coverages are applied to separate draws, so they don't add up to
    // one, and the black backdrop shows through the shortfall. That is
    // the faint dark line that survives hiding the seam. Snapping in
    // DEVICE px, not CSS px, is the point: at a fractional
    // devicePixelRatio (a scaled display, or Firefox, which reports the
    // true ratio where others round to 2) a whole CSS pixel is still
    // half a device pixel, which is why the line shows at most seam
    // positions but not all of them.
    function seamPos(extent) {
      const dpr = window.devicePixelRatio || 1;
      return Math.round(wipe * extent * dpr) / dpr;
    }

    function divider(g, x1, y1, x2, y2) {
      g.save();
      g.strokeStyle = 'rgba(255,255,255,.55)';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
      g.restore();
    }

    function draw() {
      const vw = viewport.clientWidth, vh = viewport.clientHeight;
      if (vw <= 0 || vh <= 0) { return; }
      // Back the canvas at device resolution so a 1:1 zoom really is one
      // image pixel per device pixel on a HiDPI screen.
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

      const s = pz.state();
      if (!iw || !ih) { placeHandle(vw, vh); return; }
      // Magnifying: show the pixels rather than a blur -- this view
      // exists to be looked at closely. Minifying keeps the filter.
      g.imageSmoothingEnabled = s.scale < 1.5;

      // The swap is resolved HERE, at draw time, rather than by moving
      // the bitmaps between slots: an incoming pair always lands in the
      // slot the backend named, so a swap can never desynchronise from
      // an update that arrives while it is in effect.
      const imgA = swapped ? bmpB : bmpA;
      const imgB = swapped ? bmpA : bmpB;

      const full = { x: 0, y: 0, w: vw, h: vh };
      if (mode === 'a') {
        blit(g, imgA, s, full, full);
      } else if (mode === 'b') {
        blit(g, imgB, s, full, full);
      } else if (mode === 'lr') {
        // Two panes: each image is positioned in, and confined to, its
        // own half.
        const half = vw / 2;
        const left  = { x: 0, y: 0, w: half, h: vh };
        const right = { x: half, y: 0, w: vw - half, h: vh };
        blit(g, imgA, s, left, left);
        blit(g, imgB, s, right, right);
        divider(g, half, 0, half, vh);
      } else if (mode === 'tb') {
        const half = vh / 2;
        const top = { x: 0, y: 0, w: vw, h: half };
        const bot = { x: 0, y: half, w: vw, h: vh - half };
        blit(g, imgA, s, top, top);
        blit(g, imgB, s, bot, bot);
        divider(g, 0, half, vw, half);
      } else if (mode === 'wipev') {
        // One picture, two sources. Both images are positioned over the
        // FULL viewport -- identically, so they overlap exactly -- and
        // the seam only decides which one supplies each column.
        // B goes UNDERNEATH, across the whole frame, and A is laid over
        // its half, so the two never merely abut: whatever softness the
        // clip edge has blends A into B rather than into the backdrop.
        // Belt and braces with seamPos -- an exact edge shouldn't be
        // antialiased at all, but nothing here has to depend on that.
        // The images are padded to a common size and share one
        // transform, so B under A covers precisely the pixels A covers,
        // letterboxing included. With no A there is nothing to lay over
        // it and B keeps its own half, leaving A's side black.
        const sx = seamPos(vw);
        blit(g, imgB, s, full,
             imgA ? full : { x: sx, y: 0, w: vw - sx, h: vh });
        blit(g, imgA, s, full, { x: 0, y: 0, w: sx, h: vh });
      } else if (mode === 'wipeh') {
        const sy = seamPos(vh);
        blit(g, imgB, s, full,
             imgA ? full : { x: 0, y: sy, w: vw, h: vh - sy });
        blit(g, imgA, s, full, { x: 0, y: 0, w: vw, h: sy });
      }
      placeHandle(vw, vh);
    }

    // ---- wipe handle -------------------------------------------------
    // True while the seam is being dragged. Declared before placeHandle
    // because that rebuilds the handle's class list from this state.
    let dragging = false;

    function placeHandle(vw, vh) {
      if (!isWipe(mode)) {
        handle.style.display = 'none';
        return;
      }
      handle.style.display = '';
      // Rebuild the WHOLE class list from state. placeHandle runs on
      // every frame of a drag, so assigning a partial string here would
      // strip `dragging` on the first move and take the grab dot with it.
      handle.className = 'cmpi-handle ' + (mode === 'wipev' ? 'v' : 'h')
                       + (seamHidden ? ' line-hidden' : '')
                       + (dragging ? ' dragging' : '');
      // Placed at the SNAPPED seam, the same one draw() clips to, so the
      // visible line sits on the pixels it claims to divide rather than
      // up to half a pixel off them.
      if (mode === 'wipev') {
        handle.style.left = seamPos(vw).toFixed(3) + 'px';
        handle.style.top = '';
      } else {
        handle.style.top = seamPos(vh).toFixed(3) + 'px';
        handle.style.left = '';
      }
    }

    // The handle takes its own pointer events so dragging the seam never
    // reaches the viewport's pan handler.
    handle.addEventListener('pointerdown', (e) => {
      if (!isWipe(mode)) { return; }
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      // Keep the grab dot up for the whole drag: pointer capture can
      // carry the pointer off the hit band, and :hover would drop out.
      handle.classList.add('dragging');
      try { handle.setPointerCapture(e.pointerId); } catch (x) {}
    });
    handle.addEventListener('pointermove', (e) => {
      if (!dragging) { return; }
      e.stopPropagation();
      const r = viewport.getBoundingClientRect();
      const f = (mode === 'wipev')
        ? (e.clientX - r.left) / Math.max(1, r.width)
        : (e.clientY - r.top) / Math.max(1, r.height);
      wipe = Math.min(1, Math.max(0, f));
      draw();
    });
    const endDrag = (e) => {
      if (!dragging) { return; }
      dragging = false;
      handle.classList.remove('dragging');
      try { handle.releasePointerCapture(e.pointerId); } catch (x) {}
      cfg.wipe = wipe;
      saveConfig();
    };
    handle.addEventListener('pointerup', endDrag);
    handle.addEventListener('pointercancel', endDrag);

    // ---- mode --------------------------------------------------------
    function setMode(m) {
      mode = m;
      cfg.mode = m;
      saveConfig();
      syncModeButtons();
      syncSeamButton();          // the seam only exists in the wipes
      // lr/tb change the pane the fit scale is derived from, so the
      // controller has to recompute before the redraw.
      pz.refresh();
      draw();
    }
    function syncModeButtons() {
      for (const [id, b] of modeBtns) {
        b.classList.toggle('on', id === mode);
      }
    }

    // ---- swap / collapse ----------------------------------------------
    // Exchange the A and B roles. Only a redraw: the transform, the mode
    // and the wipe position all still mean the same thing, so the user
    // keeps the exact framing they were inspecting and just sees the two
    // images trade places -- which is the whole point of the button.
    function setSwapped(v) {
      swapped = (v === undefined) ? !swapped : !!v;
      cfg.swapped = swapped;
      saveConfig();
      syncSwapButton();
      draw();
    }
    function syncSwapButton() {
      swapBtn.classList.toggle('on', swapped);
      // Say which way round it currently is, so the state is readable
      // without having to recognise the images.
      swapBtn.title = swapped ? t('compare.swap_on') : t('compare.swap_tip');
    }

    // Hide the seam's line, shadow and grab dot -- everything painted
    // over the picture -- while leaving the seam itself draggable. The
    // point is to judge detail right AT the seam, which is exactly where
    // a 2px line and its drop shadow sit.
    function setSeamHidden(v) {
      seamHidden = (v === undefined) ? !seamHidden : !!v;
      cfg.seamHidden = seamHidden;
      saveConfig();
      syncSeamButton();
      draw();                           // placeHandle re-applies the class
    }
    function syncSeamButton() {
      seamBtn.classList.toggle('on', seamHidden);
      seamBtn.title = seamHidden
        ? t('compare.seam_show') : t('compare.seam_hide');
      // Only the wipes have a seam; dim it elsewhere rather than drop it,
      // so the control row keeps its shape across mode changes.
      seamBtn.disabled = !isWipe(mode);
    }

    function setCollapsed(v) {
      collapsed = (v === undefined) ? !collapsed : !!v;
      cfg.collapsed = collapsed;
      saveConfig();
      syncCollapse();
    }
    function syncCollapse() {
      controls.classList.toggle('collapsed', collapsed);
      collapseBtn.textContent = collapsed ? '»' : '«';
      collapseBtn.title = collapsed
        ? t('compare.controls_show') : t('compare.controls_hide');
    }

    syncModeButtons();
    syncSwapButton();
    syncSeamButton();
    syncCollapse();
    draw();

    viewer = {
      onPair, onBinary, setStatus,
      destroy() {
        releaseSlot('a');
        releaseSlot('b');
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
    teardownViewer();
    channel.close();
  };
}
