// Composer view: a free-form dashboard where any other view is inserted
// as a panel that can FLOAT (movable/resizable window), DOCK to a side
// (left/right/top/bottom, resizable, several per side), or be MAXIMIZED
// as the background. Layout is stored in relative ratios (% of the stage)
// so it scales with the browser size, and can be saved/loaded as JSON --
// standalone, or associated with a loaded pipeline.
//
// Panel types come from PANEL_TYPES, a registry mapping a type id to a
// label / icon / mount(body, actions, config, ctx). New panel types are
// added there; the layout engine and serialization are type-agnostic, so
// future view types drop in without touching the rest of this file.
//
// Individual User-I/O sub-views (text I/O, live preview, HLS, session
// log) are exposed as their OWN panel types, per the spec.

import { el, clear, openMenu, openModal, toast } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

import { mountPipelineManager, mountPipelineEditor }
  from './pipeline-manager.js';
import { mountProfiler } from './profiler.js';
import { mountDatabase } from './database.js';
import { mountFileBrowser } from './file-browser.js';
import { mountUserIo } from './user-io.js';
import { mountPreview } from './preview-video.js';
import { mountHlsVideo } from './hls-video.js';
import { mountLog } from './log.js';

// ---- panel registry (extensible) ----------------------------------
// mount(body, actions, config, ctx) -> cleanup | void. `ctx.onTitle(str)`
// lets a panel rename its title bar; `config` is an opaque per-panel blob
// that a view may read (and future views may serialize into).
const PANEL_TYPES = [
  { type: 'pipelines', labelKey: 'nav.pipelines', icon: 'pipeline',
    mount: (b, a, cfg, ctx) => hostEditor(mountPipelineManager(b), ctx) },
  // The stage/config editor for one designated pipeline (no selector). The
  // canvas shows run/pause/stop; the stage:config split is saved in config.
  { type: 'pipeline-editor', labelKey: 'composer.pipeline_editor',
    icon: 'pipeline', needsPipeline: true,
    mount: (b, a, cfg, ctx) => hostEditor(mountPipelineEditor(b, {
      pipelineId: cfg.pipeline,
      split: cfg.split,
      onSplit: (s) => { cfg.split = s;
        if (ctx.onConfigChange) { ctx.onConfigChange(); } },
      // The bound pipeline was unloaded and the operator picked another
      // from the rebind menu: re-point the panel config + title so the
      // choice survives a save/reload.
      onRebind: (id) => { cfg.pipeline = id;
        ctx.setTitle(t('composer.pipeline_editor') + ': ' + id);
        if (ctx.onConfigChange) { ctx.onConfigChange(); } },
    }), ctx) },
  { type: 'profiler', labelKey: 'nav.profiler', icon: 'profiler',
    mount: (b) => mountProfiler(b) },
  { type: 'database', labelKey: 'nav.database', icon: 'database',
    mount: (b) => mountDatabase(b) },
  { type: 'files', labelKey: 'nav.files', icon: 'files',
    mount: (b) => mountFileBrowser(b) },
  { type: 'text-io', labelKey: 'nav.io', icon: 'io',
    mount: (b, a) => mountUserIo(b, a) },
  { type: 'preview', labelKey: 'io.preview', icon: 'video',
    mount: (b, a, cfg, ctx) => mountPreview(b, a, {
      onTitle: ctx.onTitle,
      // Persisted (pipeline, stage) designation -- watched + auto-connected
      // when it goes live; saved back to the panel config on change.
      designation: cfg.designation,
      onDesignate: (d) => { cfg.designation = d;
        if (ctx.onConfigChange) { ctx.onConfigChange(); } },
    }) },
  { type: 'hls', labelKey: 'io.hls', icon: 'video',
    mount: (b, a, cfg, ctx) => mountHlsVideo(b, a,
      { onTitle: ctx.onTitle, stream: cfg && cfg.stream }) },
  { type: 'log', labelKey: 'io.session_log', icon: 'log',
    mount: (b, a) => mountLog(b, a) },
];
const typeDef = (type) => PANEL_TYPES.find((d) => d.type === type) || null;

// Register an editor panel's re-arm hook with the host (via ctx) and
// return its cleanup, so the composer resumes the buffer-fullness overlay
// when the panel is shown again after a nav switch.
function hostEditor(handle, ctx) {
  if (handle && handle.onShow && ctx.registerShow) {
    ctx.registerShow(handle.onShow);
  }
  return (handle && handle.cleanup) || null;
}

const SIDES = ['left', 'right', 'top', 'bottom'];
// Floating panels are renumbered into [FLOAT_Z, FLOAT_Z+n) each render so
// they stay below the menu/modal layer (background=1, docked=5, ctx-menu
// z=60). Docked/background sit below; the menu always wins.
const FLOAT_Z = 10;
const LS_CUR = 'vpipe_composer_current';       // auto-persisted layout
const LS_PL  = (id) => 'vpipe_composer_pl_' + id;   // per-pipeline layout

const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));

// One live composer instance persists for the life of the page so its
// panels (a playing video, a streaming console) survive nav switches.
let singleton = null;

export function mountComposer(container) {
  if (!singleton) { singleton = build(); }
  clear(container);
  container.append(singleton.root);
  singleton.onShow();
}

function build() {
  // ---- state ------------------------------------------------------
  const state = {
    floating: [],                       // panels in float mode
    background: null,                   // the maximized panel, or null
    docks: {
      left:   { size: 24, panels: [] }, // size = % perpendicular (of stage)
      right:  { size: 24, panels: [] },
      top:    { size: 30, panels: [] },
      bottom: { size: 30, panels: [] },
    },
    nextZ: 10,
    nextId: 1,
    pipeline: null,                     // associated pipeline id (or null)
  };

  // ---- DOM shell --------------------------------------------------
  const addBtn = tbBtn('composer.add', 'plus');
  const saveBtn = tbBtn('composer.save', 'save');
  const loadBtn = tbBtn('composer.load', 'load');
  const clearBtn = tbBtn('composer.clear', 'trash');
  const plLabel = el('span', { class: 'cmp-pl' });
  const toolbar = el('div', { class: 'cmp-toolbar' },
    addBtn, el('span', { class: 'cmp-tb-sep' }), saveBtn, loadBtn,
    el('span', { class: 'cmp-grow' }), plLabel, clearBtn);

  const emptyEl = el('div', { class: 'cmp-empty' }, t('composer.empty'));
  const handles = el('div', { class: 'cmp-handles' });
  const stage = el('div', { class: 'cmp-stage' }, emptyEl, handles);
  const root = el('div', { class: 'cmp' }, toolbar, stage);

  // ---- helpers ----------------------------------------------------
  function tbBtn(labelKey, icon) {
    return el('button', { class: 'btn ghost mini', type: 'button' },
      makeIcon(icon, 'sm'), el('span', {}, t(labelKey)));
  }
  const hasPanels = () =>
    state.floating.length > 0 || state.background !== null ||
    SIDES.some((s) => state.docks[s].panels.length > 0);
  function allPanels() {
    const a = [...state.floating,
      ...SIDES.flatMap((s) => state.docks[s].panels)];
    if (state.background) { a.push(state.background); }
    return a;
  }

  function place(node, x, y, w, h, z) {
    node.style.left = x + 'px';
    node.style.top = y + 'px';
    node.style.width = Math.max(0, w) + 'px';
    node.style.height = Math.max(0, h) + 'px';
    if (z != null) { node.style.zIndex = z; }
  }
  function setMode(p) {
    p.el.className = 'cmp-panel cmp-' + p.mode;
  }
  function bringToFront(p) {
    // A monotonic order hint only; render() renumbers floating z into the
    // compact FLOAT_Z band (so it can never exceed the menu's z-index).
    p.z = state.nextZ++;
    scheduleRender();
  }

  // ---- layout / render --------------------------------------------
  let rafId = 0;
  function scheduleRender() {
    if (rafId) { return; }
    rafId = requestAnimationFrame(() => { rafId = 0; render(); });
  }

  function layoutStrip(panels, dir, x, y, w, h) {
    const total = panels.reduce(
      (s, p) => s + Math.max(0.05, p.extent), 0) || 1;
    let off = 0;
    for (const p of panels) {
      const frac = Math.max(0.05, p.extent) / total;
      if (dir === 'v') {
        const ph = frac * h; place(p.el, x, y + off, w, ph, 5); off += ph;
      } else {
        const pw = frac * w; place(p.el, x + off, y, pw, h, 5); off += pw;
      }
    }
  }

  function render() {
    const r = stage.getBoundingClientRect();
    const CW = r.width, CH = r.height;
    emptyEl.hidden = hasPanels();
    if (CW <= 0 || CH <= 0) { return; }
    const L = state.docks.left, R = state.docks.right;
    const T = state.docks.top, B = state.docks.bottom;
    const lw = L.panels.length ? clamp(L.size, 5, 85) / 100 * CW : 0;
    const rw = R.panels.length ? clamp(R.size, 5, 85) / 100 * CW : 0;
    const th = T.panels.length ? clamp(T.size, 5, 85) / 100 * CH : 0;
    const bh = B.panels.length ? clamp(B.size, 5, 85) / 100 * CH : 0;
    const cx = lw, cy = th;
    const cw = Math.max(0, CW - lw - rw), ch = Math.max(0, CH - th - bh);
    if (state.background) { place(state.background.el, cx, cy, cw, ch, 1); }
    layoutStrip(L.panels, 'v', 0, 0, lw, CH);
    layoutStrip(R.panels, 'v', CW - rw, 0, rw, CH);
    layoutStrip(T.panels, 'h', lw, 0, cw, th);
    layoutStrip(B.panels, 'h', lw, CH - bh, cw, bh);
    // Renumber floating z-indices into a compact band starting at FLOAT_Z,
    // preserving stacking order (sort on the running z, then reassign). This
    // keeps them safely below the menu / modal layer (ctx-menu z=60) so a
    // panel never climbs high enough to paint over its own ... menu -- the
    // `nextZ`-driven bring-to-front counter is otherwise unbounded.
    [...state.floating]
      .sort((a, b) => (a.z || 0) - (b.z || 0))
      .forEach((p, i) => { p.z = FLOAT_Z + i; });
    for (const p of state.floating) {
      place(p.el, p.x / 100 * CW, p.y / 100 * CH,
        p.w / 100 * CW, p.h / 100 * CH, p.z);
    }
    buildHandles(CW, CH, { lw, rw, th, bh, cx, cy, cw, ch });
  }

  // Drag helper: run `onMove(e)` until pointer-up, then persist.
  function drag(onMove) {
    const mv = (e) => onMove(e);
    const up = () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
      persist();
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  }

  // Strip edge handles (perpendicular resize) + dividers (along-side).
  function buildHandles(CW, CH, m) {
    clear(handles);
    const L = state.docks.left, R = state.docks.right;
    const T = state.docks.top, B = state.docks.bottom;
    if (L.panels.length) {
      edgeHandle('v', m.lw, 0, CH,
        (e, r) => { L.size = clamp((e.clientX - r.left) / r.width * 100,
          5, 85); scheduleRender(); });
      dividers(L.panels, 'v', 0, 0, m.lw, CH);
    }
    if (R.panels.length) {
      edgeHandle('v', CW - m.rw, 0, CH,
        (e, r) => { R.size = clamp((r.right - e.clientX) / r.width * 100,
          5, 85); scheduleRender(); });
      dividers(R.panels, 'v', CW - m.rw, 0, m.rw, CH);
    }
    if (T.panels.length) {
      edgeHandle('h', m.th, m.lw, m.cw,
        (e, r) => { T.size = clamp((e.clientY - r.top) / r.height * 100,
          5, 85); scheduleRender(); });
      dividers(T.panels, 'h', m.lw, 0, m.cw, m.th);
    }
    if (B.panels.length) {
      edgeHandle('h', CH - m.bh, m.lw, m.cw,
        (e, r) => { B.size = clamp((r.bottom - e.clientY) / r.height * 100,
          5, 85); scheduleRender(); });
      dividers(B.panels, 'h', m.lw, CH - m.bh, m.cw, m.bh);
    }
  }

  // A perpendicular strip-resize handle. `dir` 'v' => vertical bar at x=pos
  // spanning [off, off+len] in y; 'h' => horizontal bar at y=pos.
  function edgeHandle(dir, pos, off, len, onDrag) {
    const h = el('div', { class: 'cmp-edge cmp-edge-' + dir });
    if (dir === 'v') { place(h, pos - 3, off, 6, len, 40); }
    else { place(h, off, pos - 3, len, 6, 40); }
    h.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      const r = stage.getBoundingClientRect();
      drag((e) => onDrag(e, r));
    });
    handles.append(h);
  }

  // Dividers between adjacent panels of a strip (resize the along-side
  // split by shifting weight between the two neighbours).
  function dividers(panels, dir, x, y, w, h) {
    const total = panels.reduce(
      (s, p) => s + Math.max(0.05, p.extent), 0) || 1;
    let off = 0;
    for (let i = 0; i < panels.length - 1; i++) {
      const a = panels[i], b = panels[i + 1];
      off += Math.max(0.05, a.extent) / total * (dir === 'v' ? h : w);
      const hd = el('div', { class: 'cmp-div cmp-div-' + dir });
      if (dir === 'v') { place(hd, x, y + off - 3, w, 6, 40); }
      else { place(hd, x + off - 3, y, 6, h, 40); }
      hd.addEventListener('pointerdown', (ev) => {
        ev.preventDefault();
        const sa = a.extent, sb = b.extent;
        const s0 = dir === 'v' ? ev.clientY : ev.clientX;
        const span = dir === 'v' ? h : w;
        drag((e) => {
          const d = ((dir === 'v' ? e.clientY : e.clientX) - s0)
            / span * total;
          a.extent = Math.max(0.05, sa + d);
          b.extent = Math.max(0.05, sb - d);
          scheduleRender();
        });
      });
      handles.append(hd);
    }
  }

  // ---- panel lifecycle --------------------------------------------
  function createPanel(type, config, title) {
    const def = typeDef(type);
    if (!def) { return null; }              // unknown type: skip (robust)
    const id = 'p' + (state.nextId++);
    const titleEl = el('span', { class: 'cmp-title' },
      title || t(def.labelKey));
    const actions = el('span', { class: 'cmp-actions' });
    const menuBtn = el('button', { class: 'cmp-tb-btn',
      title: t('io.split_options'), type: 'button' }, '⋯');
    const closeBtn = el('button', { class: 'cmp-tb-btn',
      title: t('composer.close'), type: 'button' }, '×');
    const bar = el('div', { class: 'cmp-titlebar' },
      makeIcon(def.icon, 'sm'), titleEl, el('span', { class: 'cmp-grow' }),
      actions, menuBtn, closeBtn);
    const body = el('div', { class: 'cmp-body' });
    const rs = el('div', { class: 'cmp-resize' });
    const panelEl = el('div', { class: 'cmp-panel cmp-float' },
      bar, body, rs);

    const p = { id, type, title: title || null, config: config || {},
      el: panelEl, bodyEl: body, titleEl, actions, cleanup: null,
      mode: 'float', side: null,
      x: 12, y: 10, w: 46, h: 56, z: state.nextZ++, extent: 1 };

    const ctx = { onTitle: (txt) => { titleEl.textContent = txt; },
      // Like onTitle, but ALSO persists the new name as the panel title
      // (onTitle is ephemeral -- for live status text that shouldn't be
      // saved; setTitle is for a durable rename, e.g. an editor rebind).
      setTitle: (txt) => { p.title = txt; titleEl.textContent = txt; },
      onConfigChange: () => persist(),
      // An editor panel registers a re-arm hook (see hostEditor); the
      // composer calls it when this view is shown again after a switch.
      registerShow: (fn) => { p.onShow = fn; } };
    try {
      p.cleanup = def.mount(body, actions, p.config, ctx) || null;
    } catch (e) {
      body.append(el('div', { class: 'cmp-err' },
        String((e && e.message) || e)));
    }

    bar.addEventListener('pointerdown', (ev) => {
      // Don't start a window drag when the pointer lands on an interactive
      // control in the title bar -- the ... / x buttons, or a view's own
      // header controls (e.g. the log-level <select>, which live in
      // .cmp-actions). Dragging preventDefault()s the event and would stop
      // those controls from opening/changing.
      if (ev.target.closest(
          'button, select, input, textarea, label, .cmp-actions')) {
        return;
      }
      if (p.mode === 'float') { bringToFront(p); startMove(p, ev); }
    });
    panelEl.addEventListener('pointerdown', () => {
      if (p.mode === 'float') { bringToFront(p); }
    }, true);
    rs.addEventListener('pointerdown', (ev) => {
      if (p.mode === 'float') { startResize(p, ev); }
    });
    menuBtn.addEventListener('click', (ev) => {
      ev.stopPropagation(); openPanelMenu(p, menuBtn);
    });
    closeBtn.addEventListener('click', (ev) => {
      ev.stopPropagation(); removePanel(p);
    });
    stage.append(panelEl);
    return p;
  }

  function detach(p) {
    const i = state.floating.indexOf(p);
    if (i >= 0) { state.floating.splice(i, 1); }
    for (const s of SIDES) {
      const j = state.docks[s].panels.indexOf(p);
      if (j >= 0) { state.docks[s].panels.splice(j, 1); }
    }
    if (state.background === p) { state.background = null; }
  }
  function assignFloatDefaults(p) {
    const n = state.floating.length;
    p.x = 8 + (n % 6) * 4;
    p.y = 8 + (n % 6) * 4;
    p.w = 46; p.h = 56; p.z = state.nextZ++;
  }
  function toFloat(p) {
    detach(p); p.mode = 'float'; setMode(p);
    assignFloatDefaults(p); state.floating.push(p);
    render(); persist();
  }
  function toDock(p, side) {
    detach(p); p.mode = 'dock'; p.side = side; setMode(p);
    if (!(p.extent > 0)) { p.extent = 1; }
    state.docks[side].panels.push(p);
    render(); persist();
  }
  function toBg(p) {
    const prev = state.background;
    detach(p);
    if (prev && prev !== p) {
      prev.mode = 'float'; setMode(prev);
      assignFloatDefaults(prev); state.floating.push(prev);
    }
    p.mode = 'bg'; setMode(p); state.background = p;
    render(); persist();
  }
  function removePanel(p) {
    detach(p);
    if (p.cleanup) { try { p.cleanup(); } catch (e) { /* ignore */ } }
    p.el.remove();
    render(); persist();
  }
  function clearAll() {
    const all = [...state.floating,
      ...SIDES.flatMap((s) => state.docks[s].panels)];
    if (state.background) { all.push(state.background); }
    for (const p of all) {
      if (p.cleanup) { try { p.cleanup(); } catch (e) { /* ignore */ } }
      p.el.remove();
    }
    state.floating = [];
    state.background = null;
    for (const s of SIDES) { state.docks[s].panels = []; }
  }

  function addFloating(type, config, title) {
    const p = createPanel(type, config, title);
    if (!p) { return; }
    assignFloatDefaults(p);
    state.floating.push(p);
    bringToFront(p);
    render(); persist();
  }
  async function addPanel(type) {
    const def = typeDef(type);
    if (def && def.needsPipeline) {
      // Require the user to designate a loaded pipeline at creation.
      const list = await pipelines();
      if (!list.length) { toast(t('composer.no_pipeline'), 'error'); return; }
      const r = addBtn.getBoundingClientRect();
      openMenu(r.left, r.bottom + 4, list.map((id) => ({
        label: id, onClick: () => addFloating(type,
          { pipeline: id, split: 2 / 3 }, t(def.labelKey) + ': ' + id) })));
      return;
    }
    addFloating(type, {}, null);
  }

  function openPanelMenu(p, anchor) {
    const r = anchor.getBoundingClientRect();
    const items = [];
    if (p.mode !== 'float') {
      items.push({ label: t('composer.float'), onClick: () => toFloat(p) });
    }
    items.push({ label: t('composer.dock_left'),
      onClick: () => toDock(p, 'left') });
    items.push({ label: t('composer.dock_right'),
      onClick: () => toDock(p, 'right') });
    items.push({ label: t('composer.dock_top'),
      onClick: () => toDock(p, 'top') });
    items.push({ label: t('composer.dock_bottom'),
      onClick: () => toDock(p, 'bottom') });
    items.push(null);
    if (p.mode === 'bg') {
      items.push({ label: t('composer.restore'), onClick: () => toFloat(p) });
    } else {
      items.push({ label: t('composer.maximize'), onClick: () => toBg(p) });
    }
    items.push(null);
    items.push({ label: t('composer.close'), danger: true,
      onClick: () => removePanel(p) });
    openMenu(r.left, r.bottom + 4, items);
  }

  // ---- float drag / resize ----------------------------------------
  function startMove(p, ev) {
    ev.preventDefault();
    const r = stage.getBoundingClientRect();
    const sx = ev.clientX, sy = ev.clientY, ox = p.x, oy = p.y;
    drag((e) => {
      p.x = clamp(ox + (e.clientX - sx) / r.width * 100, 0, 100 - p.w);
      p.y = clamp(oy + (e.clientY - sy) / r.height * 100, 0, 100 - p.h);
      scheduleRender();
    });
  }
  function startResize(p, ev) {
    ev.preventDefault(); ev.stopPropagation();
    const r = stage.getBoundingClientRect();
    const sx = ev.clientX, sy = ev.clientY, ow = p.w, oh = p.h;
    drag((e) => {
      p.w = clamp(ow + (e.clientX - sx) / r.width * 100, 15, 100 - p.x);
      p.h = clamp(oh + (e.clientY - sy) / r.height * 100, 12, 100 - p.y);
      scheduleRender();
    });
  }

  // ---- serialize / persist ----------------------------------------
  const cfgOf = (p) => ({ type: p.type, title: p.title, config: p.config });
  function serialize() {
    const norm = (panels) => {
      const tot = panels.reduce(
        (s, p) => s + Math.max(0.05, p.extent), 0) || 1;
      return panels.map((p) => ({ ...cfgOf(p),
        extent: +(Math.max(0.05, p.extent) / tot * 100).toFixed(2) }));
    };
    const dock = (d) => ({ size: +clamp(d.size, 5, 85).toFixed(2),
      panels: norm(d.panels) });
    return {
      version: 1,
      pipeline: state.pipeline || null,
      background: state.background ? cfgOf(state.background) : null,
      docks: { left: dock(state.docks.left), right: dock(state.docks.right),
        top: dock(state.docks.top), bottom: dock(state.docks.bottom) },
      floating: state.floating.map((p) => ({ ...cfgOf(p),
        x: +p.x.toFixed(2), y: +p.y.toFixed(2),
        w: +p.w.toFixed(2), h: +p.h.toFixed(2) })),
    };
  }
  function deserialize(json) {
    clearAll();
    if (!json || typeof json !== 'object') { render(); return; }
    state.pipeline = json.pipeline || null;
    const mk = (sp) => sp && createPanel(sp.type, sp.config, sp.title);
    if (json.background) {
      const p = mk(json.background);
      if (p) { p.mode = 'bg'; setMode(p); state.background = p; }
    }
    for (const s of SIDES) {
      const d = json.docks && json.docks[s];
      if (!d) { continue; }
      state.docks[s].size = clamp(Number(d.size) || 25, 5, 85);
      for (const sp of (d.panels || [])) {
        const p = mk(sp);
        if (p) {
          p.mode = 'dock'; p.side = s; setMode(p);
          p.extent = Math.max(0.05, Number(sp.extent) || 1);
          state.docks[s].panels.push(p);
        }
      }
    }
    for (const sp of (json.floating || [])) {
      const p = mk(sp);
      if (p) {
        p.mode = 'float'; setMode(p);
        p.x = num(sp.x, 10); p.y = num(sp.y, 10);
        p.w = num(sp.w, 46); p.h = num(sp.h, 56); p.z = state.nextZ++;
        state.floating.push(p);
      }
    }
    updatePlLabel();
    render(); persist();
  }
  const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);

  let persistT = 0;
  function persist() {
    clearTimeout(persistT);
    persistT = setTimeout(() => {
      try { localStorage.setItem(LS_CUR, JSON.stringify(serialize())); }
      catch (e) { /* storage blocked -- ignore */ }
    }, 300);
  }
  function updatePlLabel() {
    plLabel.textContent = state.pipeline
      ? t('composer.pick_pipeline') + ': ' + state.pipeline : '';
  }

  // ---- toolbar actions --------------------------------------------
  addBtn.addEventListener('click', () => {
    const r = addBtn.getBoundingClientRect();
    openMenu(r.left, r.bottom + 4, PANEL_TYPES.map((d) => ({
      label: t(d.labelKey), onClick: () => addPanel(d.type) })));
  });
  saveBtn.addEventListener('click', () => {
    const r = saveBtn.getBoundingClientRect();
    openMenu(r.left, r.bottom + 4, [
      { label: t('composer.save_file'), onClick: saveToFile },
      { label: t('composer.save_pipeline'),
        onClick: () => saveWithPipeline(r) },
    ]);
  });
  loadBtn.addEventListener('click', () => {
    const r = loadBtn.getBoundingClientRect();
    openMenu(r.left, r.bottom + 4, [
      { label: t('composer.load_file'), onClick: loadFromFile },
      { label: t('composer.load_pipeline'),
        onClick: () => loadForPipeline(r) },
    ]);
  });
  clearBtn.addEventListener('click', () => {
    if (!hasPanels()) { return; }
    openModal({
      title: t('composer.clear'),
      body: el('p', {}, t('composer.confirm_clear')),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('composer.clear'), kind: 'danger', onClick: (c) => {
          c(); clearAll(); state.pipeline = null; updatePlLabel();
          render(); persist();
        } },
      ],
    });
  });

  function saveToFile() {
    const blob = new Blob([JSON.stringify(serialize(), null, 2)],
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: 'composer-layout.json' });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(t('composer.saved'), 'ok');
  }
  async function saveWithPipeline(anchor) {
    const list = await pipelines();
    if (!list.length) { toast(t('composer.no_pipeline'), 'error'); return; }
    openMenu(anchor.left, anchor.bottom + 4, list.map((id) => ({
      label: id, onClick: () => {
        state.pipeline = id; updatePlLabel();
        const doc = serialize(); doc.pipeline = id;
        try { localStorage.setItem(LS_PL(id), JSON.stringify(doc)); }
        catch (e) { /* ignore */ }
        persist();
        toast(t('composer.saved_pl', { id }), 'ok');
      } })));
  }
  function loadFromFile() {
    const inp = el('input', { type: 'file', accept: '.json,application/json' });
    inp.style.display = 'none';
    inp.addEventListener('change', () => {
      const f = inp.files && inp.files[0];
      if (!f) { return; }
      const rd = new FileReader();
      rd.onload = () => {
        try { deserialize(JSON.parse(String(rd.result)));
          toast(t('composer.loaded'), 'ok'); }
        catch (e) { toast(t('composer.load_failed',
          { msg: e.message }), 'error'); }
      };
      rd.readAsText(f);
      inp.remove();
    });
    document.body.append(inp); inp.click();
  }
  async function loadForPipeline(anchor) {
    // Pipelines that have a saved layout in localStorage.
    const saved = [];
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith('vpipe_composer_pl_')) {
          saved.push(k.slice('vpipe_composer_pl_'.length));
        }
      }
    } catch (e) { /* ignore */ }
    if (!saved.length) { toast(t('composer.no_saved'), 'error'); return; }
    openMenu(anchor.left, anchor.bottom + 4, saved.map((id) => ({
      label: id, onClick: () => {
        try {
          const raw = localStorage.getItem(LS_PL(id));
          deserialize(JSON.parse(raw));
          toast(t('composer.loaded'), 'ok');
        } catch (e) {
          toast(t('composer.load_failed', { msg: e.message }), 'error');
        }
      } })));
  }
  async function pipelines() {
    // api.listPipelines() returns a bare array of {id,...}; tolerate a
    // {pipelines:[...]} wrapper too.
    try {
      const d = await api.listPipelines();
      const arr = Array.isArray(d) ? d : (d && d.pipelines) || [];
      return arr.map((p) => (typeof p === 'string' ? p : p.id));
    } catch (e) { return []; }
  }

  // ---- boot -------------------------------------------------------
  const ro = new ResizeObserver(() => scheduleRender());
  ro.observe(stage);
  window.addEventListener('resize', scheduleRender);

  // Restore the last layout (survives page reloads).
  try {
    const raw = localStorage.getItem(LS_CUR);
    if (raw) { deserialize(JSON.parse(raw)); }
  } catch (e) { /* ignore */ }
  updatePlLabel();

  // Called by mountComposer whenever the composer is (re-)shown: re-layout
  // and re-arm every editor panel's live overlay (buffer-fullness poll) so
  // navigating away and back is seamless from the user's point of view.
  return { root, onShow: () => {
    scheduleRender();
    for (const p of allPanels()) {
      if (p.onShow) { try { p.onShow(); } catch (e) { /* ignore */ } }
    }
  } };
}
