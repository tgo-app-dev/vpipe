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
import { openFsDialog, splitPath } from '../fs-dialog.js';

import { mountPipelineManager, mountPipelineEditor }
  from './pipeline-manager.js';
import { mountProfiler } from './profiler.js';
import { mountDatabase } from './database.js';
import { mountFileBrowser } from './file-browser.js';
import { mountUserIo } from './user-io.js';
import { mountHlsVideo } from './hls-video.js';
import { mountLog } from './log.js';
import { stageViewPanelTypes } from '../stage-views.js';
import * as tree from './composer-tree.js';

// ---- panel registry (extensible) ----------------------------------
// mount(body, actions, config, ctx) -> cleanup | void. `ctx.onTitle(str)`
// lets a panel rename its title bar; `config` is an opaque per-panel blob
// that a view may read (and future views may serialize into).
//
// This list holds the APP's own views. Panels a STAGE provides (its
// module embedded in libvpipe -- see stage-views.js) are discovered from
// the backend and appended by registerStageViews() below, so a stage can
// add a panel without this file changing.
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
  { type: 'hls', labelKey: 'io.hls', icon: 'video',
    mount: (b, a, cfg, ctx) => mountHlsVideo(b, a,
      { onTitle: ctx.onTitle, stream: cfg && cfg.stream }) },
  { type: 'log', labelKey: 'io.session_log', icon: 'log',
    mount: (b, a) => mountLog(b, a) },
  // What a split produces: an empty pane offering the type chooser, exactly
  // as the User I/O workspace does. `ctx.choose(type)` swaps this panel for
  // a real one in the same slot. Hidden from the Add menu (`internal`) --
  // an empty panel is only ever reached by splitting.
  { type: 'empty', labelKey: 'io.new_view', icon: 'plus', internal: true,
    mount: (b, a, cfg, ctx) => {
      const grid = el('div', { class: 'cmp-empty-choices' });
      for (const d of PANEL_TYPES) {
        if (d.internal || d.needsPipeline) { continue; }
        grid.append(el('button', { class: 'btn', type: 'button',
          onclick: () => ctx.choose(d.type) },
          makeIcon(d.icon, 'sm'), el('span', {}, t(d.labelKey))));
      }
      b.append(el('div', { class: 'cmp-empty-pane' },
        el('div', { class: 'cmp-empty-title' }, t('io.add_view')), grid));
    } },
];
const typeDef = (type) => PANEL_TYPES.find((d) => d.type === type) || null;

// Append the panels the STAGES registered. Discovery is kicked off at
// module load and is awaited before the singleton is built, so a
// restored layout referring to a stage-provided type (e.g. "preview")
// resolves it -- and so the Add menu and the empty-pane chooser list
// them alongside the app's own.
const stageViewsReady = stageViewPanelTypes()
  .then((types) => {
    for (const d of types) {
      if (!PANEL_TYPES.some((x) => x.type === d.type)) {
        PANEL_TYPES.push(d);
      }
    }
  })
  .catch(() => { /* no backend views: the built-ins still work */ });

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
// Per-pipeline arrangements now live on the backend as auxiliary data
// objects (aux.composer), travelling with the pipeline file -- see
// saveWithPipeline / loadForPipeline.

const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));

// Default `ext` when the user typed a bare name (no extension) in a dialog.
function withExt(p, ext) {
  if (!p) { return p; }
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  const base = i >= 0 ? p.slice(i + 1) : p;
  return base.indexOf('.') === -1 ? p + ext : p;
}

// One live composer instance persists for the life of the page so its
// panels (a playing video, a streaming console) survive nav switches.
let singleton = null;

export function mountComposer(container) {
  if (singleton) {
    clear(container);
    container.append(singleton.root);
    singleton.onShow();
    return;
  }
  // First mount: wait for the stage-provided panel types before building,
  // so restoring a saved layout that uses one finds its definition
  // instead of dropping the panel. Discovery is already in flight from
  // module load, so on a local server this is imperceptible.
  clear(container);
  stageViewsReady.then(() => {
    // The user may have navigated to another view while we waited; that
    // view mounted into this same container, so anything present now is
    // someone else's. Build the singleton regardless (the next mount
    // reuses it) but don't steal the container back.
    if (!singleton) { singleton = build(); }
    if (container.childElementCount > 0) { return; }
    container.append(singleton.root);
    singleton.onShow();
  });
}

function build() {
  // ---- state ------------------------------------------------------
  const state = {
    floating: [],                       // panels in float mode
    // Every docked panel lives in a region TREE (see composer-tree.js), so
    // any of them can be split either way instead of only along its strip.
    // `center` is the area a MAXIMIZED panel fills -- treating it as a dock
    // region is what makes a maximized panel splittable too.
    docks: {
      left:   { size: 24, tree: null }, // size = % perpendicular (of stage)
      right:  { size: 24, tree: null },
      top:    { size: 30, tree: null },
      bottom: { size: 30, tree: null },
      center: { size: 0,  tree: null },
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
  const regionPanels = (r) => tree.panels(state.docks[r].tree);
  const hasPanels = () =>
    state.floating.length > 0 ||
    tree.REGIONS.some((r) => state.docks[r].tree !== null);
  function allPanels() {
    return [...state.floating,
      ...tree.REGIONS.flatMap((r) => regionPanels(r))];
  }
  // A lone panel in `center` IS "maximized", and keeps the seamless
  // background look; once it is split, its panes are ordinary docked panels.
  function refreshModes() {
    for (const r of tree.REGIONS) {
      const solo = r === 'center' && tree.leafCount(state.docks[r].tree) === 1;
      for (const p of regionPanels(r)) {
        p.mode = solo ? 'bg' : 'dock';
        p.region = r;
        setMode(p);
      }
    }
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

  // Place every panel of a region's tree inside `rect`. z 1 for a lone
  // maximized panel (it sits behind everything), 5 for docked panes.
  function layoutRegion(region, rect) {
    const node = state.docks[region].tree;
    if (!node || rect.w <= 0 || rect.h <= 0) { return; }
    const z = (region === 'center' && tree.leafCount(node) === 1) ? 1 : 5;
    for (const s of tree.layout(node, rect)) {
      place(s.panel.el, s.x, s.y, s.w, s.h, z);
    }
  }

  function render() {
    const r = stage.getBoundingClientRect();
    const CW = r.width, CH = r.height;
    emptyEl.hidden = hasPanels();
    if (CW <= 0 || CH <= 0) { return; }
    const L = state.docks.left, R = state.docks.right;
    const T = state.docks.top, B = state.docks.bottom;
    const has = (d) => d.tree !== null;
    const lw = has(L) ? clamp(L.size, 5, 85) / 100 * CW : 0;
    const rw = has(R) ? clamp(R.size, 5, 85) / 100 * CW : 0;
    const th = has(T) ? clamp(T.size, 5, 85) / 100 * CH : 0;
    const bh = has(B) ? clamp(B.size, 5, 85) / 100 * CH : 0;
    const cx = lw, cy = th;
    const cw = Math.max(0, CW - lw - rw), ch = Math.max(0, CH - th - bh);
    layoutRegion('center', { x: cx, y: cy, w: cw, h: ch });
    layoutRegion('left',   { x: 0, y: 0, w: lw, h: CH });
    layoutRegion('right',  { x: CW - rw, y: 0, w: rw, h: CH });
    layoutRegion('top',    { x: lw, y: 0, w: cw, h: th });
    layoutRegion('bottom', { x: lw, y: CH - bh, w: cw, h: bh });
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
    if (L.tree) {
      edgeHandle('v', m.lw, 0, CH,
        (e, r) => { L.size = clamp((e.clientX - r.left) / r.width * 100,
          5, 85); scheduleRender(); });
    }
    if (R.tree) {
      edgeHandle('v', CW - m.rw, 0, CH,
        (e, r) => { R.size = clamp((r.right - e.clientX) / r.width * 100,
          5, 85); scheduleRender(); });
    }
    if (T.tree) {
      edgeHandle('h', m.th, m.lw, m.cw,
        (e, r) => { T.size = clamp((e.clientY - r.top) / r.height * 100,
          5, 85); scheduleRender(); });
    }
    if (B.tree) {
      edgeHandle('h', CH - m.bh, m.lw, m.cw,
        (e, r) => { B.size = clamp((r.bottom - e.clientY) / r.height * 100,
          5, 85); scheduleRender(); });
    }
    treeDividers('center', { x: m.cx, y: m.cy, w: m.cw, h: m.ch });
    treeDividers('left',   { x: 0, y: 0, w: m.lw, h: CH });
    treeDividers('right',  { x: CW - m.rw, y: 0, w: m.rw, h: CH });
    treeDividers('top',    { x: m.lw, y: 0, w: m.cw, h: m.th });
    treeDividers('bottom', { x: m.lw, y: CH - m.bh, w: m.cw, h: m.bh });
  }

  // One draggable divider per split node of a region's tree. Dragging shifts
  // that split's ratio only, so neighbouring splits stay put.
  function treeDividers(region, rect) {
    const node = state.docks[region].tree;
    if (!node || rect.w <= 0 || rect.h <= 0) { return; }
    for (const d of tree.dividers(node, rect)) {
      const hd = el('div', { class: 'cmp-div cmp-div-' + d.dir });
      if (d.dir === 'v') { place(hd, d.x - 3, d.y, 6, d.h, 40); }
      else               { place(hd, d.x, d.y - 3, d.w, 6, 40); }
      hd.addEventListener('pointerdown', (ev) => {
        ev.preventDefault();
        const n = d.node;
        const r0 = typeof n.ratio === 'number' ? n.ratio : 0.5;
        const s0 = d.dir === 'v' ? ev.clientX : ev.clientY;
        const span = d.span || 1;
        drag((e) => {
          const cur = d.dir === 'v' ? e.clientX : e.clientY;
          n.ratio = clamp(r0 + (cur - s0) / span, 0.05, 0.95);
          scheduleRender();
        });
      });
      handles.append(hd);
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
      mode: 'float', region: null,
      x: 12, y: 10, w: 46, h: 56, z: state.nextZ++ };

    const ctx = { onTitle: (txt) => { titleEl.textContent = txt; },
      // Like onTitle, but ALSO persists the new name as the panel title
      // (onTitle is ephemeral -- for live status text that shouldn't be
      // saved; setTitle is for a durable rename, e.g. an editor rebind).
      setTitle: (txt) => { p.title = txt; titleEl.textContent = txt; },
      onConfigChange: () => persist(),
      // An editor panel registers a re-arm hook (see hostEditor); the
      // composer calls it when this view is shown again after a switch.
      registerShow: (fn) => { p.onShow = fn; },
      // An empty pane picking its view: swap this panel for the chosen type
      // in the same tree slot.
      choose: (type) => replacePanel(p, type) };
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
    for (const r of tree.REGIONS) {
      const d = state.docks[r];
      if (d.tree && tree.findLeaf(d.tree, p)) {
        d.tree = tree.removeLeaf(d.tree, p);
      }
    }
  }
  function assignFloatDefaults(p) {
    const n = state.floating.length;
    p.x = 8 + (n % 6) * 4;
    p.y = 8 + (n % 6) * 4;
    p.w = 46; p.h = 56; p.z = state.nextZ++;
  }
  function toFloat(p) {
    detach(p); p.mode = 'float'; p.region = null; setMode(p);
    assignFloatDefaults(p); state.floating.push(p);
    // Leaving a region can change what its survivors look like -- the last
    // panel in `center` goes back to the seamless maximized style.
    refreshModes();
    render(); persist();
  }
  // Dock into a region, appending along that region's natural direction --
  // the placement the old flat strip gave. `center` is a region like any
  // other, so this is also how a panel is maximized.
  function toDock(p, region) {
    detach(p);
    const d = state.docks[region];
    d.tree = tree.appendLeaf(d.tree, p, tree.defaultDir(region));
    refreshModes();
    render(); persist();
  }
  const toBg = (p) => toDock(p, 'center');

  // Split a DOCKED panel, putting a new empty pane beside it. `dir` names
  // the divider, as in the User I/O workspace: 'v' side by side, 'h'
  // stacked. The empty pane offers the panel-type chooser.
  function splitPanel(p, dir) {
    const region = p.region;
    const d = region && state.docks[region];
    if (!d || !d.tree || !tree.findLeaf(d.tree, p)) { return; }
    const np = createPanel('empty', {}, null);
    if (!np) { return; }
    d.tree = tree.splitAt(d.tree, p, dir, np);
    refreshModes();
    render(); persist();
  }

  // Replace an existing panel's view in place, keeping its slot in the tree
  // -- how an empty pane becomes a real one after the chooser is used.
  function replacePanel(p, type) {
    const np = createPanel(type, {}, null);
    if (!np) { return; }
    let swapped = false;
    for (const r of tree.REGIONS) {
      const leaf = state.docks[r].tree
          && tree.findLeaf(state.docks[r].tree, p);
      if (leaf) { leaf.panel = np; swapped = true; break; }
    }
    if (!swapped) {
      const i = state.floating.indexOf(p);
      if (i >= 0) {
        state.floating[i] = np;
        np.x = p.x; np.y = p.y; np.w = p.w; np.h = p.h;
        swapped = true;
      }
    }
    if (!swapped) { removePanel(np); return; }
    if (p.cleanup) { try { p.cleanup(); } catch (e) { /* ignore */ } }
    p.el.remove();
    refreshModes();
    render(); persist();
  }
  function removePanel(p) {
    detach(p);
    if (p.cleanup) { try { p.cleanup(); } catch (e) { /* ignore */ } }
    p.el.remove();
    refreshModes();              // see toFloat
    render(); persist();
  }
  function clearAll() {
    for (const p of allPanels()) {
      if (p.cleanup) { try { p.cleanup(); } catch (e) { /* ignore */ } }
      p.el.remove();
    }
    state.floating = [];
    for (const r of tree.REGIONS) { state.docks[r].tree = null; }
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
    // Splitting applies to any DOCKED panel -- including a maximized one,
    // which is simply the sole occupant of the `center` region.
    if (p.mode !== 'float') {
      items.push(null);
      items.push({ label: t('io.split_v'),
        onClick: () => splitPanel(p, 'v') });
      items.push({ label: t('io.split_h'),
        onClick: () => splitPanel(p, 'h') });
    }
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
    // version 2: each dock region is a TREE (`tree`) rather than a flat
    // `panels` array, and the maximized panel is the `center` region rather
    // than a separate `background`. Version 1 layouts still LOAD -- see
    // deserialize -- including any stored in a pipeline's `aux`.
    const dock = (r) => ({ size: +clamp(state.docks[r].size, 5, 85).toFixed(2),
      tree: tree.toJSON(state.docks[r].tree, cfgOf) });
    return {
      version: 2,
      pipeline: state.pipeline || null,
      docks: { left: dock('left'), right: dock('right'),
        top: dock('top'), bottom: dock('bottom'),
        center: { tree: tree.toJSON(state.docks.center.tree, cfgOf) } },
      floating: state.floating.map((p) => ({ ...cfgOf(p),
        x: +p.x.toFixed(2), y: +p.y.toFixed(2),
        w: +p.w.toFixed(2), h: +p.h.toFixed(2) })),
    };
  }
  function deserialize(json) {
    clearAll();
    if (!json || typeof json !== 'object') { render(); return; }
    state.pipeline = json.pipeline || null;
    const mk = (sp) => (sp ? createPanel(sp.type, sp.config, sp.title) : null);
    // v1 `background` (a single maximized panel) becomes the center region.
    if (json.background) {
      state.docks.center.tree = tree.fromJSON(json.background, mk);
    }
    if (json.docks && json.docks.center) {
      state.docks.center.tree =
          tree.fromJSON(json.docks.center.tree, mk) || state.docks.center.tree;
    }
    for (const s of SIDES) {
      const d = json.docks && json.docks[s];
      if (!d) { continue; }
      state.docks[s].size = clamp(Number(d.size) || 25, 5, 85);
      // v2 tree, else fold a v1 flat strip into one along the region's
      // natural direction, preserving the saved extent weights.
      state.docks[s].tree = d.tree
          ? tree.fromJSON(d.tree, mk)
          : tree.fromLegacy(d.panels, tree.defaultDir(s), mk);
    }
    refreshModes();
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
    openMenu(r.left, r.bottom + 4, PANEL_TYPES
      .filter((d) => !d.internal)
      .map((d) => ({
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

  // Write the arrangement as a standalone JSON file on the SERVER (a
  // Save-As browse into the sandbox), like the pipeline save -- not a
  // client-side download.
  function saveToFile() {
    const doc = serialize();
    openFsDialog({
      mode: 'save', kind: 'file',
      filters: [{ label: t('composer.layout_filter'), exts: ['.json'] }],
      title: t('composer.save_file_title'),
      defaultName: 'composer-layout.json',
      onPick: async (p) => {
        if (!p) { return; }
        try {
          const r = await api.fsWrite(
            withExt(p, '.json'), JSON.stringify(doc, null, 2));
          toast(t('composer.saved_pl_path', { path: r.path }), 'ok');
        } catch (e) {
          toast(t('composer.save_pl_failed', { msg: e.message }), 'error');
        }
      },
    });
  }
  // Save a pipeline together with THIS arrangement: pick the pipeline, then
  // a Save-As dialog makes it explicit that the pipeline file is being
  // written (with the arrangement bundled in as aux.composer). It travels
  // with the pipeline file -- written on save, recovered on load.
  async function saveWithPipeline(anchor) {
    const list = await pipelines();
    if (!list.length) { toast(t('composer.no_pipeline'), 'error'); return; }
    openMenu(anchor.left, anchor.bottom + 4, list.map((id) => ({
      label: id, onClick: async () => {
        state.pipeline = id; updatePlLabel();
        const doc = serialize(); doc.pipeline = id;
        persist();
        // Seed the dialog with the pipeline's current file, if any.
        let seed = { dir: '', name: '' };
        try {
          const d = await api.getPipeline(id);
          seed = splitPath((d && d.storage_path) || '');
        } catch (e) { /* fall back to defaults */ }
        openFsDialog({
          mode: 'save', kind: 'file',
          filters: [{ label: t('pl.vpipeline_filter'),
            exts: ['.vpipeline'] }],
          title: t('composer.save_pl_title', { id }),
          startDir: seed.dir,
          defaultName: seed.name || (id + '.vpipeline'),
          onPick: async (p) => {
            if (!p) { return; }
            try {
              const r = await api.savePipeline(
                id, withExt(p, '.vpipeline'), { composer: doc });
              toast(t('composer.saved_pl_path',
                { path: r.storage_path }), 'ok');
            } catch (e) {
              toast(t('composer.save_pl_failed',
                { msg: e.message }), 'error');
            }
          },
        });
      } })));
  }
  // Read a standalone layout JSON from the SERVER (a browse into the
  // sandbox), like the pipeline load -- not a client-side file picker.
  function loadFromFile() {
    openFsDialog({
      mode: 'open', kind: 'file',
      filters: [{ label: t('composer.layout_filter'), exts: ['.json'] }],
      title: t('composer.load_file_title'),
      onPick: async (p) => {
        if (!p) { return; }
        try {
          const r = await api.fsText(p, 4 * 1024 * 1024);
          if (r.truncated) { throw new Error(t('composer.file_too_large')); }
          deserialize(JSON.parse(r.text));
          toast(t('composer.loaded'), 'ok');
        } catch (e) {
          toast(t('composer.load_failed', { msg: e.message }), 'error');
        }
      },
    });
  }
  // Restore a pipeline's arrangement from the BACKEND (aux.composer). Lists
  // the loaded pipelines; a pick with no saved arrangement just says so.
  async function loadForPipeline(anchor) {
    const list = await pipelines();
    if (!list.length) { toast(t('composer.no_pipeline'), 'error'); return; }
    openMenu(anchor.left, anchor.bottom + 4, list.map((id) => ({
      label: id, onClick: async () => {
        try {
          const d = await api.getPipeline(id);
          const doc = d && d.aux && d.aux.composer;
          if (!doc || typeof doc !== 'object') {
            toast(t('composer.no_saved_pl', { id }), 'error');
            return;
          }
          deserialize(doc);
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
