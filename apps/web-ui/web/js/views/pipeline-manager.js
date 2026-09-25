// Pipeline Manager view: the pipeline list beside the reusable editor,
// which is the stage graph plus the configuration panel. The panel docks
// BELOW the graph by default and can be moved to its right; either way a
// divider sets the split. See the layout preferences below.

import { el, clear, append, toast, openModal, openErrorModal, openMenu, kbd }
  from '../dom.js';
import { makeIcon } from '../icons.js';
import { api, MODEL_REGISTRY_DB } from '../api.js';
import { modelMatches, modalitiesPresent, compatibleModels }
  from '../model-filter.js';
import { renderGraph, applyBufferStats, shortType, worldToPin }
  from '../graph.js';
import { topoOrder, assignLanes, laneArt, laneX, gutterWidth }
  from '../lane-graph.js';
import { openConnectEditor } from '../connect-editor.js';
import { typesCompatible, tagsCompatible } from '../port-compat.js';
import { t, tOr } from '../i18n.js';
import { openFsDialog, filterForCategory, splitPath } from '../fs-dialog.js';

// --- editor layout preferences --------------------------------------
// Where the configuration pane docks, whether field documentation shows,
// and whether the graph pane is pinned to the lane list. Per USER rather
// than per pipeline: they describe how someone likes to work, not
// anything about the graph, so they are not part of a pipeline file and
// not part of a composer panel's config -- the same reasoning the theme
// and the file dialog's preview toggle already follow. Every read and
// write is guarded: storage is blocked in a private window, and a
// layout toggle is not worth a thrown exception.
const LS_DOCK  = 'vpipe_pe_cfg_dock';    // 'bottom' (default) | 'right'
const LS_DOCS  = 'vpipe_pe_cfg_docs';    // '1' (default) | '0'
const LS_LANES = 'vpipe_pe_lane_view';   // '1' | '0' (default)
// WHAT A BARE WHEEL DOES ON THE CANVAS. Nothing in a wheel event says
// whether a trackpad or a mouse sent it, and the right default differs
// between them -- a two-finger swipe should move the canvas, a mouse's
// one wheel should zoom it -- so it is a preference, not a heuristic.
// Default 'pan', which is the trackpad this is mostly used from.
const LS_WHEEL = 'vpipe_pe_wheel';       // 'pan' (default) | 'zoom'
// THE SELECTOR : EDITOR SPLIT, as the selector's fraction of the view.
// A fraction, like the stage:config split, so a window resize keeps the
// proportion someone set rather than stranding a pixel width. Only the
// full manager has a selector; the standalone editor ignores this.
const LS_SELW  = 'vpipe_pm_sel_width';   // '0.19' by default (the CSS)
// WHICH PIPELINE THE SELECTOR LAST HAD. Every view switch re-mounts this
// view (app.js calls mount() again), so the selection cannot live in the
// closure: leaving the Pipelines view for User I/O and coming back built
// a fresh state with no selection and fell to the first pipeline, losing
// whichever one was being edited.
//
// Advisory, like the rest: a pipeline can be unloaded or renamed from
// anywhere, so a miss just means "no opinion" and the first pipeline is
// used. Kept separate from the phone shell's own last-pipeline key
// (phone-recent.js): that one steers which pipeline the phone's stage
// panels open on, and a desktop selection should not quietly re-point
// them.
const LS_PIPE  = 'vpipe_pe_pipeline';    // '' when nothing is selected

function prefGet(key, dflt) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? dflt : v;
  } catch (e) { return dflt; }
}
function prefSet(key, v) {
  try { localStorage.setItem(key, v); } catch (e) { /* storage blocked */ }
}

export function mountPipelineManager(container) {
  return mountEditor(container, { showSelector: true, showControls: false });
}

// The reusable per-pipeline editor -- the stage canvas + configuration
// panel for ONE pipeline, without the pipeline-list selector. Options:
//   pipelineId  the linked pipeline id (required; designated at creation).
//   split       stage:config width ratio as the stage FRACTION (default
//               2/3 == 2:1); onSplit(fraction) fires when the user drags
//               the divider, so a host (composer) can persist it.
// Shows run / pause / stop on the canvas top-left (the selector, which
// carries per-row controls, is absent here).
// `onRebind(newId)` fires when the bound pipeline is unloaded and the
// operator picks another from the rebind menu (see renderRebindPane), so
// the host (composer) can re-point the panel config + re-title.
export function mountPipelineEditor(container, opts = {}) {
  return mountEditor(container, {
    showSelector: false, showControls: true,
    pipelineId: opts.pipelineId || null,
    split: opts.split, onSplit: opts.onSplit,
    onRebind: opts.onRebind,
  });
}

function mountEditor(container, opts = {}) {
  const showSelector = opts.showSelector !== false;
  const showControls = !!opts.showControls;
  const clampN = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));
  // Stage:config split as the stage fraction (2:1 default). Adjustable via
  // the divider; onSplit lets a host persist it as a view config.
  let split = clampN(
    Number.isFinite(opts.split) ? opts.split : 2 / 3, 0.2, 0.85);
  // BOTTOM by default. A pipeline reads left to right, so width is what
  // the canvas is always short of and height is what it has spare -- a
  // configuration pane beside it takes the scarce axis to show a column
  // of fields that does not need it. Docked below, the same form spreads
  // into however many columns fit (see the .cfg grid) and the canvas
  // keeps the full width.
  let cfgDock = prefGet(LS_DOCK, 'bottom') === 'right' ? 'right' : 'bottom';
  let showDocs = prefGet(LS_DOCS, '1') !== '0';
  // The lane list is normally a FALLBACK the pane drops to when it is too
  // narrow to draw a graph. This pins it on at any width: it says more
  // about a long pipeline per vertical pixel than the canvas does, and
  // someone reading topology rather than editing it may simply prefer it.
  let forceLanes = prefGet(LS_LANES, '0') === '1';
  let wheelMode = prefGet(LS_WHEEL, 'pan') === 'zoom' ? 'zoom' : 'pan';
  // The CSS default (19%) is the fallback, so the two cannot disagree
  // about where an unconfigured selector starts.
  let selSplit = clampN(
    parseFloat(prefGet(LS_SELW, '')) || 0.19, 0.08, 0.5);
  // ONE step of undo for a configuration edit, which is the mistake this
  // is for: a value typed over, a checkbox flipped, a field cleared --
  // noticed immediately. {sid, before, after, undone}; `undone` says
  // which way the pair currently points, so the same slot serves Redo
  // rather than needing a second stack.
  //
  // The PUT replaces a stage's config outright (it is not a merge), so
  // restoring `before` really does remove a key the edit added.
  let cfgUndo = null;
  // What the form on screen is known to hold, so the NEXT apply can pair
  // itself against it. A render seeds it from the schema; each apply
  // moves it on -- which is what keeps repeated auto-applies (one per
  // blur) collapsing into a single step rather than pairing every edit
  // against the state the form was first built from.
  let cfgLast = null;    // {sid, cfg}
  // The rendered pair of buttons, held so an apply can re-enable them
  // WITHOUT a re-render. The auto-apply-on-blur path does not rebuild the
  // form on purpose -- that would take the caret out of the field the
  // user just tabbed into -- so without this the first edit of a session
  // leaves Undo greyed out until something else redraws the header.
  let undoBtns = null;   // {undo, redo}
  // Which configuration render is the current one. renderConfig empties
  // the header BEFORE awaiting the stage's config and fills it AFTER, so
  // two overlapping renders interleave as clear, clear, fill, fill --
  // and the header ends up holding two sets of buttons, then three,
  // crowding the layout toggles out of the row. Anything that re-renders
  // in bursts reaches this: a dock flip, or a panel dragged through the
  // narrow crossover. Only the newest render may write.
  let cfgSeq = 0;
  const state = {
    pipelines: [],        // summaries
    stageTypes: [],
    selectedId: null,     // selected pipeline id
    detail: null,         // {id,state,storage_path,graph}
    selectedStage: null,  // selected stage id
    // Composer edge-editing state. `pending` is the armed output port
    // ({from, from_port}) after an oport click, awaiting an input click
    // to complete the wire; `selectedEdge` is the edgeKey() of the
    // highlighted edge (Delete removes it). Both are stopped-only.
    pending: null,
    selectedEdge: null,
    graphView: {},        // pan/zoom {k,cx,cy}; reset (new {}) per pipeline
    // Previous layout's node positions (id -> {x,y,...}), fed back into
    // the next render so re-layout is placement-aware (minimal movement)
    // and can animate the reflow. Reset (null) per pipeline so a fresh
    // pipeline lays out clean with no phantom glide.
    graphLayout: null,
    // User-placed drop pins (stageId -> {col, y}): where the operator
    // dropped a new stage from the toolbox. Honoured by the layout as a
    // column floor + row seed so the chosen grid cell survives. Cleared
    // per pipeline.
    graphPins: new Map(),
    graphContainer: null, // live element from renderGraph (overlay target)
    // Which rendering the graph pane currently holds: the canvas, or the
    // linearized lane list it falls back to when the pane is too narrow
    // to draw one. The resize watcher compares against this.
    renderedNarrow: false,
    // Whether the editor is drawn as ONE COLUMN (no configuration pane,
    // no divider). Narrower than `renderedNarrow`: a wide editor whose
    // divider was dragged hard left shows the lane list but keeps its
    // pane, so the two are not the same question.
    oneColumn: false,
    // One-column layout only: the lane row expanded to show its stage's
    // configuration inline. There is no configuration PANE at this width,
    // so the form lives under its row -- the phone shell's arrangement,
    // for the same reason.
    laneOpen: null,
    // Buffer-utilization overlay: poll the running pipeline's per-edge
    // backlog/capacity and annotate the graph. Enabled by default so a
    // freshly-running pipeline shows depth immediately; the interval is
    // operator-adjustable (seconds).
    bufEnabled: true,
    bufIntervalMs: 2000,
    // In-flight lifecycle ops keyed by pipeline id. Set while the
    // POST /api/pipelines/{id}/{op} is awaiting -- big pipelines can
    // take many seconds to drain on stop, so the UI shows a "Stopping
    // …" badge and disables every lifecycle button until the
    // backend's response confirms the new state.
    inflight: new Map(),  // id -> { op, label }
    // Standalone editor only: the bound pipeline was unloaded (removed
    // from the manager or another panel) and no longer exists. The graph
    // pane then shows a rebind menu (renderRebindPane) instead of a stale
    // graph; `rebindKey` guards against re-rendering it when unchanged.
    missing: false,
    rebindChoices: [],    // loaded pipeline ids to rebind to
    rebindKey: '',
  };
  // Standalone editor: the pipeline is fixed (no selector to change it),
  // but it CAN be re-pointed via the rebind menu, so track it mutably.
  let boundId = opts.pipelineId || null;
  if (!showSelector) { state.selectedId = boundId; }

  // Labels shown in the pipeline row's state pill while a lifecycle
  // op is in-flight. "Stopping" specifically addresses the user's
  // request -- big pipelines take time to drain and the operator
  // needs to know the click registered.
  const INFLIGHT_LABEL = {
    launch: t('pl.starting'),
    pause:  t('pl.pausing'),
    stop:   t('pl.stopping'),
  };

  // --- skeleton -----------------------------------------------------
  const listBody  = el('div', { class: 'pane-body' });
  const graphHead = el('div', { class: 'pane-head' });
  const graphBody = el('div', { class: 'pane-body graph-wrap' });
  const cfgBody   = el('div', { class: 'pane-body' });

  const btnCreate = iconBtn('plus',  t('common.create'), () => onCreate(), 'N');
  const btnLoad   = iconBtn('load',  t('common.load'),   () => onLoad(),   'O');
  const btnRename = iconBtn('edit',  t('common.rename'),
                            () => onRenamePipeline(), 'R');
  const btnSave   = iconBtn('save',  t('common.save'),   () => onSave(),   'S');
  const btnUnload = iconBtn('trash', t('common.unload'), () => onUnload(), 'U');

  const listPane = el('div', { class: 'pane' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('nav.pipelines'))),
    el('div', { class: 'toolbar' },
      btnCreate, btnLoad, btnRename, btnSave, btnUnload),
    listBody);


  // Buffer-fill overlay control. Styled to match the graph view-control
  // buttons and appended into that cluster (bottom-left) per render: a
  // toggle button + a compact poll-interval input.
  const bufBtn = el('button', {
    class: 'graph-ctl buf-ctl',
    title: t('pl.buf_toggle'),
    onclick: () => setBufEnabled(!state.bufEnabled),
  }, t('pl.buf'));
  const bufInt = el('input', {
    type: 'number', min: '0.25', step: '0.25', class: 'graph-ctl buf-interval',
    title: t('pl.buf_interval'),
  });
  bufInt.value = String(state.bufIntervalMs / 1000);
  bufInt.addEventListener('change', () => {
    const s = parseFloat(bufInt.value);
    if (Number.isFinite(s) && s >= 0.25) {
      state.bufIntervalMs = Math.round(s * 1000);
      scheduleBufferPoll();
    } else {
      bufInt.value = String(state.bufIntervalMs / 1000);
    }
  });
  // Persistent group element re-appended into the (re-rendered) graph
  // controls cluster each render (see renderGraphPane).
  const bufGroup = el('div', { class: 'buf-group' }, bufBtn, bufInt);
  bufBtn.classList.toggle('active', state.bufEnabled);
  bufInt.disabled = !state.bufEnabled;
  function setBufEnabled(v) {
    state.bufEnabled = v;
    bufBtn.classList.toggle('active', v);
    bufInt.disabled = !v;
    if (!v) {
      if (state.graphContainer) { applyBufferStats(state.graphContainer, []); }
    } else {
      pollBuffersNow();
    }
  }

  // Live stage count, right-aligned in the Stages title bar. Updated by
  // renderGraphPane whenever the graph changes.
  const stagesCount = el('span', { class: 'stages-count' });
  // The NARROW view's way in to the palette.
  //
  // Below the canvas crossover the pane shows the linearized lane list,
  // and a docked toolbox there is the wrong shape twice over: it spends
  // a third of an already-cramped pane on a permanent panel, and what it
  // offers -- drag a chip onto the canvas -- cannot be done, because
  // there is no canvas to drop onto. So the panel is hidden and this
  // takes its place: one button, and a picker that opens on demand,
  // filters, and adds on a single click.
  //
  // It is the SAME palette (see stageSections), so the two views cannot
  // come to disagree about what can be added.
  //
  // Declared BEFORE the header that mounts it: `const` is in its
  // temporal dead zone until this line runs, so appending it above
  // would throw on mount rather than merely read oddly.
  const addStageBtn = el('button', {
    class: 'graph-add-stage', type: 'button',
    title: tOr('pl.add_stage', 'Add stage'),
    onclick: (e) => { e.stopPropagation(); openStagePicker(); },
  }, '+');

  // The documentation toggle, repeated in the GRAPH header.
  //
  // One column hides the configuration pane and the three layout toggles
  // in its header with it, so turning documentation off while reading a
  // narrow panel meant widening it, toggling, and narrowing it again.
  //
  // Only this one is repeated. With no pane there is nothing to dock,
  // and the lane list is what this width draws whatever the lane toggle
  // says -- so the other two would be buttons that do nothing here.
  //
  // Built with el() rather than the toolBtn helper below: `const` is in
  // its temporal dead zone until its own line runs, and this is needed
  // before the header is assembled.
  const docsBtnHead = el('button', {
    class: 'graph-head-tool', type: 'button',
    onclick: (e) => { e.stopPropagation(); toggleDocs(); },
  }, 'ⓘ');

  graphHead.append(
    el('span', { class: 'title' }, t('pl.stages')),
    el('span', { class: 'grow' }),
    stagesCount,
    docsBtnHead,
    addStageBtn);

  function openStagePicker() {
    if (!canEdit()) { return; }
    const body = el('div', { class: 'sp-body' });
    const filter = el('input', {
      type: 'search', class: 'sp-filter', placeholder: t('pl.filter_ph'),
      oninput: () => { clear(body); fill(); },
    });
    const pop = el('div', { class: 'stage-picker' }, filter, body);
    const close = () => {
      pop.remove();
      document.removeEventListener('keydown', onKey, true);
      document.removeEventListener('pointerdown', onAway, true);
      window.removeEventListener('resize', close);
    };
    const fill = () => renderSections(body, filter.value || '', {
      onPick: (type) => { close(); promptCreateStage(type); },
    });
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
    };
    const onAway = (e) => {
      if (!pop.contains(e.target) && e.target !== addStageBtn) { close(); }
    };
    fill();
    // Anchored under the button and clamped to the viewport, so it does
    // not hang off the right edge of a narrow pane -- which is the only
    // width this thing is ever opened at.
    const r = addStageBtn.getBoundingClientRect();
    (document.getElementById('modal-root') || document.body).append(pop);
    const w = pop.offsetWidth || 260;
    pop.style.left = Math.max(6, Math.min(r.right - w,
      window.innerWidth - w - 6)) + 'px';
    pop.style.top = (r.bottom + 4) + 'px';
    document.addEventListener('keydown', onKey, true);
    document.addEventListener('pointerdown', onAway, true);
    window.addEventListener('resize', close);
    setTimeout(() => filter.focus(), 0);
  }

  // Toolbox: a filterable palette of every registered stage type.
  // Drag a chip onto the canvas (or double-click it) to add a stage.
  const toolboxFilter = el('input', {
    type: 'search', class: 'toolbox-filter', placeholder: t('pl.filter_ph'),
    oninput: () => renderToolbox(),
  });
  const toolboxBody = el('div', { class: 'toolbox-body' });
  // Collapse/expand toggle: an extruding tab anchored to the toolbox's
  // right edge (it's a CHILD of the toolbox, so it rides the real edge
  // whatever the toolbox's width), vertically centered and rounded on
  // its outer side (a Chrome tab rotated 90°). ◀ collapses the toolbox
  // to zero width (its content hidden) while this tab stays at the
  // canvas's left edge showing ▶ to re-extend. No re-render -- the
  // toolbox element persists across graph re-renders.
  const tbToggle = el('button', {
    class: 'tb-toggle', title: t('pl.hide_toolbox'),
    onclick: () => setToolboxVisible(toolbox.classList.contains('collapsed')),
  }, '◀');
  function setToolboxVisible(v) {
    toolbox.classList.toggle('collapsed', !v);
    tbToggle.textContent = v ? '◀' : '▶';
    tbToggle.title = v ? t('pl.hide_toolbox') : t('pl.show_toolbox');
  }
  const toolbox = el('div', { class: 'toolbox' },
    el('div', { class: 'toolbox-head' },
      el('span', { class: 'title' }, t('pl.toolbox'))),
    toolboxFilter, toolboxBody, tbToggle);

  // graphBody is the drop target; the toolbox sits to its left.
  const graphSplit = el('div', { class: 'graph-split' }, toolbox, graphBody);
  const graphPane = el('div', { class: 'pane' }, graphHead, graphSplit);

  // Drag-create: accept a stage-type drop anywhere on the canvas. The
  // drop point is mapped to a grid cell and pinned, so a new (edge-less)
  // stage lands where the operator put it -- handy for pre-positioning it
  // next to what it will connect to.
  graphBody.addEventListener('dragover', (e) => {
    if (!canEdit()) { return; }
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    graphBody.classList.add('drop-target');
  });
  graphBody.addEventListener('dragleave', (e) => {
    if (e.target === graphBody) { graphBody.classList.remove('drop-target'); }
  });
  graphBody.addEventListener('drop', (e) => {
    e.preventDefault();
    graphBody.classList.remove('drop-target');
    const dropType = e.dataTransfer.getData('text/vpipe-stage-type')
           || e.dataTransfer.getData('text/plain');
    if (!canEdit()) {
      toast(t('pl.select_stopped'), 'error');
      return;
    }
    if (dropType && state.stageTypes.some((s) => s.type === dropType)) {
      // Map the drop point to a world coordinate (null when the canvas is
      // empty / has no graph yet -> auto-place). worldToPin snaps + caps it.
      let world = null;
      const gc = state.graphContainer;
      if (gc && gc.clientToWorld && state.graphLayout) {
        world = gc.clientToWorld({ clientX: e.clientX, clientY: e.clientY });
      }
      promptCreateStage(dropType, world);
    }
  });

  // Apply / Remove live in the config pane HEADER (not the scrolling body)
  // so they stay reachable no matter where a long form is scrolled.
  // renderConfig fills this per stage; it is empty when none is selected.
  const cfgActions = el('span', { class: 'pane-head-actions' });
  // The three LAYOUT toggles, grouped at the far right of the same
  // header: where this pane sits, whether its fields carry their
  // documentation, and whether the graph pane draws a canvas or the lane
  // list. They live here rather than on the graph header because all
  // three are about the shape of the editor, and one group in one corner
  // is easier to find again than three buttons in three places.
  //
  // Titles and pressed state are filled by the apply* functions below, so
  // the button and the state it reports cannot be spelled twice.
  const toolBtn = (glyph, onclick) => el('button', {
    class: 'pe-tool', type: 'button', onclick,
  }, glyph);
  const dockBtn = toolBtn('', () => {
    cfgDock = cfgDock === 'bottom' ? 'right' : 'bottom';
    prefSet(LS_DOCK, cfgDock);
    applyDock();
    // The dock decides how much WIDTH the configuration costs, which is
    // the input to the one-column crossover -- so the layout has to be
    // re-decided here rather than waiting for the next resize.
    renderGraphPane();
    if (!state.oneColumn) { renderConfig(); }
  });
  // One flip, two buttons: this one and the graph header's copy for the
  // one-column layout. A handler each would be two spellings of a single
  // decision, which is how they come to disagree.
  function toggleDocs() {
    showDocs = !showDocs;
    prefSet(LS_DOCS, showDocs ? '1' : '0');
    applyDocs();
  }
  const docsBtn = toolBtn('ⓘ', () => toggleDocs());
  const laneBtn = toolBtn('☰', () => {
    forceLanes = !forceLanes;
    prefSet(LS_LANES, forceLanes ? '1' : '0');
    applyLanes();
  });
  const cfgTools = el('span', { class: 'pe-cfg-tools' },
    dockBtn, docsBtn, laneBtn);
  const cfgPane = el('div', { class: 'pane' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('pl.configuration')),
      cfgActions, cfgTools),
    cfgBody);

  // Run / pause / stop overlay pinned to the canvas top-left (standalone
  // editor only; the selector's per-row controls cover the full view).
  // Contents are rebuilt by renderControls() from the pipeline state.
  const canvasControls = showControls
    ? el('div', { class: 'pe-canvas-ctl' }) : null;

  // Stage editor : config editor split, adjustable via a divider. The
  // stage pane's flex-grow is `split`, the config pane's is `1 - split`.
  graphPane.classList.add('pe-graph');
  cfgPane.classList.add('pe-config');
  const divider = el('div', { class: 'pe-divider', title: t('pl.stages') });
  const editorArea = el('div', { class: 'pe-editor' },
    graphPane, divider, cfgPane);
  function applySplit() {
    graphPane.style.flexBasis = '0';
    cfgPane.style.flexBasis = '0';
    graphPane.style.flexGrow = String(split);
    cfgPane.style.flexGrow = String(1 - split);
  }
  // `split` is the GRAPH's share of whichever axis the dock chose, so one
  // fraction serves both orientations and a dock flip keeps the
  // proportion the operator set rather than resetting it.
  function applyDock() {
    const bottom = cfgDock === 'bottom';
    editorArea.classList.toggle('dock-bottom', bottom);
    dockBtn.textContent = bottom ? '⬓' : '◨';
    dockBtn.title = bottom ? t('pl.dock_right') : t('pl.dock_bottom');
    divider.title = bottom ? t('pl.resize_height') : t('pl.resize_width');
    applySplit();
  }
  function applyDocs() {
    // On the EDITOR, not the pane: the one-column layout renders the same
    // form inline under a lane row, and a reader who turned the
    // documentation off meant everywhere.
    editorArea.classList.toggle('no-docs', !showDocs);
    // Both copies: the pane header's, and the graph header's for the
    // one-column layout. They are one setting, so they read the same.
    for (const b of [docsBtn, docsBtnHead]) {
      b.title = showDocs ? t('pl.docs_hide') : t('pl.docs_show');
      b.setAttribute('aria-pressed', showDocs ? 'true' : 'false');
    }
    // The tooltip stands in for the documentation, so it exists exactly
    // while the lines do not -- on whatever form is on screen already.
    for (const k of editorArea.querySelectorAll('.field .key')) {
      const d = k.dataset.doc || '';
      if (!showDocs && d) { k.title = d; } else { k.removeAttribute('title'); }
    }
  }
  // The button's own state, separate from acting on it: at mount there is
  // no graph to re-render yet (renderGraphPane runs once the pipeline
  // detail arrives), and spelling the pressed state twice is how the
  // button and the layout come to disagree.
  function applyLaneBtn() {
    laneBtn.title = forceLanes ? t('pl.lane_view_off') : t('pl.lane_view_on');
    laneBtn.setAttribute('aria-pressed', forceLanes ? 'true' : 'false');
  }
  function applyLanes() {
    applyLaneBtn();
    // Returning to the canvas: its pan/zoom was recorded against a
    // viewport the lane list has since resized, so refit rather than
    // restore (the resize watcher does the same on its own crossover).
    if (!forceLanes) { state.graphView = {}; }
    renderGraphPane();
  }
  applySplit();
  applyDock();
  applyDocs();
  applyLaneBtn();
  divider.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    const r = editorArea.getBoundingClientRect();
    // The divider resizes the axis the dock laid the panes out on.
    const bottom = cfgDock === 'bottom';
    const mv = (e) => {
      split = clampN(bottom ? (e.clientY - r.top) / r.height
                            : (e.clientX - r.left) / r.width, 0.2, 0.85);
      applySplit();
    };
    const up = () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
      if (typeof opts.onSplit === 'function') {
        opts.onSplit(+split.toFixed(4));
      }
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  });

  // The selector : editor divider, the same affordance as the editor's
  // own and deliberately the same class, because it is the same thing:
  // a pipeline list is narrow until the names are long, and a canvas is
  // short of width always. Only the full manager has one -- the
  // standalone editor has no selector to resize against.
  const selDivider = showSelector
    ? el('div', { class: 'pe-divider pm-divider',
                  title: t('pl.resize_selector') })
    : null;
  function applySelSplit() {
    // flex-basis only: .pm > .pane keeps grow/shrink at 0 and the CSS
    // min-width still floors it, so a hard drag left cannot squeeze the
    // list into an unreadable sliver.
    listPane.style.flexBasis = (selSplit * 100).toFixed(3) + '%';
  }
  if (selDivider) { applySelSplit(); }

  const pmRoot = showSelector
    ? el('div', { class: 'pm' }, listPane, selDivider, editorArea)
    : el('div', { class: 'pm no-sel' }, editorArea);
  clear(container).append(pmRoot);

  if (selDivider) {
    selDivider.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      const r = pmRoot.getBoundingClientRect();
      const mv = (e) => {
        selSplit = clampN((e.clientX - r.left) / r.width, 0.08, 0.5);
        applySelSplit();
      };
      // Persisted on RELEASE, not per frame: a drag fires dozens of
      // moves and localStorage is synchronous.
      const up = () => {
        window.removeEventListener('pointermove', mv);
        window.removeEventListener('pointerup', up);
        prefSet(LS_SELW, selSplit.toFixed(4));
      };
      window.addEventListener('pointermove', mv);
      window.addEventListener('pointerup', up);
    });
  }

  // Watch the graph pane for the canvas/list crossover. Every cause of a
  // width change lands here -- the window, the stage:config divider, the
  // toolbox collapsing -- so there is one trigger rather than a hook per
  // cause.
  //
  // It re-renders only when the MODE actually flips. A divider drag
  // fires this on almost every frame, and rebuilding a canvas (or a
  // list) per frame would make the drag itself stutter; comparing
  // against the last decision makes all but one of those a no-op.
  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      if (!document.body.contains(pmRoot)) { return; }
      const one = editorNarrow();
      const now = laneMode(one);
      // Compared against WHAT IS ON SCREEN (recorded by renderGraphPane)
      // rather than against the last observation. The two differ at
      // mount: the pane is measured at 0 before layout, which isNarrow()
      // reads as wide, so a first render can legitimately disagree with
      // the first observation and must be allowed to correct itself --
      // while a first observation that agrees must not cost a rebuild.
      if (now === state.renderedNarrow && one === state.oneColumn) { return; }
      // The canvas keeps its pan/zoom in state.graphView, which a trip
      // through the list would leave pointing at a viewport of a
      // different width. Refit on the way back to it.
      if (!now) { state.graphView = {}; }
      // Going the other way, carry the selection into the expanded row:
      // the configuration was on screen a moment ago, and it should not
      // vanish merely because the editor got narrower. This is also what
      // keeps laneOpen and selectedStage naming the same stage, which is
      // the invariant the inline form is built on.
      if (one) { state.laneOpen = state.selectedStage; }
      renderGraphPane();
      // Coming back to two panes, the configuration PANE reappears holding
      // whatever it had before the one-column trip -- which is stale, or
      // nothing at all, because that layout rendered the form inline
      // instead. Fill it from the current selection.
      if (!one) { renderConfig(); }
    });
    // BOTH: the editor's width decides one-column, the graph body's decides
    // canvas-vs-list, and a resize can move either without the other.
    ro.observe(graphBody);
    ro.observe(editorArea);
  }
  // The configuration's own width, which the window, the divider and the
  // dock button all move. A SECOND observer because the one above
  // returns early whenever the canvas/list decision is unchanged -- which
  // is most of a divider drag, and exactly when the columns must reflow.
  if (typeof ResizeObserver !== 'undefined') {
    const cro = new ResizeObserver(() => layoutCfgColumns());
    cro.observe(cfgBody);
    cro.observe(editorArea);
  }

  function iconBtn(icon, label, onclick, key) {
    return el('button', {
      class: 'btn', onclick,
      title: label + (key ? '  (' + key + ')' : ''),
    },
      makeIcon(icon, 'sm'), el('span', {}, label),
      key ? kbd(key) : null);
  }

  // --- keyboard shortcuts ------------------------------------------
  // Active while this view is mounted; self-removes once its root
  // leaves the DOM (the shell has no unmount hook). Ignored while
  // typing in a field, while a modal is open, or with a modifier held.
  function onShortcut(e) {
    if (!document.body.contains(pmRoot)) {
      document.removeEventListener('keydown', onShortcut);
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    if (document.querySelector('.modal-back')) { return; }  // modal owns keys
    const t = e.target;
    const tag = (t && t.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
        || (t && t.isContentEditable)) { return; }

    if (e.key === 'Escape') {
      if (state.pending || state.selectedEdge) {
        e.preventDefault();
        state.pending = null;
        state.selectedEdge = null;
        renderGraphPane();
      }
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      // A selected edge takes priority over a selected stage.
      if (state.selectedEdge && canEdit()) {
        e.preventDefault();
        const { to, to_port } = parseEdgeKey(state.selectedEdge);
        doDisconnect(to, to_port);
      } else if (canEdit() && state.selectedStage) {
        e.preventDefault(); onRemoveStage();
      }
      return;
    }
    // Zoom the canvas. `-` and `=` are the UNSHIFTED pair, so neither
    // needs a second finger; they drive the same step as the − / +
    // buttons in the view-control cluster.
    //
    // ABOVE the showSelector return on purpose: the standalone editor
    // (the composer's canvas) is the view with the most graph on screen
    // and the least reason to reach for the mouse. The text-field guard
    // is the one at the top of this function, which already covers
    // INPUT / TEXTAREA / SELECT / contentEditable -- so typing a minus
    // into a config field still types a minus.
    if (e.key === '-' || e.key === '=') {
      const gc = state.graphContainer;
      if (gc && gc.zoomBy) {
        e.preventDefault();
        const step = gc.zoomStep || 1.2;
        gc.zoomBy(e.key === '=' ? step : 1 / step);
      }
      return;
    }

    // Pipeline-level shortcuts (create / load / save / unload) belong to
    // the selector; skip them in the standalone editor (no such buttons).
    if (!showSelector) { return; }
    const fire = (btn, fn) => {
      if (!btn.disabled) { e.preventDefault(); fn(); }
    };
    switch (e.key.toLowerCase()) {
      case 'n': fire(btnCreate, onCreate); break;
      case 'o': fire(btnLoad,   onLoad);   break;
      case 's': fire(btnSave,   onSave);   break;
      case 'u': fire(btnUnload, onUnload); break;
      default: break;
    }
  }
  document.addEventListener('keydown', onShortcut);

  // --- data flow ----------------------------------------------------
  async function refreshList(keepSel = true) {
    // Standalone editor: the pipeline is fixed; just refresh its detail
    // (state + graph) and the canvas controls. No list to enumerate --
    // but we DO fetch it once to confirm the bound pipeline still exists.
    // If it was unloaded, clear the editor and show the rebind menu.
    if (!showSelector) {
      state.selectedId = boundId;
      if (!state.selectedId) {
        state.missing = false;
        renderGraphPane(); renderConfig(); renderList();
        return;
      }
      let list = null;
      try { list = await api.listPipelines(); }
      catch (e) { /* transient -- fall through to loadDetail */ }
      if (list && !list.find((p) => p.id === state.selectedId)) {
        enterRebind(list);
        renderList();
        return;
      }
      state.missing = false;
      await loadDetail(state.selectedId);
      renderList();   // updates the canvas run/pause/stop controls
      return;
    }
    try {
      state.pipelines = await api.listPipelines();
    } catch (e) { toast(t('pl.list_failed', { msg: e.message }), 'error');
      return; }
    if (!keepSel || !state.pipelines.find((p) => p.id === state.selectedId)) {
      // No selection to keep -- a fresh mount, or the selected pipeline
      // is gone. Prefer the one this operator last chose, and fall back
      // to the first when it is no longer loaded: the memory is a
      // preference, never a reason to show nothing.
      const want = prefGet(LS_PIPE, '');
      const remembered = !!want && state.pipelines.some((p) => p.id === want);
      state.selectedId = remembered
          ? want
          : (state.pipelines[0] ? state.pipelines[0].id : null);
      state.detail = null;
      state.selectedStage = null;
      state.laneOpen = null;      // fold the inline form with its stage
      state.graphView = {};   // refit when the selected pipeline changes
      state.graphLayout = null;   // lay out the new pipeline clean (no glide)
      state.graphPins = new Map();
    }
    rememberSelection();
    renderList();
    if (state.selectedId) { await loadDetail(state.selectedId); }
    else { renderGraphPane(); renderConfig(); }
  }

  // The selection, written down wherever it SETTLES rather than at each
  // of the half-dozen places that move it (create, load, rename, rebind,
  // a click in the list): those all end in refreshList or selectPipeline.
  // The standalone editor is bound to its pipeline rather than choosing
  // one, so it has no opinion to record.
  function rememberSelection() {
    if (!showSelector) { return; }
    prefSet(LS_PIPE, state.selectedId || '');
  }

  async function loadDetail(id) {
    try {
      state.detail = await api.getPipeline(id);
    } catch (e) { toast(t('pl.detail_failed', { msg: e.message }), 'error');
      return; }
    if (state.selectedStage &&
        !state.detail.graph.nodes.find((n) => n.id === state.selectedStage)) {
      state.selectedStage = null;
    }
    renderGraphPane();
    renderConfig();
  }

  async function selectPipeline(id) {
    state.selectedId = id;
    rememberSelection();
    state.selectedStage = null;
    state.pending = null;
    state.selectedEdge = null;
    state.graphView = {};   // refit the graph for the newly-selected pipeline
    state.graphLayout = null;   // lay out the new pipeline clean (no glide)
    state.graphPins = new Map();
    renderList();
    await loadDetail(id);
  }

  // --- buffer-utilization overlay polling --------------------------
  // One self-healing loop runs while the view is mounted. Each tick
  // fetches the selected RUNNING pipeline's per-edge buffer stats and
  // patches the graph overlay IN PLACE (no re-render -> pan/zoom and
  // selection survive). It does nothing while stopped/paused or the
  // toggle is off, and dies once the view leaves the DOM. The overlay
  // is cleared naturally on stop/pause because those re-render the
  // graph from a fresh (empty-label) container.
  let bufTimer = null;

  function stopBufferPoll() {
    if (bufTimer !== null) { clearTimeout(bufTimer); bufTimer = null; }
  }
  function scheduleBufferPoll() {
    stopBufferPoll();
    if (!document.body.contains(pmRoot)) { return; }   // unmounted -> stop
    bufTimer = setTimeout(pollBuffers,
                          Math.max(250, state.bufIntervalMs || 2000));
  }
  // Should the poll loop hit the server this tick? True while a RUNNING
  // pipeline is selected and no lifecycle op is mid-flight for it --
  // independent of the overlay toggle, so an auto-stop (all stages
  // signalled done) is detected and reflected even with the overlay off.
  function pollActive() {
    return !!state.selectedId && !!state.detail
        && state.detail.state === 'running'
        && !state.inflight.has(state.selectedId);
  }
  async function pollBuffers() {
    bufTimer = null;
    if (!document.body.contains(pmRoot)) { return; }
    // Standalone editor: watch for an EXTERNAL unload of the bound
    // pipeline (removed from the manager or another panel) and flip to
    // the rebind menu; restore in place if it is reloaded under the same
    // id. Cheap GET; the loop already ticks while mounted.
    if (!showSelector && boundId) {
      let list = null;
      try { list = await api.listPipelines(); } catch (e) { /* transient */ }
      if (!document.body.contains(pmRoot)) { return; }
      if (list) {
        const present = list.some((p) => p.id === boundId);
        if (!present) {
          const key = list.map((p) => p.id).join('\n');
          if (!state.missing || state.rebindKey !== key) {
            enterRebind(list); renderList();
          }
          scheduleBufferPoll();
          return;
        }
        if (state.missing) {   // came back under the same id -> restore
          state.missing = false;
          await loadDetail(boundId);
          renderList();
        }
      }
    }
    if (pollActive()) {
      const id = state.selectedId;
      try {
        const r = await api.bufferStatus(id);
        // A switch/stop may have happened during the await -- re-check.
        if (pollActive() && state.selectedId === id) {
          const st = r && r.state;
          if (st && st !== 'running') {
            // The backend auto-stopped this pipeline (its stages all
            // signalled done). Reflect the new state across the list,
            // buttons, and graph (refreshList rebuilds the overlay).
            const stateLabel = {
              stopped: t('pl.state_stopped'),
              paused:  t('pl.state_paused'),
              running: t('pl.state_running'),
            };
            toast(id + ' ' + (stateLabel[st] || st), 'ok');
            await refreshList();
          } else if (state.bufEnabled && state.graphContainer) {
            applyBufferStats(state.graphContainer, (r && r.edges) || []);
          }
        }
      } catch (e) { /* transient (server busy / stopping); keep polling */ }
    }
    scheduleBufferPoll();
  }
  // Poll once immediately (on enable, or right after the graph was
  // rebuilt) instead of waiting a full interval. No-op when inactive.
  function pollBuffersNow() {
    if (!pollActive()) { return; }
    stopBufferPoll();
    pollBuffers();
  }

  // --- left pane ----------------------------------------------------
  function renderList() {
    if (canvasControls) { renderControls(); }
    if (!showSelector) { return; }     // standalone editor: no selector
    clear(listBody);
    const ul = el('ul', { class: 'pl-list' });
    for (const p of state.pipelines) {
      const running = p.state === 'running';
      const paused  = p.state === 'paused';
      const stopped = p.state === 'stopped';
      // An in-flight lifecycle op disables every action button on
      // this row -- the operator can't queue another op until the
      // backend confirms (otherwise a slow Stop followed by a quick
      // Start could race the materialiser).
      const busy = state.inflight.get(p.id) || null;
      const stateText  = busy ? busy.label + '…' : p.state;
      const stateClass = busy ? 'busy' : p.state;
      const row = el('li', {
        class: 'pl-item' + (p.id === state.selectedId ? ' selected' : ''),
        onclick: () => selectPipeline(p.id),
      },
        el('span', { class: 'pl-name', title: p.id }, p.id),
        el('span', { class: 'pl-state ' + stateClass,
          title: busy
            ? busy.label + ' ' + p.id + ' (' + t('pl.inflight_hint') + ')'
            : p.state }, stateText),
        el('span', { class: 'pl-actions' },
          actIcon('play',  t('common.start'), !busy && stopped,
                  (e) => lc(e, p.id, 'launch')),
          actIcon('pause', t('common.pause'), !busy && running,
                  (e) => lc(e, p.id, 'pause')),
          actIcon('stop',  t('common.stop'),  !busy && (running || paused),
                  (e) => lc(e, p.id, 'stop'))));
      ul.append(row);
    }
    if (state.pipelines.length === 0) {
      ul.append(el('li', { class: 'pl-item',
        style: 'cursor:default;color:var(--fg-dim)' },
        t('pl.no_pipelines')));
    }
    listBody.append(ul);
    btnSave.disabled = btnUnload.disabled = btnRename.disabled =
      !state.selectedId;
  }

  function actIcon(icon, title, enabled, onclick) {
    return el('button', {
      class: 'icon-action ' + icon, title, disabled: !enabled,
      onclick: enabled ? onclick : null,
    }, makeIcon(icon, 'sm'));
  }

  // Rebuild the canvas run/pause/stop overlay from the current pipeline
  // state (standalone editor). Enabled/disabled mirrors the selector's
  // per-row lifecycle buttons; an in-flight op disables all three.
  function renderControls() {
    if (!canvasControls) { return; }
    clear(canvasControls);
    const id = state.selectedId;
    const st = state.detail ? state.detail.state : null;
    const busy = id ? state.inflight.has(id) : false;
    canvasControls.append(
      actIcon('play', t('common.start'),
        !!id && !busy && st === 'stopped', (e) => lc(e, id, 'launch')),
      actIcon('pause', t('common.pause'),
        !!id && !busy && st === 'running', (e) => lc(e, id, 'pause')),
      actIcon('stop', t('common.stop'),
        !!id && !busy && (st === 'running' || st === 'paused'),
        (e) => lc(e, id, 'stop')));
  }

  async function lc(ev, id, op) {
    ev.stopPropagation();
    // Reject a re-click while the previous lifecycle op for this
    // pipeline is still draining (the buttons are disabled but a
    // keyboard repeat can still arrive).
    if (state.inflight.has(id)) { return; }
    state.inflight.set(id, {
      op, label: INFLIGHT_LABEL[op] || op,
    });
    renderList();   // disable buttons + show "Stopping …" badge now
    const opLabel = {
      launch: t('common.start'),
      pause:  t('common.pause'),
      stop:   t('common.stop'),
    };
    try {
      await api[op](id);
      toast(t('pl.op_done', { op: opLabel[op] || op, id }), 'ok');
    } catch (e) {
      // A refused START carries the runtime's reason for refusing it,
      // which is the same kind of text a refused load carries; pause
      // and stop fail with one line and stay toasts.
      if (op === 'launch') {
        openErrorModal({
          title: t('pl.launch_failed_title'),
          message: e.message,
          okLabel: t('common.close'),
        });
      } else {
        toast(t('pl.op_failed', { op: opLabel[op] || op, msg: e.message }),
              'error');
      }
    } finally {
      state.inflight.delete(id);
    }
    await refreshList();
  }

  // --- rebind menu (standalone editor: bound pipeline unloaded) ------
  // Enter the rebind state: drop the stale detail/selection and remember
  // which pipelines are currently loaded so the menu can offer them.
  function enterRebind(list) {
    state.missing = true;
    state.detail = null;
    state.selectedStage = null;
    state.pending = null;
    state.selectedEdge = null;
    state.rebindChoices = (list || []).map((p) => p.id);
    state.rebindKey = state.rebindChoices.join('\n');
    renderGraphPane();   // routes to renderRebindPane() while missing
    renderConfig();
  }
  // Re-point this editor at another loaded pipeline (picked from the menu).
  async function rebindTo(id) {
    boundId = id;
    state.selectedId = id;
    state.missing = false;
    state.detail = null;
    state.selectedStage = null;
    state.pending = null;
    state.selectedEdge = null;
    state.graphView = {};       // refit + clean layout for the new pipeline
    state.graphLayout = null;
    state.graphPins = new Map();
    if (typeof opts.onRebind === 'function') { opts.onRebind(id); }
    await refreshList();
    toast(t('pl.rebound', { id }), 'ok');
  }
  function renderRebindPane() {
    clear(graphBody);
    state.graphContainer = null;
    if (canvasControls) { renderControls(); graphBody.append(canvasControls); }
    stagesCount.textContent = '';
    const box = el('div', { class: 'graph-rebind' });
    box.append(el('div', { class: 'graph-rebind-title' },
      t('pl.rebind_gone', { id: state.selectedId || '' })));
    const choices = state.rebindChoices || [];
    if (!choices.length) {
      box.append(el('div', { class: 'graph-rebind-hint' },
        t('pl.no_pipelines')));
    } else {
      box.append(el('div', { class: 'graph-rebind-hint' },
        t('pl.rebind_pick')));
      const list = el('div', { class: 'graph-rebind-list' });
      for (const id of choices) {
        list.append(el('button', { class: 'graph-rebind-item',
          onclick: () => rebindTo(id) },
          makeIcon('pipeline', 'sm'), el('span', {}, id)));
      }
      box.append(list);
    }
    graphBody.append(box);
  }

  // --- middle pane --------------------------------------------------
  // The canvas needs a NODE's width (176px) plus room to route edges
  // around it; below roughly two of those it stops being a graph and
  // becomes a column of boxes with the edges folded behind them. At that
  // point the linearized rendering -- the same one the phone uses, from
  // lane-graph.js -- says strictly more about the topology in the space
  // available, so the pane switches to it rather than showing a canvas
  // nobody can read.
  const NARROW_PX = 420;

  function isNarrow() {
    // clientWidth is 0 while the pane is detached (first render, or a
    // hidden composer panel); treat unknown as WIDE so a freshly-mounted
    // editor never flashes the fallback before it has been laid out.
    const w = graphBody.clientWidth;
    return w > 0 && w < NARROW_PX;
  }

  // A configuration pane narrower than this is not worth the column it
  // costs -- the field labels alone fill it.
  const CFG_MIN_PX = 260;

  // The SECOND crossover: below it the editor stops being two panes and
  // becomes one column, with the configuration inline under its lane row.
  //
  // Measured from the EDITOR, never from the graph pane. Going one column
  // HIDES the configuration pane, which widens the graph pane -- so a
  // decision read off the graph pane would immediately un-decide itself
  // and thrash between the two layouts every frame. The editor's own
  // width is the one measurement the decision cannot move.
  //
  // Derived from the canvas crossover rather than picked, so the two
  // cannot drift apart: at this width the canvas has already given up at
  // the default split, and what is left to decide is only where the
  // configuration goes.
  function editorNarrow() {
    const w = editorArea.clientWidth;
    if (w <= 0) { return false; }        // unmeasured -> wide, as above
    // A FIXED crossover, measured from the editor and NOTHING ELSE.
    //
    // It used to add `toolbox.offsetWidth`, which is a measurement of the
    // decision's own outcome: the toolbox is hidden exactly when the lane
    // list shows, so the moment this returned true the term went to zero
    // and the next render answered false. The `+ CFG_MIN_PX` term hid
    // that -- until the dock made it conditional, and a bottom-docked
    // panel in the band between the two thresholds settled with the lane
    // list drawn and `oneColumn` false, where clicking a row selected it
    // and expanded nothing.
    //
    // Nor does the DOCK belong here any more. Where the configuration
    // sits decides how much width it costs the canvas, but this decides
    // something else: whether the panel is big enough to be worth
    // splitting at all. Below this, there is one column and the form
    // goes inline under its row, wherever the pane would have been.
    return w < NARROW_PX + CFG_MIN_PX;
  }

  // Whether the graph pane draws the lane list rather than the canvas.
  // Three causes, one answer, so the renderer and the resize watcher
  // cannot reach different ones: the editor is down to a single column,
  // the operator pinned the list on, or the pane is below the crossover.
  function laneMode(one) { return one || forceLanes || isNarrow(); }

  function renderGraphPane(o = {}) {
    if (state.missing) { renderRebindPane(); return; }
    // Recorded for the resize watcher: what this render actually drew,
    // decided once here so every branch below (including the empty
    // ones) leaves an accurate answer behind.
    // ONE COLUMN: the configuration pane and its divider go away and the
    // form is rendered inline under its lane row (see renderLaneList).
    const one = editorNarrow();
    // The lane list is what a one-column editor shows -- it is the only
    // rendering the inline form has anywhere to live. A WIDE editor whose
    // divider has been dragged hard left still reaches it on its own
    // crossover, and keeps its configuration pane.
    const narrow = laneMode(one);
    state.oneColumn = one;
    state.renderedNarrow = narrow;
    editorArea.classList.toggle('pe-narrow', one);
    // One column HIDES the configuration pane, and a hidden pane keeps
    // whatever its last render put in the header -- which would still be
    // there, stale, when the panel widens again. Empty it on the way in
    // so it is rebuilt from nothing on the way out.
    if (one) { clear(cfgActions); undoBtns = null; }
    // The surviving pane has to be TOLD to take the whole width. Being the
    // only flex item is not enough: `split` is a fraction, and an item whose
    // flex-grow sums to less than 1 gets only that share of the free space.
    // MEASURED: the pane sat at 269px with 540px of the editor empty beside
    // it.
    graphPane.style.flexGrow = one ? '1' : String(split);
    clear(graphBody);
    state.graphContainer = null;   // old overlay target is gone
    // Re-dock the (persistent) canvas run/pause/stop overlay; it floats at
    // the top-left regardless of graph/empty content (absolute-positioned).
    if (canvasControls) { renderControls(); graphBody.append(canvasControls); }
    const editable = canEdit();
    // NARROW: the docked toolbox goes away and the header's + takes over
    // (see openStagePicker). Both are driven off the SAME `narrow` the
    // lane list is, so the panel cannot be left sitting beside a list
    // there is no way to drag onto.
    graphSplit.classList.toggle('tb-hidden', narrow);
    addStageBtn.classList.toggle('show', narrow && editable);
    // The documentation toggle appears here only where the pane's own
    // copy is out of reach -- i.e. exactly when the pane is hidden.
    docsBtnHead.classList.toggle('show', one);
    // Edit affordances only make sense while stopped; drop any stale
    // arming/selection when the pipeline isn't editable.
    if (!editable) { state.pending = null; state.selectedEdge = null; }
    // Title-bar total-stage count (empty until a pipeline is selected).
    const nStages = state.detail ? state.detail.graph.nodes.length : 0;
    stagesCount.textContent = state.detail
      ? t('pl.stages_total', { n: nStages, s: nStages === 1 ? '' : 's' })
      : '';
    if (!state.detail) {
      graphBody.append(el('div', { class: 'graph-empty' },
        t('pl.select_pipeline')));
      return;
    }
    const g = state.detail.graph;
    if (!g.nodes.length) {
      graphBody.append(el('div', { class: 'graph-empty' },
        editable
          ? t('pl.empty_drag')
          : t('pl.empty')));
    } else if (narrow) {
      renderLaneList(g);
    } else {
      const gc = renderGraph(g, {
        selected: state.selectedStage,
        onSelect: (sid) => selectStage(sid),
        view: state.graphView,
        // Placement-aware re-layout + reflow animation: feed the previous
        // positions in, capture the fresh ones for next time.
        seedPos: state.graphLayout,
        onLayout: (pos) => { state.graphLayout = pos; },
        pins: state.graphPins,   // user drop-placements (column floor + row)
        // One-shot fresh (tidy) re-layout, triggered by Auto-arrange.
        freshLayout: !!o.fresh,
        // Drag-to-move a stage (persists as a pin) + the Auto-arrange
        // control next to the view buttons.
        onNodeMove,
        onAutoArrange: autoArrange,
        onNodeContext,
        // Scroll-to-pan (trackpad) or scroll-to-zoom (mouse). The graph
        // owns the toggle button -- it sits with the other view
        // controls -- and this view owns the preference.
        wheelMode,
        onWheelMode: (m) => { wheelMode = m; prefSet(LS_WHEEL, m); },
        editable,
        pending: state.pending,
        pendingType: state.pending
          ? portType(nodeById(state.pending.from), 'out',
                     state.pending.from_port)
          : null,
        // Tags of the armed output (for dimming tag-incompatible inputs
        // on top of the beat-type filter).
        pendingTags: state.pending
          ? portTags(nodeById(state.pending.from), 'out',
                     state.pending.from_port)
          : null,
        selectedEdge: state.selectedEdge,
        inputPorts: inputSlots,
        showAddInput,
        onPortClick,
        onAddInput,
        onPortDisconnect,
        onEdgeSelect,
        onBackgroundClick,
      });
      state.graphContainer = gc;
      graphBody.append(gc);
      // Dock the buffer-fill control into the graph view-control cluster
      // (bottom-left). The cluster is rebuilt with each render; bufGroup
      // is persistent and just moves into the fresh one.
      const ctls = gc.querySelector('.graph-controls');
      if (ctls) { ctls.append(bufGroup); }
      // Fill the overlay right away (don't wait a full interval) when a
      // running pipeline's graph was just (re)built.
      pollBuffersNow();
    }
  }

  // The narrow fallback: one row per stage in dependency order, with the
  // connections in a `git log --graph` gutter.
  //
  // Read-only about TOPOLOGY -- there is nowhere to drop a stage or aim
  // at a port in this width, which is why the canvas gave up in the
  // first place. Everything else the editor does still works: a row
  // EXPANDS to its stage's configuration, fully editable, rendered by
  // the same renderConfig() the wide layout puts in its own pane. The
  // note at the end says how to get the canvas back, because a pane that
  // silently changes shape owes the reader that.
  function renderLaneList(g) {
    const laid = assignLanes(topoOrder(g.nodes, g.edges), g.edges);
    const gutter = gutterWidth(laid.width);
    const list = el('div', { class: 'lane-list' });
    // Rows expand only when there is no configuration PANE to fill. A wide
    // editor dragged past the canvas crossover still has one, and clicking
    // a row there just selects it -- as it always did.
    const inline = state.oneColumn;
    // An open row whose stage has since disappeared (removed, or a
    // different pipeline selected) must not keep a block open over a row
    // that is no longer there.
    if (state.laneOpen && !laid.rows.some((r) => r.node.id === state.laneOpen)) {
      state.laneOpen = null;
    }
    for (const r of laid.rows) {
      const n = r.node;
      const bad = !!n.config_error;
      const open = inline && n.id === state.laneOpen;
      // The gutter (the graph NODE) and the row text are separate
      // controls: the node edits connections, the text opens the stage.
      // Hence a div holding two buttons -- a button inside a button is
      // invalid HTML, and the outer one swallows the inner.
      const gut = el('button', {
        class: 'lane-gutter', type: 'button', style: `width:${gutter}px`,
        title: t('conn.edit_hint'),
      }, laneArt(r, laid.width, bad));
      gut.addEventListener('click', (ev) => {
        ev.stopPropagation();
        openStageConnections(n);
      });
      const text = el('button', {
        class: 'lane-body', type: 'button',
        title: n.config_error || n.id,
        onclick: () => (inline ? toggleLaneStage(n.id) : selectStage(n.id)),
      },
        el('span', { class: 'lane-text' },
           el('span', { class: 'lane-id' }, n.id),
           el('span', { class: 'lane-type' }, n.type)),
        bad ? el('span', { class: 'lane-warn' }, '!') : null,
        inline ? el('span', { class: 'lane-caret' }, '▸') : null);
      list.append(el('div', {
        class: 'lane-row' + (bad ? ' bad' : '') + (open ? ' open' : '')
             + (n.id === state.selectedStage ? ' selected' : ''),
      }, gut, text));
      if (open) { list.append(laneDetail(r, gutter)); }
    }
    // The note says how to get the canvas back, and that differs by WHY
    // the list is on screen: a pane that shrank into it asks to be
    // widened, one the operator pinned asks for the button back.
    list.append(el('div', { class: 'lane-note' },
      t(forceLanes && !inline ? 'pl.lane_note_pinned'
        : (inline ? 'pl.narrow_note' : 'pl.narrow_note_pane'))));
    graphBody.append(list);
  }

  // The lane graph's answer to the canvas's aim-at-a-port editing: pick
  // an input from a list, then pick the output to feed it. Selecting the
  // stage first keeps the rest of the editor pointed at what is being
  // changed, exactly as expanding a row does.
  function openStageConnections(node) {
    selectStage(node.id);
    openConnectEditor({
      pipelineId: state.selectedId,
      node,
      graph: (state.detail && state.detail.graph) || { nodes: [], edges: [] },
      specFor: specForType,
      editable: canEdit(),
      onChanged: (detail) => {
        state.detail = detail;
        renderGraphPane();
        refreshList();
      },
    });
  }

  // Expand a row, or fold it back. Selecting is part of expanding, so the
  // rest of the editor (Delete, the shortcuts, a later trip back to the
  // canvas) still agrees about which stage is current.
  function toggleLaneStage(id) {
    if (state.laneOpen === id) {
      state.laneOpen = null;
      renderGraphPane();
      return;
    }
    state.laneOpen = id;
    selectStage(id);
  }

  // The block a row expands into: the configuration form, indented past
  // the gutter, with the lanes that pass this stage continuing down its
  // side. Those continuations are absolutely-positioned rules rather than
  // drawn art because the block's height depends on the form inside it --
  // without them a lane would appear to stop at whichever row is open.
  function laneDetail(row, gutter) {
    const block = el('div', { class: 'lane-detail-block' });
    for (const i of new Set([...row.through, ...row.branches])) {
      block.append(el('span', { class: 'lane-cont',
        style: `left:${laneX(i)}px` }));
    }
    const actions = el('div', { class: 'lane-detail-actions' });
    const body = el('div', { class: 'lane-detail-body' });
    block.append(el('div', { class: 'lane-detail',
      style: `margin-left:${gutter}px` }, actions, body));
    // Fire-and-forget: the form arrives when the config request does, and
    // renderConfig no-ops safely if the block was folded away meanwhile.
    renderConfig({ body, actions });
    return block;
  }

  // --- composer: toolbox + drag-create + click-to-connect ----------
  function canEdit() {
    return !!(state.detail && state.detail.state === 'stopped');
  }

  // Category display order + labels for the toolbox sections.
  // model-specific-config sits right after generative: the stages in it
  // exist to feed the generative ones, and they are the only group whose
  // applicability depends on which checkpoint is resident.
  const CATEGORY_ORDER = ['preparation', 'visual', 'vision', 'generative',
                          'model-specific-config', 'audio', 'text',
                          'network', 'control', 'database', 'generic'];

  function stageChip(s, opts = {}) {
    const ins = (s.iports || []).length;
    const outs = (s.oports || []).length;
    // Prefer the spec's human display name; the type name still drives
    // the drag payload and is surfaced in the tooltip.
    const label = tOr('stage.' + s.type + '.name', s.display_name || s.type);
    const docTxt = tOr('stage.' + s.type + '.doc', s.doc || '');
    const tip = (label !== s.type ? s.type + '\n' : '')
              + (docTxt ? docTxt + '\n' : '')
              + t('pl.chip_ports', { ins, outs })
              + t('pl.chip_hint');
    const chip = el('div', {
      class: 'tb-chip', draggable: 'true', 'data-type': s.type, title: tip,
    },
      el('span', { class: 'tb-grip' }),
      el('span', { class: 'tb-name' }, label),
      el('span', { class: 'tb-io' }, ins + '→' + outs));
    // PICK mode, for the narrow view's picker: there is no canvas to drop
    // onto, so the chip is a button and one click adds the stage. Drag is
    // turned off rather than left dangling -- a drag that can only end
    // nowhere reads as a broken affordance.
    if (opts.onPick) {
      chip.removeAttribute('draggable');
      chip.classList.add('pickable');
      chip.addEventListener('click', () => opts.onPick(s.type));
      return chip;
    }
    chip.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/vpipe-stage-type', s.type);
      e.dataTransfer.setData('text/plain', s.type);
      e.dataTransfer.effectAllowed = 'copy';
      chip.classList.add('dragging');
    });
    chip.addEventListener('dragend', () => chip.classList.remove('dragging'));
    chip.addEventListener('dblclick', () => promptCreateStage(s.type));
    return chip;
  }

  // The palette's contents for a query, grouped and ordered. Shared by
  // the side panel and the narrow view's picker so the two cannot come
  // to offer different stages, or the same stages in a different order.
  function stageSections(query) {
    const q = (query || '').toLowerCase();
    // `plugin_enabled === false` means a loaded plugin the session has
    // switched off: its stages stay in state.stageTypes so an
    // already-placed instance still renders with its spec, and drop out
    // of the toolbox so nothing new can be built from them.
    const hit = (v) => (v || '').toLowerCase().includes(q);
    // The type name and the server's English are always searchable; so is
    // what the toolbox actually SHOWS -- the localized label, doc and
    // category heading. Without those last three a reader on a non-English
    // locale can only search by text the UI is not displaying to them, and
    // the display name is not in the English text either ("Frame Dropper"
    // is neither the type nor a word in its doc).
    const match = state.stageTypes.filter((s) => !s.hidden
      && s.plugin_enabled !== false && (
      hit(s.type) || hit(s.doc) || hit(s.category)
      || hit(tOr('stage.' + s.type + '.name', s.display_name))
      || hit(tOr('stage.' + s.type + '.doc', ''))
      || hit(tOr('cat.' + s.category, ''))));
    // Built-ins bucket by category; anything a PLUGIN contributed gets a
    // section of its own, named after the plugin, after the built-ins.
    //
    // Grouped by origin rather than category on purpose. A plugin stage's
    // category says what it does, which the built-in sections already
    // cover -- what a reader cannot otherwise tell is that the stage
    // exists only because a .dylib was loaded, and will vanish from a
    // deployment that does not load it. That is the property worth a
    // heading. Plugin stages therefore appear ONCE, under their plugin,
    // not also under their category.
    const byCat = new Map();
    const byPlugin = new Map();
    for (const s of match) {
      const p = s.plugin || '';
      if (p) {
        if (!byPlugin.has(p)) { byPlugin.set(p, []); }
        byPlugin.get(p).push(s);
        continue;
      }
      const c = s.category || 'generic';
      if (!byCat.has(c)) { byCat.set(c, []); }
      byCat.get(c).push(s);
    }
    const cats = [...byCat.keys()].sort((a, b) => {
      const ia = CATEGORY_ORDER.indexOf(a);
      const ib = CATEGORY_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    const out = [];
    for (const c of cats) {
      const items = byCat.get(c).sort((a, b) => a.type.localeCompare(b.type));
      out.push({ label: tOr('cat.' + c, c) + ' (' + items.length + ')',
                 plugin: false, items });
    }
    for (const p of [...byPlugin.keys()].sort((a, b) => a.localeCompare(b))) {
      const items = byPlugin.get(p).sort(
        (a, b) => a.type.localeCompare(b.type));
      // The plugin NAME is not translated -- it is a proper noun the
      // plugin chose. Only the "plugin" label around it is.
      out.push({ label: t('pl.plugin_group') + ' ' + p
                        + ' (' + items.length + ')',
                 plugin: true, items });
    }
    return { sections: out, total: match.length };
  }

  // Render grouped sections into `body`. `opts` is passed through to
  // stageChip, which is what makes the picker's chips click-to-add.
  function renderSections(body, query, opts) {
    const { sections, total } = stageSections(query);
    for (const sec of sections) {
      body.append(el('div',
        { class: 'tb-cat' + (sec.plugin ? ' tb-plugin' : '') }, sec.label));
      for (const st of sec.items) { body.append(stageChip(st, opts)); }
    }
    if (total === 0) {
      body.append(el('div', { class: 'tb-empty' }, t('pl.no_matches')));
    }
    return total;
  }

  function renderToolbox() {
    clear(toolboxBody);
    renderSections(toolboxBody, toolboxFilter.value || '', {});
  }

  // --- type helpers (for client-side connection validation) --------
  function nodeById(id) {
    return ((state.detail && state.detail.graph.nodes) || [])
      .find((n) => n.id === id) || null;
  }
  function specForType(typeName) {
    return state.stageTypes.find((s) => s.type === typeName) || null;
  }
  // Declared beat type of a port. For an iport index past the node's
  // current ports (an append target) fall back to the stage spec's
  // declared port at that index; 'any' when unknown.
  function portType(node, kind, idx) {
    if (!node) { return 'any'; }
    const live = kind === 'in' ? node.iports : node.oports;
    if (idx < live.length) { return live[idx].type || 'any'; }
    const sp = specForType(node.type);
    const decl = sp ? (kind === 'in' ? sp.iports : sp.oports) : null;
    return (decl && decl[idx] && decl[idx].type) || 'any';
  }
  // typesCompatible / tagsCompatible come from port-compat.js: the
  // connection editor asks the same question about the same ports, and a
  // second copy of the rule is one that can fall behind the runtime's.
  // Parse a comma-separated tag list into a trimmed, non-empty array.
  function parseTags(s) {
    return (s || '').split(',').map((x) => x.trim()).filter(Boolean);
  }
  // Declared payload tags of a port (finer constraint on the beat type).
  // Prefer the live port's tags; fall back to the stage spec's declared
  // port at that index. Empty array = no tag constraint.
  function portTags(node, kind, idx) {
    if (!node) { return []; }
    const live = kind === 'in' ? node.iports : node.oports;
    if (idx < live.length && live[idx] && live[idx].tags !== undefined) {
      return parseTags(live[idx].tags);
    }
    const sp = specForType(node.type);
    const decl = sp ? (kind === 'in' ? sp.iports : sp.oports) : null;
    return parseTags(decl && decl[idx] && decl[idx].tags);
  }

  // Input-port slots to render for a node: the stage spec's declared
  // iports (named + typed, indexed), widened to the wired count so an
  // extra live connection is never hidden. Each iport index has a
  // distinct semantic, so the slot keeps its index + name. Falls back
  // to the live wired iports when the type has no registered spec.
  function inputSlots(node) {
    const sp = specForType(node.type);
    if (!sp) { return node.iports; }
    const decl = sp.iports || [];
    const live = node.iports || [];
    const n = Math.max(decl.length, live.length);
    const slots = [];
    for (let i = 0; i < n; i++) {
      const d = decl[i] || null;
      const l = live[i] || null;
      slots.push({
        name: (d && d.name) || '',
        type: (d && d.type) || (l && l.type) || 'any',
        tags: (d && d.tags) || (l && l.tags) || '',
        doc:  (d && d.doc) || '',
      });
    }
    return slots;
  }
  // The "+ add input" stub only makes sense for a stage whose input
  // arity is unknown (no registered spec) -- those still grow one input
  // at a time. Spec-driven stages expose their fixed named slots
  // instead, so the stub is suppressed for them.
  function showAddInput(node) {
    return !specForType(node.type);
  }

  // A unique stage id for a freshly dropped type: the type name, or
  // "type-2", "type-3", … if taken.
  function suggestId(type) {
    const taken = new Set(
      (state.detail && state.detail.graph.nodes || []).map((n) => n.id));
    if (!taken.has(type)) { return type; }
    let i = 2;
    while (taken.has(type + '-' + i)) { ++i; }
    return type + '-' + i;
  }

  // `dropWorld` (optional) is the {x,y} world point the stage was dropped
  // at; when given, the new stage is pinned to that grid cell (column
  // floor + row) so its placement is honoured.
  function promptCreateStage(type, dropWorld) {
    if (!canEdit()) {
      toast(t('pl.select_stopped'), 'error');
      return;
    }
    const idIn = el('input', { type: 'text', value: suggestId(type) });
    openModal({
      title: t('pl.add_stage_title', { type }),
      body: el('div', {},
        el('label', { class: 'fl' }, t('pl.stage_id')), idIn,
        el('p', { class: 'doc', style: 'margin-top:10px' },
          t('pl.add_stage_help'))),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.add'), kind: 'primary', onClick: async (c) => {
            const id = idIn.value.trim();
            if (!id) { toast(t('pl.stage_id_required'), 'error'); return; }
            try {
              state.detail = await api.insertStage(state.selectedId, {
                id, type, iports: [], config: {} });
              c();
              // Pin the new stage to the dropped grid cell (snapped +
              // distance-capped) so the layout honours where the operator
              // put it. No drop point (toolbar add / empty canvas) ->
              // auto-place as before.
              if (dropWorld) {
                state.graphPins.set(id, worldToPin(dropWorld,
                  state.graphLayout));
              }
              state.selectedStage = id;
              state.pending = null;
              state.selectedEdge = null;
              renderGraphPane();
              await renderConfig();
              await refreshList();
              toast(t('pl.added', { id }), 'ok');
            } catch (e) {
              toast(t('pl.add_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
    setTimeout(() => { idIn.focus(); idIn.select(); }, 0);
  }

  // Edge key is "from#from_port>to#to_port" (see graph.js edgeKey).
  function parseEdgeKey(key) {
    const gt = key.lastIndexOf('>');
    const right = key.slice(gt + 1);
    const h = right.lastIndexOf('#');
    return { to: right.slice(0, h), to_port: parseInt(right.slice(h + 1), 10) };
  }

  function onPortClick(kind, stageId, idx) {
    if (!canEdit()) { return; }
    if (kind === 'out') {
      // Arm this output, or toggle it off if it's already armed.
      if (state.pending && state.pending.from === stageId
          && state.pending.from_port === idx) {
        state.pending = null;
      } else {
        state.pending = { from: stageId, from_port: idx };
      }
      state.selectedEdge = null;
      renderGraphPane();
      return;
    }
    // kind === 'in': complete the connection. Iports are positional and
    // may be gapped -- the core wires a higher-indexed input while a
    // lower one stays unconnected (it pads the skipped lower slots as
    // gaps), so any declared slot is connectable in any order. Wiring an
    // already-wired slot re-points it.
    if (!state.pending) {
      toast(t('pl.wire_output_first'), 'info');
      return;
    }
    if (!checkCompatible(stageId, idx)) { return; }
    doConnect({ from: state.pending.from, from_port: state.pending.from_port,
                to: stageId, to_port: idx });
  }

  // The dashed "+ add input" stub: append a new input (no to_port).
  function onAddInput(stageId) {
    if (!canEdit()) { return; }
    if (!state.pending) {
      toast(t('pl.wire_output_first'), 'info');
      return;
    }
    const node = nodeById(stageId);
    const newIdx = node ? node.iports.length : 0;
    if (!checkCompatible(stageId, newIdx)) { return; }
    doConnect({ from: state.pending.from, from_port: state.pending.from_port,
                to: stageId });
  }

  // Refuse a wire whose endpoint beat types disagree (both declared and
  // unequal). 'any' on either side is permitted. The server's
  // h_connect_ re-checks as a backstop.
  function checkCompatible(toId, toPort) {
    const ot = portType(nodeById(state.pending.from), 'out',
                        state.pending.from_port);
    const it = portType(nodeById(toId), 'in', toPort);
    if (!typesCompatible(ot, it)) {
      toast(t('pl.incompatible',
              { from: shortType(ot), to: shortType(it) }), 'error');
      return false;
    }
    // Deeper check: payload tags (finer than the beat type; OR semantics).
    const otg = portTags(nodeById(state.pending.from), 'out',
                         state.pending.from_port);
    const itg = portTags(nodeById(toId), 'in', toPort);
    if (!tagsCompatible(otg, itg)) {
      toast(t('pl.incompatible_tags',
              { from: otg.join(', ') || '—', to: itg.join(', ') || '—' }),
            'error');
      return false;
    }
    return true;
  }

  async function doConnect(edge) {
    try {
      state.detail = await api.stageConnect(state.selectedId, edge);
      state.pending = null;
      renderGraphPane();
      await refreshList();
      toast(t('pl.connected'), 'ok');
    } catch (e) {
      toast(t('pl.connect_failed', { msg: e.message }), 'error');
    }
  }

  function onPortDisconnect(stageId, iport) {
    if (!canEdit()) { return; }
    doDisconnect(stageId, iport);
  }

  async function doDisconnect(to, to_port) {
    try {
      // Freeze the consumer where it currently sits: losing an input edge
      // would otherwise drop it to column 0 on the re-layout. Pin its
      // present cell so it stays put (Auto-arrange re-tidies later).
      if (state.graphLayout && state.graphLayout.get(to)) {
        const p = state.graphLayout.get(to);
        state.graphPins.set(to, { col: p.rank || 0, y: p.y });
      }
      state.detail = await api.stageDisconnect(state.selectedId,
                                               { to, to_port });
      state.selectedEdge = null;
      renderGraphPane();
      await refreshList();
      toast(t('pl.disconnected'), 'ok');
    } catch (e) {
      toast(t('pl.disconnect_failed', { msg: e.message }), 'error');
    }
  }

  function onEdgeSelect(key) {
    state.selectedEdge = key;
    if (key) { state.pending = null; state.selectedStage = null; }
    renderGraphPane();
    renderConfig();
  }

  function onBackgroundClick() {
    if (state.pending) { state.pending = null; renderGraphPane(); }
  }

  // A stage was dragged to a new spot: persist it as a pin (so a later
  // re-layout keeps the placement) and update the stored positions in
  // place. Deliberately does NOT re-render -- the node already sits at its
  // snapped position and nothing else on the canvas should shift.
  function onNodeMove(id, pin, pos) {
    state.graphPins.set(id, pin);
    if (state.graphLayout) {
      const cur = state.graphLayout.get(id);
      if (cur) {
        state.graphLayout.set(id, { ...cur, x: pos.x, y: pos.y });
      }
    }
  }

  // Auto-arrange: drop every manual placement (drag pins, drop pins,
  // disconnect freezes) and re-run a fresh tidy layout, gliding from the
  // current positions so the tidy-up is visible.
  function autoArrange() {
    if (!state.detail || !state.detail.graph.nodes.length) { return; }
    state.graphPins = new Map();
    renderGraphPane({ fresh: true });
  }

  // Toggle the selected-node highlight in the LIVE graph without a
  // re-layout, so a stage the user hand-placed keeps its spot when they
  // click it (only Auto-arrange should reshuffle). Returns false when there
  // is no live graph to patch (caller falls back to a full render).
  function patchSelection(sid) {
    const gc = state.graphContainer;
    if (!gc) { return false; }
    gc.querySelectorAll('.gnode').forEach((elm) => {
      elm.classList.toggle('selected', elm.dataset.id === sid);
    });
    return true;
  }

  async function selectStage(sid) {
    state.selectedStage = sid;
    // In the narrow layout, selecting IS expanding -- the form has nowhere
    // else to go. Set this before the render below, which draws the block.
    if (state.oneColumn) { state.laneOpen = sid; }
    // If an edge/arming overlay is showing it must be cleared, which needs
    // a real re-render; otherwise just repaint the highlight in place so
    // dragged positions are preserved.
    const hadOverlay = !!(state.selectedEdge || state.pending);
    state.selectedEdge = null;
    state.pending = null;
    if (hadOverlay || !patchSelection(sid)) {
      renderGraphPane();
    }
    // Narrow: renderGraphPane just rebuilt the list, and the expanded row
    // rendered the form inline. Filling the (hidden) pane too would only
    // buy a second request for the same config.
    if (!state.oneColumn) { await renderConfig(); }
  }

  // How many columns the configuration form gets, and which fields
  // start a row.
  //
  // MEASURED rather than declared: the panel is resized by the window,
  // by the divider and by the dock button, and the answer has to follow
  // all three. It is not a CSS auto-fill because two of the rules cannot
  // be said that way -- never more columns than there are FIELDS (three
  // fields in a wide panel are three columns, not five with two empty),
  // and a one-field form takes the WHOLE width rather than a column's
  // worth of it, which is what a text-prompt stage wants.
  const CFG_COL_MIN = 280;   // a column worth having: ~250px of content
  const CFG_FIELD_GAP = 12;  // .field's margin-bottom, in the stack sum
  function layoutCfgColumns(scope) {
    const root = scope || editorArea;
    for (const grid of root.querySelectorAll('.cfg-fields')) {
      const items = grid.children;
      const n = items.length;
      const w = grid.clientWidth;
      // Detached (a form built before its panel is laid out) or empty:
      // one column, and the next call measures for real.
      let cols = 1;
      if (w > 0 && n > 0) {
        // What the panel can hold side by side...
        const byWidth = Math.max(1, Math.floor(w / CFG_COL_MIN));
        // ...and what it needs. HOW TALL THE FIELDS ARE AS ONE STACK
        // against how much panel is left below them: a form that already
        // fits stays a single column, however wide the panel gets, and
        // only a stack that would scroll is split. Field heights barely
        // move with the column width (a label may wrap), so this is an
        // estimate -- which is all the decision needs.
        let stack = 0;
        for (let i = 0; i < n; i++) {
          stack += items[i].offsetHeight + CFG_FIELD_GAP;
        }
        const scroller = grid.closest('.pane-body');
        const avail = scroller
            ? scroller.getBoundingClientRect().bottom
                - grid.getBoundingClientRect().top
            : 0;
        // No height to measure against (detached, or a panel collapsed
        // to nothing): fall back to what the width allows.
        const byHeight = avail > 40 ? Math.ceil(stack / avail) : byWidth;
        cols = Math.min(byWidth, Math.max(1, byHeight), n);
      }
      // Written only when it CHANGES. Re-styling the form is what the
      // resize observer is watching for, so an unconditional write would
      // keep waking it -- and a count that oscillates between two values
      // would then never settle.
      if (grid.style.getPropertyValue('--cols') !== String(cols)) {
        grid.style.setProperty('--cols', String(cols));
      }
    }
  }

  // --- the configuration form --------------------------------------
  // `target` names where the form goes: the standalone pane by default,
  // or -- in the narrow layout, which has no such pane -- the block an
  // expanded lane row supplies. Everything else about the form (the
  // header Apply/Remove, auto-apply on blur, the schema fields) is
  // identical, because it is the same function.
  async function renderConfig(target) {
    // The narrow layout has no configuration pane -- an expanded row
    // renders the form itself and passes a target. An untargeted call
    // there would fetch the same config again to fill a hidden pane.
    if (!target && state.oneColumn) { return; }
    const body = (target && target.body) || cfgBody;
    const cfgHead = (target && target.actions) || cfgActions;
    const seq = ++cfgSeq;
    // Whether THIS render may still write. Two ways it may not: an inline
    // block folded away while its request was in flight (detached, so
    // there is nothing to fill), or a newer render started meanwhile --
    // which has already emptied the header this one is about to fill.
    const live = () => (target ? body.isConnected : true) && seq === cfgSeq;
    // WHAT THE USER IS DOING, remembered across the teardown.
    //
    // A rebuild replaces every control, so the focused one is removed
    // from the document: focus falls back to <body> and the scroller
    // jumps to the top. Tabbing through a form the user has EDITED then
    // behaves nothing like tabbing through one they have not, which is
    // the report this exists to answer.
    //
    // The key is captured rather than the element, because the element
    // is exactly the thing that will not survive.
    const scroller = body.closest('.pane-body') || body;
    const prevTop = scroller.scrollTop;
    const act = document.activeElement;
    const prevKey = act && body.contains(act) ? act.dataset.cfgKey : null;
    const prevSel = prevKey && typeof act.selectionStart === 'number'
        ? [act.selectionStart, act.selectionEnd] : null;
    const restore = () => {
      // ONLY when the rebuild happened under the user's hands. Every
      // other caller -- selecting another stage, coming back from the
      // one-column layout -- is showing a DIFFERENT form, and putting
      // the old scroll position on it would be its own bug.
      if (!prevKey) { return; }
      // Scroll first: focusing scrolls too, and doing it after would
      // fight the browser's own scroll-into-view.
      scroller.scrollTop = prevTop;
      const next = body.querySelector(
          `[data-cfg-key="${CSS.escape(prevKey)}"]`);
      if (!next) { return; }
      // preventScroll: the field is already where it was; letting the
      // browser scroll to it would undo the line above.
      try { next.focus({ preventScroll: true }); } catch (e) { next.focus(); }
      if (prevSel && typeof next.setSelectionRange === 'function') {
        try { next.setSelectionRange(prevSel[0], prevSel[1]); } catch (e) {}
      }
      scroller.scrollTop = prevTop;
    };
    clear(body);
    clear(cfgHead);   // header buttons; refilled below when editable
    undoBtns = null;  // the ones just cleared are detached now
    if (!state.detail || !state.selectedStage) {
      body.append(el('div', { class: 'cfg' },
        el('div', { class: 'empty' }, t('pl.select_stage_config'))));
      return;
    }
    let info;
    try {
      info = await api.getStageConfig(state.selectedId, state.selectedStage);
    } catch (e) {
      if (!live()) { return; }
      body.append(el('div', { class: 'cfg' },
        el('div', { class: 'empty' },
          t('pl.config_unavailable', { msg: e.message }))));
      return;
    }
    if (!live()) { return; }
    const editable = !!info.editable;
    const wrap = el('div', { class: 'cfg' });

    // Apply / Remove go in the pane HEADER (or, inline, at the top of the
    // block) so they stay pinned above the scroll. `inputs` is captured by
    // Apply's handler and filled in by the schema loop below.
    const inputs = [];
    // The config this form was built from, in the shape a PUT takes: the
    // PRESENT keys with their current values, which is exactly what the
    // untouched form would read back. It is what Undo restores to, and
    // what the first apply after this render is paired against.
    {
      const seed = {};
      for (const f of info.schema) {
        const here = f.present !== undefined
            ? !!f.present
            : (f.current !== undefined && f.current !== null);
        if (here && f.current !== undefined) { seed[f.key] = f.current; }
      }
      cfgLast = { sid: state.selectedStage, cfg: seed };
    }
    // Undo / Redo are the same slot read in two directions, so one
    // handler serves both and only their enabled state differs. Both are
    // edits, so both need a stopped pipeline.
    const undoBtn = el('button', {
      class: 'btn ghost mini', disabled: true,
      title: t('pl.undo_hint'),
      onclick: () => stepConfigUndo(),
    }, t('common.undo'));
    const redoBtn = el('button', {
      class: 'btn ghost mini', disabled: true,
      title: t('pl.redo_hint'),
      onclick: () => stepConfigUndo(),
    }, t('common.redo'));
    undoBtns = { undo: undoBtn, redo: redoBtn };
    syncUndoBtns();
    const applyBtn = el('button', {
      class: 'btn primary mini', disabled: !editable,
      onclick: () => applyConfig(inputs),
    }, t('common.apply'));
    // Remove the stage (topology edit -- stopped pipelines only). Label
    // only: an icon beside the text makes this button taller than the
    // plain ones next to it, and a row of header buttons that do not
    // line up reads as a mistake.
    const removeBtn = el('button', {
      class: 'btn danger mini', disabled: !editable,
      title: editable ? t('pl.remove_stage_title')
                      : t('pl.stop_to_edit'),
      onclick: () => onRemoveStage(),
    }, t('common.remove'));
    cfgHead.append(undoBtn, redoBtn, applyBtn, removeBtn);

    // Auto-apply on blur (stopped pipelines only): when a field commits (a
    // text/number box loses focus after an edit), re-POST the whole config
    // without rebuilding the pane. Self-gates on canEdit() so a running
    // pipeline never hot-applies -- there the Apply button is the deliberate
    // trigger (running-config support lands later). `inputs` is captured here
    // and populated by the schema loop below (filled before any blur fires).
    const commit = () => {
      if (canEdit()) { applyConfig(inputs, { rerender: false }); }
    };
    // The graph refresh an auto-apply deferred (see applyConfig). Run
    // when focus leaves the FORM, not merely the field: tabbing between
    // two fields passes through here with relatedTarget still inside,
    // and rebuilding then would be the very teardown that was deferred.
    //
    // Bound ONCE per host element -- clear() empties the body but leaves
    // its own listeners, so binding per render would stack a copy per
    // rebuild.
    if (!body.dataset.cfgFocusHook) {
      body.dataset.cfgFocusHook = '1';
      body.addEventListener('focusout', (e) => {
        if (!state.graphStale) { return; }
        if (e.relatedTarget && body.contains(e.relatedTarget)) { return; }
        state.graphStale = false;
        renderGraphPane();
      });
    }

    wrap.append(el('div', { class: 'stage-id' }, info.id));

    const sp = specForType(info.type);
    wrap.append(el('div', { class: 'stage-type' },
      info.type,
      sp && sp.category && sp.category !== 'generic'
        ? el('span', { class: 'stage-cat' },
            tOr('cat.' + sp.category, sp.category)) : null));
    if (sp && sp.doc) {
      wrap.append(el('div', { class: 'stage-desc' },
        tOr('stage.' + info.type + '.doc', sp.doc)));
    }
    if (!editable) {
      wrap.append(el('div', { class: 'ro-note' },
        t('pl.config_readonly')));
    }
    // The fields get their own grid so the stage identity above them
    // stays in normal flow and needs no column span of its own;
    // layoutCfgColumns fills in how many columns it is given.
    const fieldsWrap = el('div', { class: 'cfg-fields' });
    for (const f of info.schema) {
      const { field, read } = configField(f, !editable, info.type, commit);
      inputs.push({ key: f.key, type: f.type, read });
      // Tag the field's own controls with the key. A rebuild replaces
      // every element, so the only way to put the caret back where it
      // was is to find the NEW control that means the same thing.
      for (const c of field.querySelectorAll(
             'input, select, textarea')) {
        c.dataset.cfgKey = f.key;
      }
      fieldsWrap.append(field);
    }
    wrap.append(fieldsWrap);
    body.append(wrap);
    // Now that it is in the document and has a width to measure.
    layoutCfgColumns(body);
    restore();
  }

  // Build one config field. `read()` returns:
  //   - undefined  -> "field intentionally unset, omit from POST"
  //   - any other  -> value to send under f.key
  //
  // Fields render EMPTY when the backend reports `present === false`, with
  // the schema default in the placeholder so the user knows what the value
  // resolves to if left blank. This lets the user wipe a field (e.g. chrono's
  // frequency_hz) so the mutually-exclusive period_* set is the only thing
  // that reaches the stage's validator.
  //
  // How "empty" is read back depends on whether the type HAS an empty value:
  //   bool / string / text  tri-state. An omitted key and a key present as ""
  //                         (or a present `false`) are different configs, so
  //                         `unset` is tracked explicitly rather than inferred
  //                         from the box being blank. The Clear button moves
  //                         between the two and shows which is live.
  //   int / uint / real     blank == unset. There is no "empty number".
  //   array / object / any  blank == unset. An empty box is not valid JSON.
  //
  // Anything that fully determines a value on its own -- Clear/unset, a
  // checkbox flip, a path or model pick -- calls `onCommit` straight away;
  // free-text boxes wait for the blur-after-edit `change` instead.
  //
  // Backward-compat: older backends predate the `present` flag. When
  // it's absent we fall back to "show whatever current is" so the
  // editor doesn't strand the user with empty inputs against an
  // un-upgraded server.
  // Compatibility-aware model browser for a model-registry field.
  // Lists installed models filtered by the field's model_type allow-list
  // (suggest_db_type), its required I/O modalities, and parent
  // compatibility against what the form's other fields already name.
  // Picking a model sets the field via onPick(key).
  async function openModelBrowser(field, input, onPick) {
    let data;
    try {
      data = await api.modelsInstalled();
    } catch (e) {
      toast(t('pl.mb_failed', { msg: e.message }), 'error');
      return;
    }
    const all = (data && data.models) ? data.models : [];

    // WHAT THIS FIELD MAY OFFER: the field's own rules (its
    // suggest_db_type allow-list and its required I/O modalities) plus
    // parent compatibility, against the installed models the form's
    // OTHER fields already name. Shared with the phone sheet -- see
    // model-filter.js, where a copy of this rule drifted.
    const chosen = new Set();
    document.querySelectorAll('[id^="f_"]').forEach((elm) => {
      if (elm === input) { return; }
      const v = (elm.value || '').trim();
      if (v) { chosen.add(v); }
    });
    const compatible = compatibleModels(all, field, chosen);

    let closeModal = () => {};
    const groups = [
      ['model', t('pl.mb_g_models')],
      ['supplement', t('pl.mb_g_supplements')],
      ['dataset', t('pl.mb_g_datasets')],
    ];
    const bodyEl = el('div', { class: 'model-browser' });

    // ---- narrowing, on top of the compatibility rules above ---------
    //
    // Those decide what the FIELD can accept and are not negotiable.
    // These are the operator's: with a few dozen installed models the
    // list is longer than a glance, and the thing being looked for is
    // usually known by a fragment of its name or by what it consumes
    // and produces.
    //
    // Only offered when there is something to narrow. A filter bar over
    // three rows is furniture.
    const wantIn = new Set();
    const wantOut = new Set();
    let query = '';
    const inKinds = modalitiesPresent(compatible, 'inputs');
    const outKinds = modalitiesPresent(compatible, 'outputs');
    const matches = (m) => modelMatches(m, { query, wantIn, wantOut });

    const listWrap = el('div', { class: 'mb-results' });
    const count = el('div', { class: 'mb-count' });
    const renderList = () => {
      clear(listWrap);
      const shown = compatible.filter(matches);
      count.textContent = shown.length === compatible.length
        ? t('pl.mb_count', { n: compatible.length })
        : t('pl.mb_count_filtered',
            { n: shown.length, total: compatible.length });
      if (!shown.length) {
        // Distinct from "nothing is compatible": there ARE models, the
        // filter is what hid them, and the fix is to relax it.
        listWrap.append(el('div', { class: 'mb-empty' },
          t('pl.mb_no_match')));
        return;
      }
      for (const [cat, label] of groups) {
        const items = shown.filter((m) => (m.category || 'model') === cat);
        if (!items.length) { continue; }
        listWrap.append(el('div', { class: 'mb-group' }, label));
        const list = el('div', { class: 'mb-list' });
        for (const m of items) {
          list.append(modelCard(m, () => { onPick(m.key); closeModal(); }));
        }
        listWrap.append(list);
      }
    };

    // One row of modality chips. `want` is the live Set it toggles.
    const chipRow = (label, kinds, want) => {
      if (!kinds.length) { return null; }
      const row = el('div', { class: 'mb-chips' },
        el('span', { class: 'mb-chips-label' }, label));
      for (const x of kinds) {
        const b = el('button', { class: 'mb-chip mb-' + x, type: 'button' },
          x);
        b.addEventListener('click', () => {
          if (want.has(x)) { want.delete(x); } else { want.add(x); }
          b.classList.toggle('active', want.has(x));
          renderList();
        });
        row.append(b);
      }
      return row;
    };

    if (!compatible.length) {
      bodyEl.append(el('div', { class: 'mb-empty' }, t('pl.mb_empty')));
    } else {
      // The threshold is deliberately low. Six rows already overflow the
      // modal on a laptop, and a search box that is present sometimes is
      // harder to rely on than one that is always there -- but a bar
      // over two or three rows is noise, so it is not unconditional.
      const search = el('input', { type: 'search', class: 'mb-search',
        placeholder: t('pl.mb_search'), autocomplete: 'off' });
      search.addEventListener('input', () => {
        query = search.value.trim().toLowerCase();
        renderList();
      });
      // Escape CLEARS a non-empty box before the modal's own handler
      // closes the dialog. Losing a whole browse to a keystroke meant
      // for the filter is the wrong outcome, and an operator who wants
      // out presses it twice.
      search.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && search.value) {
          e.stopPropagation();
          search.value = '';
          query = '';
          renderList();
        }
      });
      if (compatible.length > 4) {
        bodyEl.append(el('div', { class: 'mb-filter' },
          search,
          chipRow(t('pl.mb_in'), inKinds, wantIn),
          chipRow(t('pl.mb_out'), outKinds, wantOut)));
        setTimeout(() => search.focus(), 0);
      }
      bodyEl.append(count);
      renderList();
      bodyEl.append(listWrap);
    }
    closeModal = openModal({
      title: t('pl.mb_title'),
      className: 'model-browser-modal',
      body: bodyEl,
      actions: [{ label: t('common.close'), cancel: true,
        onClick: (c) => c() }],
    });
  }

  // One model row in the browser: the registry KEY, then the variant,
  // model_type + size, the parent it attaches to (supplements), and
  // in→out modality badges.
  //
  // The key is the title because it is the only field guaranteed to
  // identify the row -- it is what picking writes into the config, and
  // it is unique by construction. The variant is a LABEL for a pack
  // ("bf16 single-file (Comfy-Org)") and several models legitimately
  // share one: MiniMax-H3 publishes its FL2VA and Ref2VA partitions
  // from a single repo, so showing `variant || key` drew two different
  // models as two identical cards, distinguishable only by a tooltip.
  function modelCard(m, onClick) {
    const io = (arr) => ((arr && arr.length)
      ? arr.map((x) => el('span', { class: 'mb-badge mb-' + x }, x))
      : [el('span', { class: 'mb-badge mb-none' }, '—')]);
    const sub = [m.model_type,
      [m.family, m.param_class].filter(Boolean).join(' ')]
      .filter(Boolean).join(' · ');
    const parentNote = m.parent_model_type
      ? el('span', { class: 'mb-parent' },
          t('pl.mb_attaches', { p: m.parent_model_type
            + (m.parent_param_class ? ' ' + m.parent_param_class : '') }))
      : null;
    const variantNote = m.variant
      ? el('div', { class: 'mb-variant' }, m.variant)
      : null;
    return el('button', { class: 'mb-card', type: 'button', onclick: onClick },
      el('div', { class: 'mb-card-main' },
        el('div', { class: 'mb-name', title: m.variant || m.key }, m.key),
        variantNote,
        el('div', { class: 'mb-sub' }, sub, parentNote)),
      el('div', { class: 'mb-io' },
        el('span', { class: 'mb-io-set' }, io(m.inputs)),
        el('span', { class: 'mb-io-arrow' }, '→'),
        el('span', { class: 'mb-io-set' }, io(m.outputs))));
  }

  function configField(f, disabled, type, onCommit) {
    const id = 'f_' + f.key;
    const placeholder = f.default !== undefined && f.default !== null
        ? t('pl.field_default', { val: typeof f.default === 'object'
            ? JSON.stringify(f.default) : String(f.default) })
        : t('pl.unset');
    const present = f.present !== undefined
        ? !!f.present
        : (f.current !== undefined && f.current !== null);
    let input, read, unsetBtn = null, datalist = null, browseBtn = null;
    let modelBtn = null;
    // Set by the branch that actually renders a JSON textarea, never
    // re-derived from f.type further down. The auto-apply wiring at the
    // bottom has to agree with what was rendered, and a second type list
    // kept in step by hand is the stale fall-through this tree keeps
    // finding: the final `else` also catches a type neither list knows.
    let isJson = false;
    // Two-state fields (numbers, JSON) have no "present but empty" reading --
    // an empty number is not a number, an empty box is not valid JSON -- so
    // blank IS unset and `read()` already omits the key. They still need the
    // same INDICATOR the tri-state fields below get, or the button keeps
    // offering "clear" on a field that is already cleared and nothing
    // distinguishes unset from set. Assigned per branch; called after any
    // programmatic change to the box.
    let syncUnset = () => {};
    // Drops the "did not parse" mark on a JSON box. Assigned by the
    // auto-apply wiring at the bottom and called from anywhere that sets
    // `input.value` PROGRAMMATICALLY: that fires no `input` event, so the
    // listener which normally clears the mark never runs and a box the
    // user has just emptied would keep wearing the warning for a value
    // it no longer holds. Declared here, assigned later -- every caller
    // is a click handler, so it runs long after the assignment.
    let clearJsonMark = () => {};
    const emptyMeansUnset = () => {
      syncUnset = () => {
        const unset = input.value.trim() === '';
        input.classList.toggle('field-unset', unset);
        unsetBtn.textContent = unset ? t('pl.btn_unset') : t('pl.btn_clear');
        unsetBtn.title = unset ? t('pl.unset_type_value') : t('pl.clear_omit');
        unsetBtn.classList.toggle('active', unset);
      };
      input.addEventListener('input', () => syncUnset());
    };
    // Tri-state for the free-text fields (string / text). An OMITTED key and a
    // key present as "" are different configs -- the first lets the stage fall
    // back to its declared default, the second forces an empty value -- so an
    // empty box alone cannot mean both. Track `unset` explicitly: typing marks
    // the field set (even when the text is deleted back to empty), and the
    // button is what moves between the two, clearing to unset and back to an
    // explicit empty. Placeholder and button label show which state is live.
    // Returns the field's `read`; assigns the shared `unsetBtn`.
    const textTriState = () => {
      let unset = !present;
      const sync = () => {
        input.classList.toggle('field-unset', unset);
        input.placeholder = unset ? placeholder : t('pl.empty_value');
        unsetBtn.textContent = unset ? t('pl.btn_unset') : t('pl.btn_clear');
        unsetBtn.title = unset ? t('pl.set_empty') : t('pl.clear_omit');
        unsetBtn.classList.toggle('active', unset);
      };
      input.addEventListener('input', () => {
        if (unset) { unset = false; sync(); }
      });
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        disabled, onclick: () => {
          unset = !unset;
          if (unset) { input.value = ''; } else { input.focus(); }
          sync();
          // Clearing (or un-clearing) is a complete edit on its own; there is
          // no blur to wait for, so commit now.
          if (onCommit) { onCommit(); }
        } });
      sync();
      return () => (unset ? undefined : input.value);
    };
    if (f.type === 'bool') {
      // Two-state checkbox + an "unset" badge that toggles tri-state
      // by removing the field entirely on the next read.
      input = el('input', { type: 'checkbox', id, disabled });
      input.checked = !!f.current;
      let unset = !present;
      const updateUi = () => {
        input.disabled = disabled || unset;
        unsetBtn.textContent = unset ? t('pl.btn_unset') : t('pl.btn_clear');
        unsetBtn.classList.toggle('active', unset);
      };
      unsetBtn = el('button', { class: 'btn ghost mini',
        type: 'button', title: t('pl.omit_field'),
        disabled, onclick: () => {
          unset = !unset; updateUi();
          // Toggling set/unset IS the whole edit for a checkbox -- there is
          // no blur to wait for, so commit immediately.
          if (onCommit) { onCommit(); }
        } });
      updateUi();
      // Same for flipping the checkbox itself: `change` fires on click, and
      // the value is already final at that point.
      input.addEventListener('change', () => { if (onCommit) { onCommit(); } });
      read = () => (unset ? undefined : input.checked);
    } else if (f.type === 'int' || f.type === 'uint' || f.type === 'real') {
      input = el('input', { type: 'number', id, disabled,
        step: f.type === 'real' ? 'any' : '1',
        placeholder,
        value: present ? (f.current ?? '') : '' });
      // The scroll wheel must never change a numeric config value. The
      // browser only increments/decrements a number input while it's
      // FOCUSED, so intercept the wheel only there -- normal page/pane
      // scrolling stays intact when the field isn't focused. (The spinner
      // up/down buttons are hidden in CSS.)
      input.addEventListener('wheel', (e) => {
        if (document.activeElement === input) { e.preventDefault(); }
      }, { passive: false });
      // An empty number box has no "present but empty" reading -- there is no
      // such number -- so blank always means unset, for both the Clear button
      // and a manual delete.
      read = () => {
        const s = input.value.trim();
        if (s === '') { return undefined; }
        const n = f.type === 'real' ? parseFloat(s) : parseInt(s, 10);
        return Number.isNaN(n) ? NaN : n;
      };
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        disabled, onclick: () => {
          // Already unset: nothing to clear, so just invite a value.
          if (input.value.trim() === '') { input.focus(); return; }
          input.value = ''; syncUnset(); input.focus();
          if (onCommit) { onCommit(); }
        } });
      emptyMeansUnset();
      syncUnset();
    } else if (f.type === 'string' && Array.isArray(f.choices)
               && f.choices.length) {
      // A CLOSED SET (the schema's `choices`): every value this key
      // accepts, so offer them instead of a text box the reader has to
      // guess at. The blank option IS the unset state -- the same one the
      // Clear button toggles -- so textTriState below drives this exactly
      // as it drives a text field.
      input = el('select', { id, disabled });
      const cur = present ? String(f.current ?? '') : '';
      input.append(el('option', { value: '' }, t('pl.unset')));
      for (const c of f.choices) {
        input.append(el('option', { value: String(c) }, String(c)));
      }
      // A value the set does not contain: an older pipeline file, or a
      // spelling the stage still accepts but no longer advertises
      // (unload_when_idle's legacy always / never). Keep it, and say so.
      // A form asked to DISPLAY a config must not quietly rewrite it.
      if (cur && !f.choices.some((c) => String(c) === cur)) {
        input.append(el('option', { value: cur },
                       cur + ' ' + t('pl.choice_unlisted')));
      }
      input.value = cur;
      read = textTriState();
    } else if (f.type === 'string') {
      input = el('input', { type: 'text', id, disabled,
        placeholder,
        value: present ? (f.current ?? '') : '' });
      // DB-key suggestions: when the schema flags a sub-db, offer its
      // keys as a datalist dropdown. Free text is still allowed (e.g.
      // hf_dir also accepts a filesystem path). Best-effort -- the field
      // stays usable if the keys can't be fetched.
      // The model registry is handled by the compatibility-aware model
      // browser instead (Browse button below), so skip the datalist there.
      if (f.suggest_db && f.suggest_db !== MODEL_REGISTRY_DB) {
        const dlId = id + '_dl';
        datalist = el('datalist', { id: dlId });
        input.setAttribute('list', dlId);
        // When the schema also pins a model_type, ask the server to keep
        // only registry records of that type (e.g. a yolo-only field
        // suggests just yolo models, not every registered model). The
        // schema may list SEVERAL types comma-separated (e.g. a vision
        // tower usable by >1 LM family: "qwen3.5-vision-encoder,
        // gemma4-vision-encoder"); each is queried and the results
        // merged.
        const allTypes = (f.suggest_db_type || '')
          .split(',').map((s) => s.trim()).filter(Boolean);
        // Leading-alpha family token: "qwen3.5-vision-encoder" -> "qwen",
        // "gemma4-vision-encoder" -> "gemma".
        const familyOf = (s) => (String(s).match(/^[a-z]+/i) || [''])[0]
          .toLowerCase();
        // Fill the datalist from the given list of model_types (empty ->
        // no type filter, i.e. every key in the sub-db). De-dupes.
        const fill = (types) => {
          datalist.replaceChildren();
          const queries = types.length
            ? types.map((tp) => ({ db: f.suggest_db,
                value_field: 'model_type', value_equals: tp }))
            : [{ db: f.suggest_db }];
          Promise.all(queries.map(
              (q) => api.dbKeys(q).catch(() => null))).then((rs) => {
            const seen = new Set();
            for (const r of rs) {
              const keys = (r && r.keys) ? r.keys : [];
              for (const k of keys) {
                const v = (k && (k.display ?? k.key)) || '';
                if (v && !seen.has(v)) {
                  seen.add(v);
                  datalist.append(el('option', { value: v }));
                }
              }
            }
          });
        };
        // When more than one type is offered, narrow to the family of
        // the LM chosen in the sibling hf_dir field (its value is the
        // model's HF path, which carries the family name) so a Qwen LM
        // suggests Qwen towers and a Gemma LM Gemma towers. Best-effort:
        // an unrecognised / empty LM shows all; free text always works.
        const refresh = () => {
          if (allTypes.length <= 1) { fill(allTypes); return; }
          const lm = document.getElementById('f_hf_dir');
          const ref = lm && lm.value ? lm.value.toLowerCase() : '';
          const fam = /gemma/.test(ref) ? 'gemma'
                    : /qwen/.test(ref)  ? 'qwen' : '';
          const narrowed = fam
            ? allTypes.filter((tp) => familyOf(tp) === fam) : [];
          fill(narrowed.length ? narrowed : allTypes);
        };
        refresh();
        if (allTypes.length > 1) {
          // The sibling hf_dir input may not exist yet while this field
          // is being built; wire the re-narrow listener after this tick.
          setTimeout(() => {
            const lm = document.getElementById('f_hf_dir');
            if (lm) { lm.addEventListener('input', refresh); }
          }, 0);
        }
      }
      read = textTriState();
    } else if (f.type === 'text') {
      // Multi-line string (backend ConfigType::Text): to the backend this is
      // identical to a `string` field, but the value is multi-line, so render
      // a resizable textarea rather than a single-line input. NOT JSON: read
      // the raw value verbatim (newlines and surrounding whitespace are
      // significant, so don't trim).
      input = el('textarea', { id, disabled, rows: 4, placeholder },
        present ? String(f.current ?? '') : '');
      read = textTriState();     // same unset-vs-empty split as `string`
    } else {
      // array / object / any -> JSON textarea.
      isJson = true;
      input = el('textarea', { id, disabled, placeholder },
        present ? JSON.stringify(f.current ?? null, null, 2) : '');
      // An empty JSON box is not a value either (it does not parse), so blank
      // means unset here too -- no present-but-empty state to distinguish.
      read = () => {
        const s = input.value.trim();
        if (s === '') { return undefined; }
        return JSON.parse(s);
      };
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        disabled, onclick: () => {
          if (input.value.trim() === '') { input.focus(); return; }
          input.value = ''; syncUnset(); clearJsonMark(); input.focus();
          // An empty box always reads cleanly as "unset", never a parse
          // error -- so Clear commits whatever the box held before it,
          // valid or not.
          if (onCommit) { onCommit(); }
        } });
      emptyMeansUnset();
      syncUnset();
    }
    // Filesystem-path fields (backend `is_path`) get a Browse… button
    // that opens the sandbox-aware file dialog. String fields take the
    // single picked path; array/any fields (e.g. load-image `url`) let
    // the user add several, appended into the field's JSON array.
    if (f.is_path && !disabled) {
      const isMulti = f.type !== 'string';
      const openBrowser = () => {
        let seedDir = '';
        if (f.type === 'string') {
          const v = input.value.trim();
          // A field with a model picker too may hold a registry KEY
          // ("org/name"), which is no directory in the sandbox -- seeding
          // from it opens the dialog on an error toast. Seed only from
          // what is plainly a path.
          const pathish = f.suggest_db !== MODEL_REGISTRY_DB
              || v.startsWith('/') || /\.safetensors$/i.test(v);
          seedDir = pathish ? splitPath(v).dir : '';
        } else {
          try {
            const v = JSON.parse(input.value.trim() || '[]');
            const arr = Array.isArray(v) ? v : [v];
            if (arr.length) { seedDir = splitPath(String(arr[arr.length - 1])).dir; }
          } catch (e) { /* leave default */ }
        }
        openFsDialog({
          mode: f.path_write ? 'save' : 'open',
          kind: f.path_kind === 'dir' ? 'dir' : 'file',
          multi: isMulti,
          filters: filterForCategory(f.path_filter),
          startDir: seedDir,
          onPick: (picked) => {
            if (f.type === 'string') {
              input.value = picked;
            } else {
              const paths = Array.isArray(picked) ? picked : [picked];
              let arr = [];
              const cur = input.value.trim();
              if (cur) {
                try {
                  const v = JSON.parse(cur);
                  arr = Array.isArray(v) ? v : [v];
                } catch (e) { arr = [cur]; }
              }
              arr.push(...paths);
              input.value = JSON.stringify(arr, null, 2);
            }
            input.dispatchEvent(new Event('input', { bubbles: true }));
            // Picking a path is a deliberate value commit -- auto-apply like a
            // blur (the programmatic set fires no `change`). Works for the JSON
            // array path fields too, which the `change` listener skips.
            if (onCommit) { onCommit(); }
          },
        });
      };
      browseBtn = el('button', { class: 'btn ghost mini browse',
        type: 'button', title: t('fs.browse'), onclick: openBrowser },
        makeIcon('folder', 'sm'));
    }
    // Model fields (schema `suggest_db` = the model registry) get a
    // compatibility-aware
    // model browser instead of a datalist: it lists only installed models
    // matching the field's model_type(s) AND the parent model chosen in a
    // sibling field. Free text is still allowed in the input.
    //
    // Its OWN button, beside the file browser rather than in place of it.
    // A field may carry both hints -- a LoRA is as often a downloaded
    // .safetensors in the sandbox as a catalogued model -- and the stage
    // resolves either (resolve_adapter_file), so offering one of them
    // would be hiding a working input.
    if (f.suggest_db === MODEL_REGISTRY_DB && f.type === 'string'
        && !disabled) {
      modelBtn = el('button', { class: 'btn ghost mini browse',
        type: 'button', title: t('pl.mb_browse'),
        onclick: () => openModelBrowser(f, input, (val) => {
          input.value = val;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          // Picking a model is a deliberate value commit -- auto-apply like a
          // blur (the programmatic set fires no `change`).
          if (onCommit) { onCommit(); }
        }) },
        makeIcon('database', 'sm'));
    }
    const docTxt = tOr('cfg.' + type + '.' + f.key, f.doc);
    // The field NAME carries its documentation as a tooltip while the
    // lines are hidden, so turning them off costs the reader the space
    // and nothing else. The text is kept on the element because the
    // toggle flips under forms that are already rendered -- applyDocs
    // reads it back rather than rebuilding them.
    const keySpan = el('span', { class: 'key' }, f.key);
    if (docTxt) {
      keySpan.dataset.doc = docTxt;
      if (!showDocs) { keySpan.title = docTxt; }
    }
    // A CHECKBOX is one small control and a small button, which is a
    // whole row of panel for almost no ink -- so a bool field puts its
    // name, its box and its unset button on ONE line (see .field-bool),
    // name left and controls right. The type hint goes with it: "bool"
    // beside a checkbox says nothing the checkbox has not already said.
    const isBool = f.type === 'bool';
    const label = el('label', { for: id },
      keySpan,
      f.required ? el('span', { class: 'req' }, '*') : null,
      isBool ? null : el('span', { class: 'ty' }, f.type));
    const inputRow = el('div', { class: 'field-input-row' }, input);
    if (datalist) { inputRow.append(datalist); }
    if (modelBtn) { inputRow.append(modelBtn); }
    if (browseBtn) { inputRow.append(browseBtn); }
    if (unsetBtn) { inputRow.append(unsetBtn); }
    const field = el('div', { class: 'field' + (isBool ? ' field-bool' : '') },
                     label, inputRow);
    if (docTxt) { field.append(el('div', { class: 'doc' }, docTxt)); }
    // Auto-apply when a text/number box loses focus after an edit. `change`
    // fires on blur-after-edit (and Enter), so tabbing through untouched
    // fields costs nothing; onCommit self-gates on the pipeline being
    // stopped. `text` is a plain-string textarea that never throws on read,
    // so it commits on blur like `string`.
    if (onCommit && (f.type === 'string' || f.type === 'text'
        || f.type === 'int' || f.type === 'uint' || f.type === 'real')) {
      input.addEventListener('change', () => onCommit());
    } else if (onCommit && isJson) {
      // JSON TEXTAREAS COMMIT ON BLUR TOO, once the box parses.
      //
      // They were excluded because a half-typed blob throws on read, and
      // the exclusion was worse than the throw it avoided. applyConfig
      // reads EVERY input, so a hand-edit here was not merely
      // uncommitted -- it rode out on whatever committed next. Blur a
      // neighbouring string field, or pick a path with Browse, and the
      // edit went with it; touch nothing else and switch stage, and it
      // was discarded without a word. That is the unpredictable part:
      // not that the edit waited, but that WHEN it landed was decided by
      // an unrelated field. The same reading is why a half-typed box is
      // not harmless either -- it makes the neighbour's commit fail too,
      // and the neighbour is where the error surfaces.
      //
      // So: parse first, commit only on success, and when it does not
      // parse SAY SO on the box itself. A toast on every blur is what
      // the old comment was right to avoid; a silent no-op is what sent
      // the user looking for the bug in the stage.
      //
      // THE PHONE SHEET HAS THIS RULE TOO, in phone/phone-config.js --
      // it renders its own controls, so the rule is copied rather than
      // shared, and a value that applies on one shell and not the other
      // is the drift this directory has already seen once. Change both,
      // and phone-config.test.mjs pins that copy.
      const markBad = (bad) => {
        input.classList.toggle('field-invalid', bad);
        if (bad) { input.title = t('pl.json_invalid'); }
        else { input.removeAttribute('title'); }
      };
      // Typing is the fix in progress -- drop the mark rather than leave
      // it nagging through every keystroke of a correction. Browse also
      // fires `input` before it commits, so a pick clears a stale mark.
      clearJsonMark = () => markBad(false);
      input.addEventListener('input', () => markBad(false));
      input.addEventListener('change', () => {
        try {
          read();
        } catch (e) {
          markBad(true);
          return;
        }
        markBad(false);
        onCommit();
      });
    }
    return { field, read };
  }

  // `opts.rerender === false` skips rebuilding the config pane after a
  // successful apply -- used by the auto-apply-on-blur path so committing one
  // field doesn't tear down (and steal focus from) the field the user just
  // tabbed into. The Apply button leaves it at the default (rebuild).
  async function applyConfig(inputs, opts) {
    const rerender = !(opts && opts.rerender === false);
    const cfg = {};
    try {
      for (const it of inputs) {
        const v = it.read();
        // `undefined` means "field intentionally left unset": omit from
        // POST so the stage's validator sees a clean partial config.
        // Required for stages with mutually-exclusive keys (chrono).
        if (v === undefined) { continue; }
        if (typeof v === 'number' && Number.isNaN(v)) {
          throw new Error(t('pl.invalid_number', { key: it.key }));
        }
        cfg[it.key] = v;
      }
    } catch (e) {
      toast(t('pl.bad_config', { msg: e.message }), 'error');
      return;
    }
    const sid = state.selectedStage;
    try {
      state.detail = await api.setStageConfig(
        state.selectedId, sid, cfg);
      // The pair, recorded only once the server has taken the new config:
      // an apply that failed changed nothing and must not become the step
      // Undo walks back to.
      if (cfgLast && cfgLast.sid === sid) {
        cfgUndo = { sid, before: cfgLast.cfg, after: cfg, undone: false };
      }
      cfgLast = { sid, cfg };
      syncUndoBtns();
      toast(t('pl.config_applied'), 'ok');
      // `rerender:false` is a promise not to tear down the form the
      // user is still in, and rebuilding the GRAPH broke that promise
      // in the one-column layout -- where the form lives inside
      // graphBody, so clearing it takes the form with it. The graph is
      // marked stale instead and redrawn when focus leaves the form.
      if (rerender) {
        state.graphStale = false;
        renderGraphPane();
        await renderConfig();
      } else {
        state.graphStale = true;
      }
    } catch (e) { toast(t('pl.apply_failed', { msg: e.message }), 'error'); }
  }

  // Which half of the pair is reachable from here. Read from the slot
  // rather than tracked beside it, so the buttons cannot disagree with
  // what pressing them would actually do.
  function syncUndoBtns() {
    if (!undoBtns) { return; }
    const pair = !!cfgUndo && cfgUndo.sid === state.selectedStage
                 && canEdit();
    undoBtns.undo.disabled = !(pair && !cfgUndo.undone);
    undoBtns.redo.disabled = !(pair && cfgUndo.undone);
  }

  // Walk the single undo pair, in whichever direction it is pointing.
  // Undo and Redo are the same move: PUT the other half of the pair and
  // flip the arrow, so a user who undid by accident can put it straight
  // back. The slot survives the re-render below -- it is keyed by stage,
  // and renderConfig only reseeds `cfgLast`.
  async function stepConfigUndo() {
    if (!cfgUndo || cfgUndo.sid !== state.selectedStage) { return; }
    if (!canEdit()) { toast(t('pl.stop_to_edit'), 'error'); return; }
    const back = !cfgUndo.undone;
    const target = back ? cfgUndo.before : cfgUndo.after;
    try {
      state.detail = await api.setStageConfig(
        state.selectedId, cfgUndo.sid, target);
      cfgUndo.undone = back;
      cfgLast = { sid: cfgUndo.sid, cfg: target };
      toast(t(back ? 'pl.config_undone' : 'pl.config_redone'), 'ok');
      renderGraphPane();
      await renderConfig();
    } catch (e) {
      toast(t('pl.undo_failed', { msg: e.message }), 'error');
    }
  }

  // --- toolbar actions ----------------------------------------------
  function onCreate() {
    const idIn = el('input', { type: 'text',
      placeholder: t('pl.pipeline_id') });
    openModal({
      title: t('pl.create_title'),
      body: el('div', {}, el('label', { class: 'fl' }, t('pl.pipeline_id')),
        idIn),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.create'), kind: 'primary', onClick: async (c) => {
            const id = idIn.value.trim();
            if (!id) { toast(t('pl.id_required'), 'error'); return; }
            try {
              await api.createPipeline(id);
              c(); state.selectedId = id; await refreshList();
            } catch (e) {
              toast(t('pl.create_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
    setTimeout(() => idIn.focus(), 0);
  }

  function onRenamePipeline() {
    if (!state.selectedId) { return; }
    if (!canEdit()) { toast(t('pl.rename_pl_stopped'), 'error'); return; }
    const old = state.selectedId;
    const idIn = el('input', { type: 'text', value: old });
    openModal({
      title: t('pl.rename_pl_title'),
      body: el('div', {}, el('label', { class: 'fl' }, t('pl.new_id')), idIn),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.rename'), kind: 'primary', onClick: async (c) => {
            const to = idIn.value.trim();
            if (!to) { toast(t('pl.id_required'), 'error'); return; }
            if (to === old) { c(); return; }
            try {
              await api.renamePipeline(old, to);
              c();
              state.selectedId = to;   // stage positions/pins survive
              await refreshList();
              toast(t('pl.pl_renamed', { from: old, to }), 'ok');
            } catch (e) {
              toast(t('pl.rename_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
    setTimeout(() => { idIn.focus(); idIn.select(); }, 0);
  }

  // Append ".vpipeline" when the basename has no extension. Free-form
  // paths with any other extension pass through untouched, satisfying
  // the "still okay to specify a non-.vpipeline path" requirement.
  function withDefaultPipelineExt(p) {
    if (!p) { return p; }
    const slash = p.lastIndexOf('/');
    const base = slash >= 0 ? p.slice(slash + 1) : p;
    // Dotfiles (".hidden") and basenames with no dot at all are both
    // treated as "no extension supplied" -> append default.
    if (base.indexOf('.') === -1) { return p + '.vpipeline'; }
    return p;
  }

  // Extension filter for pipeline-spec files, shared by load + save.
  const PIPELINE_FILTER = [{ label: t('pl.vpipeline_filter'),
    exts: ['.vpipeline'] }];

  function onLoad() {
    const seed = splitPath(state.detail ? state.detail.storage_path : '');
    openFsDialog({
      mode: 'open', kind: 'file', filters: PIPELINE_FILTER,
      title: t('pl.load_title'),
      startDir: seed.dir,
      onPick: async (p) => {
        if (!p) { return; }
        try {
          const d = await api.loadPipeline(p);
          state.selectedId = d.id; await refreshList();
        } catch (e) {
          // A spec the loader refused says exactly what it objected to
          // and where. That belongs in front of whoever picked the
          // file, not in a toast that expires before it can be read.
          openErrorModal({
            title: t('pl.load_failed_title'),
            message: e.message,
            okLabel: t('common.close'),
          });
        }
      },
    });
  }

  function onSave() {
    if (!state.selectedId) { return; }
    const seed = splitPath(state.detail ? state.detail.storage_path : '');
    openFsDialog({
      mode: 'save', kind: 'file', filters: PIPELINE_FILTER,
      title: t('pl.save_title', { id: state.selectedId }),
      startDir: seed.dir,
      defaultName: seed.name || (state.selectedId + '.vpipeline'),
      onPick: async (p) => {
        try {
          const r = await api.savePipeline(state.selectedId,
            withDefaultPipelineExt(p));
          toast(t('pl.saved', { path: r.storage_path }), 'ok');
          await loadDetail(state.selectedId);
        } catch (e) {
          toast(t('pl.save_failed', { msg: e.message }), 'error');
        }
      },
    });
  }

  function onUnload() {
    if (!state.selectedId) { return; }
    const id = state.selectedId;
    openModal({
      title: t('pl.unload_title'),
      body: el('div', {}, t('pl.unload_confirm', { id })),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.unload'), kind: 'danger', onClick: async (c) => {
            try { await api.unloadPipeline(id); c(); await refreshList(false); }
            catch (e) {
              toast(t('pl.unload_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
  }

  // --- remove stage (driven from the config panel + Delete key) ----
  function onRemoveStage() {
    if (!state.detail || state.detail.state !== 'stopped') { return; }
    if (!state.selectedStage) { return; }
    const sid = state.selectedStage;
    openModal({
      title: t('pl.remove_stage_modal'),
      body: el('div', {}, t('pl.remove_stage_confirm', { id: sid })),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.remove'), kind: 'danger', onClick: async (c) => {
            try {
              state.detail = await api.removeStage(state.selectedId, sid);
              c();
              state.graphPins.delete(sid);   // drop any stale drop-pin
              state.selectedStage = null;
              renderGraphPane();
              await renderConfig();
              await refreshList();
              toast(t('pl.removed', { id: sid }), 'ok');
            } catch (e) {
              toast(t('pl.remove_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
  }

  // Right-click a stage -> a context menu (rename / delete). Selects the
  // stage first so the menu clearly targets it (and the config pane
  // reflects it).
  function onNodeContext(sid, x, y) {
    if (!canEdit()) { toast(t('pl.select_stopped'), 'error'); return; }
    selectStage(sid);
    openMenu(x, y, [
      { label: t('pl.rename_stage'), onClick: () => promptRenameStage(sid) },
      { label: t('pl.duplicate_stage'), onClick: () => duplicateStage(sid) },
      null,
      { label: t('common.remove'), danger: true,
        onClick: () => { state.selectedStage = sid; onRemoveStage(); } },
    ]);
  }

  // Duplicate a stage's settings under a server-generated, non-colliding
  // id ("<sid>-N"). The copy has no connections; place it just below the
  // source so it's easy to find, then select it.
  async function duplicateStage(sid) {
    if (!canEdit()) { toast(t('pl.select_stopped'), 'error'); return; }
    try {
      const resp = await api.duplicateStage(state.selectedId, sid);
      state.detail = resp;
      const nid = resp.stage;
      // Pin the copy just under the source (same column) so it doesn't
      // scatter to an auto-slot far from its original. A layout entry
      // keys its column as `.rank`; the pin's `y` is only an ORDERING
      // seed (the packer normalizes the real gap), so any positive delta
      // over the source's y lands the copy directly after it.
      const at = state.graphLayout && state.graphLayout.get(sid);
      if (at) {
        state.graphPins.set(nid, { col: at.rank || 0, y: at.y + 1 });
      }
      state.selectedStage = nid;
      state.pending = null;
      state.selectedEdge = null;
      renderGraphPane();
      await renderConfig();
      await refreshList();
      toast(t('pl.stage_duplicated', { from: sid, to: nid }), 'ok');
    } catch (e) {
      toast(t('pl.duplicate_failed', { msg: e.message }), 'error');
    }
  }

  function promptRenameStage(sid) {
    if (!canEdit()) { toast(t('pl.select_stopped'), 'error'); return; }
    const idIn = el('input', { type: 'text', value: sid });
    openModal({
      title: t('pl.rename_stage_title'),
      body: el('div', {}, el('label', { class: 'fl' }, t('pl.new_id')), idIn),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.rename'), kind: 'primary', onClick: async (c) => {
            const to = idIn.value.trim();
            if (!to) { toast(t('pl.id_required'), 'error'); return; }
            if (to === sid) { c(); return; }
            try {
              await api.renameStage(state.selectedId, sid, to);
              c();
              // Carry the stage's manual placement + selection to the new id
              // so it doesn't jump on the re-layout.
              if (state.graphPins.has(sid)) {
                state.graphPins.set(to, state.graphPins.get(sid));
                state.graphPins.delete(sid);
              }
              if (state.graphLayout && state.graphLayout.get(sid)) {
                state.graphLayout.set(to, state.graphLayout.get(sid));
                state.graphLayout.delete(sid);
              }
              if (state.selectedStage === sid) { state.selectedStage = to; }
              await refreshList();
              toast(t('pl.stage_renamed', { from: sid, to }), 'ok');
            } catch (e) {
              toast(t('pl.rename_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
    setTimeout(() => { idIn.focus(); idIn.select(); }, 0);
  }

  // --- boot ---------------------------------------------------------
  (async () => {
    try { state.stageTypes = await api.stageTypes(); }
    catch (e) {
      toast(t('pl.stage_types_failed', { msg: e.message }), 'error');
    }
    renderToolbox();
    await refreshList();
    scheduleBufferPoll();   // start the buffer-utilization overlay loop
  })();

  // Re-arm hook: a host that keeps this editor alive across nav switches
  // (the composer) calls onShow() when it re-attaches the DOM, to RESUME
  // the buffer-fullness overlay. The poll loop and the keyboard listener
  // self-stop on detach (see scheduleBufferPoll / onShortcut), so away/back
  // is seamless -- this just restarts the loop and re-syncs the shown
  // state. Idempotent; a no-op while detached. cleanup() tears both down
  // promptly when the host removes the panel.
  function onShow() {
    if (!document.body.contains(pmRoot)) { return; }
    document.removeEventListener('keydown', onShortcut);
    document.addEventListener('keydown', onShortcut);
    refreshList();          // re-sync pipeline state + rebuild the overlay
    scheduleBufferPoll();   // resume the poll loop
  }
  function cleanup() {
    stopBufferPoll();
    document.removeEventListener('keydown', onShortcut);
  }
  return { onShow, cleanup };
}
