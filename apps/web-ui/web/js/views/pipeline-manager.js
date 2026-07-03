// Pipeline Manager view: three panes -- pipeline list (1/4), stage
// graph (1/2), and the config panel (1/4).

import { el, clear, append, toast, openModal, kbd } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { renderGraph, applyBufferStats, shortType } from '../graph.js';
import { t, tOr } from '../i18n.js';

export function mountPipelineManager(container) {
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
    graphContainer: null, // live element from renderGraph (overlay target)
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
  };

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
  const btnSave   = iconBtn('save',  t('common.save'),   () => onSave(),   'S');
  const btnUnload = iconBtn('trash', t('common.unload'), () => onUnload(), 'U');

  const listPane = el('div', { class: 'pane' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('nav.pipelines'))),
    el('div', { class: 'toolbar' }, btnCreate, btnLoad, btnSave, btnUnload),
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
  graphHead.append(
    el('span', { class: 'title' }, t('pl.stages')),
    el('span', { class: 'grow' }),
    stagesCount);

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
  // auto-layout positions the new node, so the drop point isn't used.
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
      promptCreateStage(dropType);
    }
  });

  const cfgPane = el('div', { class: 'pane' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('pl.configuration'))),
    cfgBody);

  const pmRoot = el('div', { class: 'pm' }, listPane, graphPane, cfgPane);
  clear(container).append(pmRoot);

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
    try {
      state.pipelines = await api.listPipelines();
    } catch (e) { toast(t('pl.list_failed', { msg: e.message }), 'error');
      return; }
    if (!keepSel || !state.pipelines.find((p) => p.id === state.selectedId)) {
      state.selectedId = state.pipelines[0] ? state.pipelines[0].id : null;
      state.detail = null;
      state.selectedStage = null;
      state.graphView = {};   // refit when the selected pipeline changes
    }
    renderList();
    if (state.selectedId) { await loadDetail(state.selectedId); }
    else { renderGraphPane(); renderConfig(); }
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
    state.selectedStage = null;
    state.pending = null;
    state.selectedEdge = null;
    state.graphView = {};   // refit the graph for the newly-selected pipeline
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
    btnSave.disabled = btnUnload.disabled = !state.selectedId;
  }

  function actIcon(icon, title, enabled, onclick) {
    return el('button', {
      class: 'icon-action ' + icon, title, disabled: !enabled,
      onclick: enabled ? onclick : null,
    }, makeIcon(icon, 'sm'));
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
      toast(t('pl.op_failed', { op: opLabel[op] || op, msg: e.message }),
            'error');
    } finally {
      state.inflight.delete(id);
    }
    await refreshList();
  }

  // --- middle pane --------------------------------------------------
  function renderGraphPane() {
    clear(graphBody);
    state.graphContainer = null;   // old overlay target is gone
    const editable = canEdit();
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
    } else {
      const gc = renderGraph(g, {
        selected: state.selectedStage,
        onSelect: (sid) => selectStage(sid),
        view: state.graphView,
        editable,
        pending: state.pending,
        pendingType: state.pending
          ? portType(nodeById(state.pending.from), 'out',
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

  // --- composer: toolbox + drag-create + click-to-connect ----------
  function canEdit() {
    return !!(state.detail && state.detail.state === 'stopped');
  }

  // Category display order + labels for the toolbox sections.
  const CATEGORY_ORDER = ['preparation', 'video', 'audio', 'text', 'network',
                          'control', 'database', 'generic'];

  function stageChip(s) {
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

  function renderToolbox() {
    clear(toolboxBody);
    const q = (toolboxFilter.value || '').toLowerCase();
    const match = state.stageTypes.filter((s) => !s.hidden && (
      s.type.toLowerCase().includes(q)
      || (s.doc || '').toLowerCase().includes(q)
      || (s.category || '').toLowerCase().includes(q)));
    // Bucket by category, then render in CATEGORY_ORDER (unknown
    // categories fall to the end, alphabetically).
    const byCat = new Map();
    for (const s of match) {
      const c = s.category || 'generic';
      if (!byCat.has(c)) { byCat.set(c, []); }
      byCat.get(c).push(s);
    }
    const cats = [...byCat.keys()].sort((a, b) => {
      const ia = CATEGORY_ORDER.indexOf(a);
      const ib = CATEGORY_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    for (const c of cats) {
      const items = byCat.get(c).sort((a, b) => a.type.localeCompare(b.type));
      toolboxBody.append(el('div', { class: 'tb-cat' },
        tOr('cat.' + c, c) + ' (' + items.length + ')'));
      for (const s of items) { toolboxBody.append(stageChip(s)); }
    }
    if (match.length === 0) {
      toolboxBody.append(el('div', { class: 'tb-empty' }, t('pl.no_matches')));
    }
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
  function typesCompatible(a, b) {
    return !a || !b || a === 'any' || b === 'any' || a === b;
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

  function promptCreateStage(type) {
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
    // kind === 'in': complete the connection (re-point this input, or
    // fill the next free slot). Inputs are stored densely and fill in
    // index order, so a click on a slot past the next free one is
    // refused with a hint rather than erroring at the server.
    if (!state.pending) {
      toast(t('pl.wire_output_first'), 'info');
      return;
    }
    const node = nodeById(stageId);
    const liveCount = node ? node.iports.length : 0;
    if (idx > liveCount) {
      toast(t('pl.wire_in_order', { n: liveCount }), 'info');
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

  async function selectStage(sid) {
    state.selectedStage = sid;
    state.selectedEdge = null;
    state.pending = null;
    renderGraphPane();
    await renderConfig();
  }

  // --- right pane (config) -----------------------------------------
  async function renderConfig() {
    clear(cfgBody);
    if (!state.detail || !state.selectedStage) {
      cfgBody.append(el('div', { class: 'cfg' },
        el('div', { class: 'empty' }, t('pl.select_stage_config'))));
      return;
    }
    let info;
    try {
      info = await api.getStageConfig(state.selectedId, state.selectedStage);
    } catch (e) {
      cfgBody.append(el('div', { class: 'cfg' },
        el('div', { class: 'empty' },
          t('pl.config_unavailable', { msg: e.message }))));
      return;
    }
    const editable = !!info.editable;
    const wrap = el('div', { class: 'cfg' });
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
    const inputs = [];
    for (const f of info.schema) {
      const { field, read } = configField(f, !editable, info.type);
      inputs.push({ key: f.key, type: f.type, read });
      wrap.append(field);
    }
    const applyBtn = el('button', {
      class: 'btn primary', disabled: !editable,
      onclick: () => applyConfig(inputs),
    }, t('common.apply'));
    // Remove the stage (topology edit -- stopped pipelines only). Lives
    // here now that the Stages title bar has no Remove button; pushed to
    // the right so it's clearly separated from Apply.
    const removeBtn = el('button', {
      class: 'btn danger', disabled: !editable,
      style: 'margin-left:auto',
      title: editable ? t('pl.remove_stage_title')
                      : t('pl.stop_to_edit'),
      onclick: () => onRemoveStage(),
    }, makeIcon('trash', 'sm'), el('span', {}, t('common.remove')));
    wrap.append(el('div', { class: 'cfg-actions' }, applyBtn, removeBtn));
    cfgBody.append(wrap);
  }

  // Build one config field. `read()` returns:
  //   - undefined  -> "field intentionally unset, omit from POST"
  //   - any other  -> value to send under f.key
  //
  // Numeric and string fields render EMPTY when the backend reports
  // `present === false`, with the schema default in the placeholder so
  // the user knows what the value will resolve to if left blank. This
  // lets the user wipe a field (e.g. chrono's frequency_hz) so the
  // mutually-exclusive period_* set is the only thing that arrives at
  // the stage's validator. Bool toggles are tri-state via a small
  // "unset" affordance to support the same pattern.
  //
  // Backward-compat: older backends predate the `present` flag. When
  // it's absent we fall back to "show whatever current is" so the
  // editor doesn't strand the user with empty inputs against an
  // un-upgraded server.
  function configField(f, disabled, type) {
    const id = 'f_' + f.key;
    const placeholder = f.default !== undefined && f.default !== null
        ? t('pl.field_default', { val: typeof f.default === 'object'
            ? JSON.stringify(f.default) : String(f.default) })
        : t('pl.unset');
    const present = f.present !== undefined
        ? !!f.present
        : (f.current !== undefined && f.current !== null);
    let input, read, unsetBtn = null, datalist = null;
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
        disabled, onclick: () => { unset = !unset; updateUi(); } });
      updateUi();
      read = () => (unset ? undefined : input.checked);
    } else if (f.type === 'int' || f.type === 'uint' || f.type === 'real') {
      input = el('input', { type: 'number', id, disabled,
        step: f.type === 'real' ? 'any' : '1',
        placeholder,
        value: present ? (f.current ?? '') : '' });
      read = () => {
        const s = input.value.trim();
        if (s === '') { return undefined; }
        const n = f.type === 'real' ? parseFloat(s) : parseInt(s, 10);
        return Number.isNaN(n) ? NaN : n;
      };
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        title: t('pl.clear_omit'),
        disabled, onclick: () => { input.value = ''; input.focus(); } },
        t('pl.btn_clear'));
    } else if (f.type === 'string') {
      input = el('input', { type: 'text', id, disabled,
        placeholder,
        value: present ? (f.current ?? '') : '' });
      // DB-key suggestions: when the schema flags a sub-db, offer its
      // keys as a datalist dropdown. Free text is still allowed (e.g.
      // hf_dir also accepts a filesystem path). Best-effort -- the field
      // stays usable if the keys can't be fetched.
      if (f.suggest_db) {
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
      read = () => {
        const s = input.value;
        // Empty input -> omit so the stage falls back to its declared
        // default. Stages that legitimately need an explicit empty
        // string can declare a non-empty default and use the JSON
        // textarea (any-typed) path -- we don't try to round-trip ""
        // through the convenience text input.
        return s === '' ? undefined : s;
      };
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        title: t('pl.clear_omit'),
        disabled, onclick: () => { input.value = ''; input.focus(); } },
        t('pl.btn_clear'));
    } else {
      // array / object / any -> JSON textarea.
      input = el('textarea', { id, disabled, placeholder },
        present ? JSON.stringify(f.current ?? null, null, 2) : '');
      read = () => {
        const s = input.value.trim();
        if (s === '') { return undefined; }
        return JSON.parse(s);
      };
      unsetBtn = el('button', { class: 'btn ghost mini', type: 'button',
        title: t('pl.clear_omit'),
        disabled, onclick: () => { input.value = ''; input.focus(); } },
        t('pl.btn_clear'));
    }
    const label = el('label', { for: id },
      el('span', { class: 'key' }, f.key),
      f.required ? el('span', { class: 'req' }, '*') : null,
      el('span', { class: 'ty' }, f.type));
    const inputRow = el('div', { class: 'field-input-row' }, input);
    if (datalist) { inputRow.append(datalist); }
    if (unsetBtn) { inputRow.append(unsetBtn); }
    const field = el('div', { class: 'field' }, label, inputRow);
    const docTxt = tOr('cfg.' + type + '.' + f.key, f.doc);
    if (docTxt) { field.append(el('div', { class: 'doc' }, docTxt)); }
    return { field, read };
  }

  async function applyConfig(inputs) {
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
    try {
      state.detail = await api.setStageConfig(
        state.selectedId, state.selectedStage, cfg);
      toast(t('pl.config_applied'), 'ok');
      renderGraphPane();
      await renderConfig();
    } catch (e) { toast(t('pl.apply_failed', { msg: e.message }), 'error'); }
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

  async function onLoad() {
    // Pull the cwd file list up front so the input's <datalist> is
    // populated before the user starts typing. Best-effort: if the
    // endpoint isn't there (older server, fetch error) the dialog
    // still works for hand-typed paths.
    let cwd = '';
    let files = [];
    try {
      const r = await api.cwdPipelines();
      cwd = r && typeof r.cwd === 'string' ? r.cwd : '';
      files = r && Array.isArray(r.files) ? r.files : [];
    } catch (e) { /* leave defaults */ }

    // <datalist> + input[list=...] gives native browser autocomplete:
    // the suggestions filter as the user types, the browser handles
    // arrow-key selection and Enter-to-fill, and free-form paths are
    // still accepted (the datalist is hint-only, not a constraint).
    const listId = 'pl-load-list';
    const dlist = el('datalist', { id: listId });
    for (const f of files) { dlist.append(el('option', { value: f })); }

    const pathIn = el('input', { type: 'text',
      placeholder: files.length
        ? files[0]
        : '/path/to/pipeline.vpipeline',
      autocomplete: 'off', list: listId });

    const hint = cwd
      ? el('div', { class: 'fl-hint' },
          files.length
            ? t('pl.vpipeline_files',
                { n: files.length, s: files.length === 1 ? '' : 's', cwd })
            : t('pl.no_vpipeline', { cwd }))
      : null;

    openModal({
      title: t('pl.load_title'),
      body: el('div', {},
        el('label', { class: 'fl' }, t('pl.file_path')),
        pathIn, dlist, hint),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.load'), kind: 'primary', onClick: async (c) => {
            const p = pathIn.value.trim();
            if (!p) { toast(t('pl.path_required'), 'error'); return; }
            try {
              const d = await api.loadPipeline(p);
              c(); state.selectedId = d.id; await refreshList();
            } catch (e) {
              toast(t('pl.load_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
    setTimeout(() => pathIn.focus(), 0);
  }

  function onSave() {
    if (!state.selectedId) { return; }
    const cur = state.detail ? state.detail.storage_path : '';
    const pathIn = el('input', { type: 'text', value: cur || '',
      placeholder: state.selectedId + '.vpipeline' });
    openModal({
      title: t('pl.save_title', { id: state.selectedId }),
      body: el('div', {},
        el('label', { class: 'fl' },
          t('pl.save_path_label')),
        pathIn,
        el('div', { class: 'fl-hint' },
          t('pl.save_hint'))),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.save'), kind: 'primary', onClick: async (c) => {
            try {
              const r = await api.savePipeline(state.selectedId,
                withDefaultPipelineExt(pathIn.value.trim()));
              c(); toast(t('pl.saved', { path: r.storage_path }), 'ok');
              await loadDetail(state.selectedId);
            } catch (e) {
              toast(t('pl.save_failed', { msg: e.message }), 'error');
            }
          } },
      ],
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
}
