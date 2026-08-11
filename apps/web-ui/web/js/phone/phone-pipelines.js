// Phone pipeline view.
//
// A phone is a poor place to WIRE a graph (there is no room for a canvas
// and no precision for a port), so the topology is presented rather than
// drawn: the DAG is LINEARIZED to one row per stage, with the connections
// in a left gutter exactly the way `git log --graph --oneline` draws
// commit parents. That form survives a narrow column -- one text line per
// node, a few pixels of lane per concurrent branch, however deep the
// graph gets -- and the reading is already familiar.
//
// Tapping a stage EXPANDS its row into a block in place rather than
// opening a sheet over the list, so the graph stays on screen and the
// stage keeps its position in it. The gutter is continued down the side
// of the expanded block (see contLanes) -- without that the lane running
// past the stage would appear to stop dead at whichever row happened to
// be open.
//
// What the phone can change: a stage's CONFIGURATION, the pipeline's
// run state (start / pause / stop), its name, and where it is saved.
// What it can't: the topology -- adding, removing, connecting or
// disconnecting stages still needs the desktop canvas, which is the part
// that genuinely doesn't fit. Config edits, like on the desktop, are
// accepted only while the pipeline is STOPPED; the backend reports that
// per stage as `editable` and the fields go read-only when it is false.

import { el, clear, toast, openModal, openMenu } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t, tOr } from '../i18n.js';
import { openFsSheet, dirOf, baseOf } from './phone-fs.js';
import { configField, readConfig } from './phone-config.js';
import { topoOrder, assignLanes, laneArt, laneX, gutterWidth }
  from '../lane-graph.js';
import { openConnectEditor } from '../connect-editor.js';
import { lastPipeline, setLastPipeline } from './phone-recent.js';

// How often the read-only view re-reads state other clients can change.
const POLL_MS = 3000;

export function mountPhonePipelines({ body, actions, setTitle }) {
  clear(body);
  clear(actions);

  const state = {
    pipelines: [],
    selectedId: null,
    detail: null,
    specs: null,        // stage-type specs, for the per-stage doc text
    openStage: null,    // the expanded stage id, if any
    inflight: null,     // lifecycle op in progress ('launch'|'pause'|'stop')
  };
  let stopped = false;
  let lastRowsKey = null;     // guards the row rebuild (see render())
  // Row elements by stage id, so expanding one is DOM surgery rather
  // than a rebuild -- a rebuild collapses the scroll container and
  // strands the reader at the top.
  const rowEls = new Map();
  let openBlock = null;

  // ---- chrome -------------------------------------------------------
  const pickBtn = el('button', { class: 'ph-pick',
    onclick: () => openPipelineSheet() });
  const bar = el('div', { class: 'ph-bar' }, pickBtn);
  const listEl = el('div', { class: 'ph-rows' });
  const scroll = el('div', { class: 'ph-scroll' }, listEl);
  body.append(bar, scroll);

  const act = (icon, label, fn) => el('button', {
    class: 'ph-act', title: label, 'aria-label': label, onclick: fn,
  }, makeIcon(icon, 'sm'));
  const btnPlay  = act('play',  t('common.start'), () => lifecycle('launch'));
  const btnPause = act('pause', t('common.pause'), () => lifecycle('pause'));
  const btnStop  = act('stop',  t('common.stop'),  () => lifecycle('stop'));
  const btnSave  = act('save',  t('common.save'),  () => onSave());
  // The rest live behind an overflow menu: seven controls do not fit a
  // phone header, and these are the ones that aren't reached mid-task.
  const btnMore = el('button', { class: 'ph-act', title: t('phone.more'),
    'aria-label': t('phone.more'),
    onclick: (e) => openMoreMenu(e.currentTarget) }, '⋯');
  actions.append(btnPlay, btnPause, btnStop, btnSave, btnMore);

  function openMoreMenu(anchor) {
    const r = anchor.getBoundingClientRect();
    openMenu(r.right, r.bottom + 4, [
      { label: t('common.load') + '…', onClick: () => onLoad() },
      { label: t('phone.save_as') + '…', onClick: () => onSaveAs() },
      { label: t('common.rename') + '…', onClick: () => onRename() },
      null,
      { label: t('common.refresh'), onClick: () => refresh() },
      null,
      // Destructive, so it sits behind the overflow, last, below a
      // divider and marked -- and then still asks. Unloading is not
      // gated on the run state (the backend dematerializes a running
      // pipeline just as the desktop lets it), which is exactly why
      // the confirmation names what is about to go.
      { label: t('common.unload') + '…', danger: true,
        onClick: () => onUnload() },
    ]);
  }

  // ---- data ---------------------------------------------------------
  async function refresh() {
    let list;
    try { list = await api.listPipelines(); }
    catch (e) { return; }                    // transient -- next tick
    if (stopped) { return; }
    state.pipelines = list || [];
    if (!state.pipelines.find((p) => p.id === state.selectedId)) {
      // Nothing selected, or what was selected is gone. Prefer the one
      // this phone last looked at -- across a view switch, a reload, or
      // a server restart, coming back to the pipeline you were working
      // on beats coming back to whichever one sorts first. It is only a
      // preference: if it is no longer loaded, fall through.
      const want = state.pipelines.find((p) => p.id === lastPipeline());
      state.selectedId = want ? want.id
        : (state.pipelines[0] ? state.pipelines[0].id : null);
      state.detail = null;
      state.openStage = null;
    }
    // Record what is actually selected, wherever it came from -- an
    // explicit pick, a load, a rename, or the fallback above.
    setLastPipeline(state.selectedId);
    if (state.selectedId) {
      try { state.detail = await api.getPipeline(state.selectedId); }
      catch (e) { state.detail = null; }
    }
    if (stopped) { return; }
    render();
  }

  // Stage specs are only needed for the detail block's description text,
  // so they are fetched once, lazily, and their absence is survivable.
  async function specs() {
    if (state.specs) { return state.specs; }
    try { state.specs = await api.stageTypes(); }
    catch (e) { state.specs = []; }
    return state.specs;
  }

  const pipelineState = () => (state.detail ? state.detail.state : null);
  const canEdit = () => pipelineState() === 'stopped';

  // Tapping the lane art opens the connection editor for that node. The
  // specs are awaited FIRST rather than passed as a promise: without
  // them the editor can only offer the ports already wired, so a stage
  // whose inputs are mostly unconnected would look like it had none to
  // pick. They are cached after the first call, so this is a round trip
  // once per session.
  async function openConnections(node) {
    const list = await specs();
    openConnectEditor({
      pipelineId: state.selectedId,
      node,
      graph: (state.detail && state.detail.graph) || { nodes: [], edges: [] },
      specFor: (ty) => (list || []).find((s) => s.type === ty) || null,
      editable: canEdit(),
      onChanged: (detail) => {
        state.detail = detail;
        renderRows();
      },
    });
  }

  // ---- render -------------------------------------------------------
  function render() {
    renderChrome();
    if (rowsKey() === lastRowsKey) { return; }
    // A rebuild would throw away a config form the user is filling in,
    // so while a stage is expanded the poll's redraw is DEFERRED: the
    // key is left unrecorded, and collapse() takes it once the form is
    // out of the way.
    if (state.openStage) { return; }
    renderRows();
  }

  function renderChrome() {
    const st = pipelineState();
    const busy = !!state.inflight;
    setTitle(t('nav.pipelines'));
    clear(pickBtn);
    if (state.selectedId) {
      pickBtn.append(
        el('span', { class: 'ph-pick-id' }, state.selectedId),
        el('span', { class: 'ph-pill ' + (busy ? 'busy' : (st || '')) },
           busy ? t('phone.busy_' + state.inflight) : (st || '')),
        el('span', { class: 'ph-pick-caret' }, '▾'));
    } else {
      pickBtn.append(el('span', { class: 'ph-pick-id dim' },
        t('phone.pick_pipeline')),
        el('span', { class: 'ph-pick-caret' }, '▾'));
    }
    // Mirror the desktop's rules: start only from stopped, pause only
    // while running, stop from either live state -- and nothing at all
    // while an op is still draining (a big pipeline takes seconds to
    // stop, and a queued second op would race the runtime).
    btnPlay.disabled  = busy || st !== 'stopped';
    btnPause.disabled = busy || st !== 'running';
    btnStop.disabled  = busy || !(st === 'running' || st === 'paused');
    btnSave.disabled  = !state.selectedId;
    btnMore.disabled  = false;
  }

  // Everything renderRows() draws from, and nothing else -- notably not
  // the run state, which must not cost a rebuild. Built through
  // JSON.stringify rather than by joining on a separator: stage ids and
  // type names are free-form, so any separator character could also
  // occur inside a field and make two different graphs compare equal.
  function rowsKey() {
    const d = state.detail;
    if (!d) { return 'none:' + state.pipelines.length; }
    const g = d.graph || { nodes: [], edges: [] };
    return JSON.stringify([
      d.id,
      (g.nodes || []).map((n) => [n.id, n.type, n.config_error || '']),
      (g.edges || []).map((e) => [e.from, e.from_port, e.to, e.to_port]),
    ]);
  }

  // Records the key it drew from, so every caller -- the poll, an apply,
  // a lifecycle change -- leaves the bookkeeping consistent without
  // having to remember to.
  function renderRows() {
    const d = state.detail;
    lastRowsKey = rowsKey();
    rowEls.clear();
    openBlock = null;
    clear(listEl);
    if (!d) {
      listEl.append(el('div', { class: 'ph-empty' },
        state.pipelines.length ? t('pl.select_pipeline')
                               : t('pl.no_pipelines')));
      return;
    }
    const g = d.graph || { nodes: [], edges: [] };
    if (!g.nodes.length) {
      listEl.append(el('div', { class: 'ph-empty' }, t('pl.empty')));
      return;
    }
    const laid = assignLanes(topoOrder(g.nodes, g.edges), g.edges);
    const gutter = gutterWidth(laid.width);
    for (const r of laid.rows) {
      const n = r.node;
      const bad = !!n.config_error;
      // Two controls, not one: the gutter IS the graph node and opens
      // the connection editor; the text opens the stage. Nesting one
      // button in the other is invalid HTML and the outer one wins, so
      // the row is a plain div holding both.
      const gut = el('button', {
        class: 'ph-gutter', type: 'button', style: `width:${gutter}px`,
        'aria-label': t('conn.edit_hint'),
      }, laneArt(r, laid.width, bad));
      gut.addEventListener('click', (ev) => {
        ev.stopPropagation();
        openConnections(n);
      });
      const row = el('div', { class: 'ph-row' + (bad ? ' bad' : '') },
        gut,
        el('button', { class: 'ph-row-body', type: 'button',
                       onclick: () => toggleStage(n.id) },
          el('span', { class: 'ph-row-text' },
             el('span', { class: 'ph-row-id' }, n.id),
             el('span', { class: 'ph-row-type' }, n.type)),
          bad ? el('span', { class: 'ph-warn', title: n.config_error }, '!')
              : null,
          el('span', { class: 'ph-row-caret' }, '▸')));
      listEl.append(row);
      rowEls.set(n.id, { row, lay: r, node: n, gutter, width: laid.width });
    }
    listEl.append(el('div', { class: 'ph-note' }, t('phone.topology_note')));
    if (state.openStage && rowEls.has(state.openStage)) {
      expand(state.openStage);       // survive a rebuild with one open
    } else {
      state.openStage = null;
    }
  }

  // ---- expand / collapse --------------------------------------------
  function toggleStage(id) {
    if (state.openStage === id) { collapse(); return; }
    collapse();
    expand(id);
    // A rebuild the poll deferred while this was open can now be taken,
    // but only once nothing is expanded -- so it waits for collapse().
  }

  function collapse() {
    if (openBlock) { openBlock.remove(); openBlock = null; }
    if (state.openStage) {
      const e = rowEls.get(state.openStage);
      if (e) { e.row.classList.remove('open'); }
    }
    state.openStage = null;
    // Take whatever redraw the poll deferred while the block was open.
    if (rowsKey() !== lastRowsKey) { renderRows(); }
  }

  function expand(id) {
    const e = rowEls.get(id);
    if (!e) { return; }
    state.openStage = id;
    e.row.classList.add('open');
    const g = state.detail ? state.detail.graph : { nodes: [], edges: [] };
    openBlock = detailBlock(e, g);
    e.row.after(openBlock);
  }

  // The lanes that carry on BELOW this row, drawn as plain absolutely
  // positioned rules down the block. An SVG would have to know the
  // block's height, which changes as sections open; a positioned span
  // just stretches.
  function contLanes(lay) {
    const seen = new Set([...lay.through, ...lay.branches]);
    return [...seen].map((i) => el('span',
      { class: 'ph-cont', style: `left:${laneX(i)}px` }));
  }

  function detailBlock(entry, graph) {
    const { node, lay, gutter } = entry;
    const block = el('div', { class: 'ph-detail-block' });
    block.append(...contLanes(lay));
    const inner = el('div', { class: 'ph-detail',
      style: `margin-left:${gutter}px` });
    block.append(inner);

    // Collapsed by default: on a phone the reason to open a stage is
    // almost always its configuration, and two headings of prose above
    // it would push the fields off the screen.
    const intro = section(inner, t('phone.about'), false);
    const typeLine = el('div', { class: 'ph-detail-type' }, node.type);
    const docEl = el('div', { class: 'ph-detail-doc' });
    intro.append(typeLine, docEl);
    if (node.config_error) {
      intro.append(el('div', { class: 'ph-detail-err' }, node.config_error));
    }
    specs().then((list) => {
      const sp = (list || []).find((s) => s.type === node.type);
      if (sp && sp.doc) {
        docEl.textContent = tOr('stage.' + node.type + '.doc', sp.doc);
      }
    });

    const io = section(inner, t('phone.connections'), false);
    renderConnections(io, node, graph);

    const cfg = section(inner, t('phone.config'), true);
    cfg.append(el('div', { class: 'ph-sheet-hint' }, t('common.loading')));
    loadConfig(cfg, node);
    return block;
  }

  // A collapsible section. Returns the BODY element to fill.
  function section(host, label, openByDefault) {
    const bodyEl = el('div', { class: 'ph-sec-body' });
    const caret = el('span', { class: 'ph-sec-caret' },
                     openByDefault ? '▾' : '▸');
    const head = el('button', { class: 'ph-sec-head', onclick: () => {
      const open = bodyEl.hidden;
      bodyEl.hidden = !open;
      caret.textContent = open ? '▾' : '▸';
    } }, caret, el('span', {}, label));
    bodyEl.hidden = !openByDefault;
    host.append(el('div', { class: 'ph-sec' }, head, bodyEl));
    return bodyEl;
  }

  function renderConnections(host, node, graph) {
    const edges = (graph && graph.edges) || [];
    host.append(el('div', { class: 'ph-detail-h' }, t('phone.inputs')));
    const iports = node.iports || [];
    if (!iports.length) { host.append(kv('—', t('phone.none'), true)); }
    for (const p of iports) {
      const src = edges.find((e) => e.to === node.id && e.to_port === p.index);
      host.append(kv('[' + p.index + '] ' + shortType(p.type),
        src ? (src.from + ' [' + src.from_port + ']')
            : t('phone.unconnected'), !src));
    }
    host.append(el('div', { class: 'ph-detail-h' }, t('phone.outputs')));
    const oports = node.oports || [];
    if (!oports.length) { host.append(kv('—', t('phone.none'), true)); }
    for (const p of oports) {
      const dst = edges.filter(
        (e) => e.from === node.id && e.from_port === p.index);
      host.append(kv('[' + p.index + '] ' + shortType(p.type),
        dst.length ? dst.map((e) => e.to + ' [' + e.to_port + ']').join(', ')
                   : t('phone.unconnected'), !dst.length));
    }
  }

  // ---- configuration -------------------------------------------------
  async function loadConfig(host, node) {
    let info;
    try {
      info = await api.getStageConfig(state.selectedId, node.id);
    } catch (e) {
      clear(host).append(el('div', { class: 'ph-sheet-hint' },
        t('pl.config_unavailable', { msg: e.message })));
      return;
    }
    // The block may have been collapsed while this was in flight.
    if (!host.isConnected) { return; }
    clear(host);
    const editable = !!info.editable;
    const schema = (info && info.schema) || [];
    if (!editable) {
      host.append(el('div', { class: 'ph-ro-note' }, t('phone.stop_to_edit')));
    }
    if (!schema.length) {
      host.append(el('div', { class: 'ph-sheet-hint' }, t('phone.no_config')));
      return;
    }
    const fields = [];
    // Auto-apply, as on the desktop: a field that has fully determined
    // its value (a blur after an edit, a tri-state flip, a picked path or
    // model) re-PUTs the whole config without waiting for the button.
    // Self-gates on canEdit(), so a pipeline that started while the form
    // was open never hot-applies -- there the button stays the deliberate
    // trigger.
    //
    // It applies WITHOUT rebuilding the rows. A rebuild re-expands the
    // stage and rebuilds the form, which on a phone means the field the
    // user just moved on to is destroyed under the soft keyboard. The
    // row's config_error badge is then stale, and that is what
    // collapse()'s deferred redraw is for: it already rebuilds when the
    // key has moved, so closing the block picks the change up.
    const commit = () => {
      if (canEdit()) { applyCfg(node, fields, { rebuild: false }); }
    };
    for (const f of schema) {
      const built = configField(f, {
        readOnly: !editable,
        stageType: node.type,
        onCommit: editable ? commit : null,
        // The other fields of THIS form, for the model picker's
        // parent-compatibility filter. `fields` is captured empty and
        // filled by the loop below -- it is fully populated long before
        // anything can be tapped.
        peerValues: () => fields
          .filter((x) => x.key !== f.key)
          .map((x) => x.value())
          .filter(Boolean),
      });
      fields.push(built);
      host.append(built.el);
    }
    if (!editable) { return; }
    // Still worth a button: the JSON textareas (array/object/any) have no
    // blur-commit, so this is the only way to apply one.
    const applyBtn = el('button', { class: 'btn primary ph-apply' },
      t('common.apply'));
    applyBtn.addEventListener('click', async () => {
      applyBtn.disabled = true;
      try { await applyCfg(node, fields, { rebuild: true }); }
      finally { applyBtn.disabled = false; }
    });
    host.append(applyBtn);
  }

  // PUT the whole form. `rebuild` redraws the rows from the response --
  // a stage's config_error can appear or clear on apply and the row shows
  // it, and the rebuild also re-seeds every present/unset flag from what
  // the stage actually KEPT (a value it rejected must not read back as
  // set). The auto-apply path passes false and leaves that to collapse().
  async function applyCfg(node, fields, opts) {
    const rebuild = !(opts && opts.rebuild === false);
    let cfg;
    try { cfg = readConfig(fields); }
    catch (e) { toast(e.message, 'error'); return; }
    try {
      state.detail = await api.setStageConfig(state.selectedId, node.id, cfg);
      toast(t('phone.cfg_applied', { id: node.id }), 'ok');
      renderChrome();
      if (rebuild) { renderRows(); }
    } catch (e) {
      toast(t('phone.cfg_apply_failed', { msg: e.message }), 'error');
    }
  }

  // ---- lifecycle ------------------------------------------------------
  async function lifecycle(op) {
    const id = state.selectedId;
    if (!id || state.inflight) { return; }
    state.inflight = op;
    renderChrome();          // disable the row + show the draining badge
    const label = { launch: t('common.start'), pause: t('common.pause'),
                    stop: t('common.stop') }[op] || op;
    try {
      await api[op](id);
      toast(t('pl.op_done', { op: label, id }), 'ok');
    } catch (e) {
      toast(t('pl.op_failed', { op: label, msg: e.message }), 'error');
    } finally {
      state.inflight = null;
    }
    await refresh();
    // The run state gates config editing (`editable`), so an open form
    // is now stale in a way the topology key cannot see -- rebuild it
    // unconditionally rather than wait for a graph change that may
    // never come.
    if (state.openStage) { renderRows(); }
  }

  // ---- save / load / rename -------------------------------------------
  async function onSave() {
    const id = state.selectedId;
    if (!id) { return; }
    // No remembered path -> there is nothing to overwrite, so this IS a
    // Save as (the backend would answer "no path given and none
    // remembered", which is a worse way to find out).
    if (!state.detail || !state.detail.storage_path) { onSaveAs(); return; }
    try {
      const r = await api.savePipeline(id, null);
      toast(t('pl.saved', { path: r.storage_path }), 'ok');
      await refresh();
    } catch (e) {
      toast(t('pl.save_failed', { msg: e.message }), 'error');
    }
  }

  function onSaveAs() {
    const id = state.selectedId;
    if (!id) { return; }
    const cur = state.detail ? state.detail.storage_path : '';
    openFsSheet({
      title: t('pl.save_title', { id }),
      start: dirOf(cur),
      exts: ['.vpipeline'],
      nameField: true,
      defaultName: baseOf(cur) || (id + '.vpipeline'),
      onPick: async (p) => {
        try {
          const r = await api.savePipeline(id, withPipelineExt(p));
          toast(t('pl.saved', { path: r.storage_path }), 'ok');
          await refresh();
        } catch (e) {
          toast(t('pl.save_failed', { msg: e.message }), 'error');
        }
      },
    });
  }

  // Same rule as the desktop: a basename with no dot at all gets the
  // default extension; anything else is taken as typed.
  function withPipelineExt(p) {
    const base = baseOf(p);
    return base.indexOf('.') === -1 ? p + '.vpipeline' : p;
  }

  function onLoad() {
    openFsSheet({
      title: t('phone.load_title'),
      start: dirOf(state.detail ? state.detail.storage_path : ''),
      exts: ['.vpipeline'],
      onPick: async (p) => {
        try {
          const d = await api.loadPipeline(p);
          state.selectedId = d.id;
          state.detail = null;
          state.openStage = null;
          await refresh();
          toast(t('phone.loaded', { id: d.id }), 'ok');
        } catch (e) {
          toast(t('pl.load_failed', { msg: e.message }), 'error');
        }
      },
    });
  }

  function onUnload() {
    const id = state.selectedId;
    if (!id) { toast(t('pl.select_pipeline'), 'info'); return; }
    openModal({
      title: t('pl.unload_title'),
      body: el('div', {}, t('pl.unload_confirm', { id })),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.unload'), kind: 'danger',
          onClick: async (c) => {
            try {
              await api.unloadPipeline(id);
              c();
              // Drop every reference to the pipeline that just went
              // away -- including the expanded stage, whose config form
              // is now editing something that no longer exists. refresh()
              // then selects whatever is left, or shows the empty state.
              state.selectedId = null;
              state.detail = null;
              state.openStage = null;
              await refresh();
              toast(t('phone.unloaded', { id }), 'ok');
            } catch (e) {
              toast(t('pl.unload_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
  }

  function onRename() {
    const old = state.selectedId;
    if (!old) { toast(t('pl.select_pipeline'), 'info'); return; }
    const idIn = el('input', { type: 'text', class: 'ph-input', value: old,
      autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' });
    openModal({
      title: t('pl.rename_pl_title'),
      body: el('div', {},
        el('div', { class: 'ph-sheet-label' }, t('pl.new_id')), idIn),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.rename'), kind: 'primary',
          onClick: async (c) => {
            const to = idIn.value.trim();
            if (!to) { toast(t('pl.id_required'), 'error'); return; }
            if (to === old) { c(); return; }
            try {
              await api.renamePipeline(old, to);
              c();
              state.selectedId = to;
                  await refresh();
              toast(t('pl.pl_renamed', { from: old, to }), 'ok');
            } catch (e) {
              toast(t('pl.rename_failed', { msg: e.message }), 'error');
            }
          } },
      ],
    });
  }

  // ---- pipeline chooser -----------------------------------------------
  function openPipelineSheet() {
    if (!state.pipelines.length) {
      toast(t('pl.no_pipelines'), 'info');
      return;
    }
    const list = el('div', { class: 'ph-sheet-list' });
    const close = openModal({
      title: t('phone.pick_pipeline'),
      body: list,
      actions: [{ label: t('common.cancel'), cancel: true,
                  onClick: (c) => c() }],
    });
    for (const p of state.pipelines) {
      list.append(el('button', {
        class: 'ph-sheet-item'
             + (p.id === state.selectedId ? ' active' : ''),
        onclick: () => {
          close();
          state.selectedId = p.id;
          state.detail = null;
          state.openStage = null;
          render();
          refresh();
        },
      },
        el('span', { class: 'ph-sheet-name' }, p.id),
        el('span', { class: 'ph-pill ' + p.state }, p.state)));
    }
  }

  // ---- small helpers ---------------------------------------------------
  function kv(k, v, dim) {
    return el('div', { class: 'ph-kv' },
      el('span', { class: 'ph-kv-k' }, k),
      el('span', { class: 'ph-kv-v' + (dim ? ' dim' : '') }, v));
  }

  // Payload type names arrive fully qualified; the tail is what reads.
  function shortType(ty) {
    if (!ty || ty === 'any') { return 'any'; }
    let s = String(ty);
    const k = s.lastIndexOf('::');
    if (k >= 0) { s = s.slice(k + 2); }
    return s.replace(/Payload$/, '').replace(/Beat$/, '');
  }

  // ---- poll -------------------------------------------------------------
  // Read-only view of state other clients change, so it re-reads while it
  // is on screen. Detached (another view selected) it skips the fetch but
  // keeps the timer, matching how the I/O console pauses.
  refresh();
  const timer = setInterval(() => {
    if (stopped) { clearInterval(timer); return; }
    if (!document.body.contains(listEl)) { return; }
    refresh();
  }, POLL_MS);

  return () => { stopped = true; clearInterval(timer); };
}
