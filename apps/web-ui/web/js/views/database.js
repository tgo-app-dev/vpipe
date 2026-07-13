// Database view: a read-only browser over the session's LMDB env.
//   left  ¼  -- database selector (named sub-DBs + entry counts)
//   right ¾  -- top: query form (filter keys); bottom: data view
//               (key list w/ paging | value of the selected key)

import { el, clear, toast, openModal } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

// --- local-time helpers --------------------------------------------
// Time keys travel as UTC strings + an `epoch` (seconds) so the client
// can re-format in the browser-local TZ when the user asks for it.

function pad2(n) { return String(n).padStart(2, '0'); }

function tzOffsetStr(d) {
  const off = -d.getTimezoneOffset();   // minutes east of UTC
  const sign = off >= 0 ? '+' : '-';
  const a = Math.abs(off);
  return sign + pad2(Math.floor(a / 60)) + ':' + pad2(a % 60);
}

function fmtLocal(epoch) {
  const d = new Date(epoch * 1000);
  return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-'
       + pad2(d.getDate()) + ' '
       + pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':'
       + pad2(d.getSeconds()) + tzOffsetStr(d);
}

// Parse a user-typed datetime as local. Returns epoch seconds, or null
// if we can't make sense of it (caller falls back to sending the raw
// string, which the backend then treats as UTC).
function parseLocal(s) {
  s = (s || '').trim();
  if (!s) { return null; }
  if (/^-?\d+(\.\d+)?$/.test(s)) { return parseFloat(s); }   // already epoch
  // Normalise common separators so the Date constructor accepts it.
  // No trailing 'Z' / explicit offset => Date treats the string as local.
  let norm = s.replace(' ', 'T').replace(/\//g, '-');
  // Date-only inputs ("2024-03-04") parse as UTC midnight per spec.
  // Force local interpretation by appending T00:00:00.
  if (/^\d{4}-\d{2}-\d{2}$/.test(norm)) { norm = norm + 'T00:00:00'; }
  const t = Date.parse(norm);
  return Number.isFinite(t) ? t / 1000 : null;
}

// The local-time toggle is sticky across queries, database switches,
// view re-mounts and reloads -- persisted like the theme / auth key.
const LOCALTIME_KEY = 'vpipe_db_localtime';
function loadLocalTimePref() {
  try { return localStorage.getItem(LOCALTIME_KEY) === '1'; }
  catch (e) { return false; }
}
function saveLocalTimePref(on) {
  try { localStorage.setItem(LOCALTIME_KEY, on ? '1' : '0'); }
  catch (e) {}
}

// Sticky "highlight value-filter matches in the value pane" toggle.
// Defaults ON (absent key => on) so a fresh session shows highlights.
const HIGHLIGHT_KEY = 'vpipe_db_highlight';
function loadHighlightPref() {
  try { return localStorage.getItem(HIGHLIGHT_KEY) !== '0'; }
  catch (e) { return true; }
}
function saveHighlightPref(on) {
  try { localStorage.setItem(HIGHLIGHT_KEY, on ? '1' : '0'); }
  catch (e) {}
}

// Page size for the client-paged (streaming value-filter) result set --
// mirrors the server's kPageSize so both paths look the same.
const STREAM_PAGE = 20;

export function mountDatabase(container) {
  const state = {
    dbs: [],
    selectedDb: null,
    mode: 'auto',          // auto | text | number | time
    match: 'exact',        // exact | range
    localTime: loadLocalTimePref(),  // sticky browser-local TZ toggle
    // Whether mutation (drop entry / database) is currently allowed --
    // the backend permits it only while every pipeline is stopped.
    // Refreshed from each /api/db/list response.
    deletable: false,
    q: '', lo: '', hi: '',
    page: 0,
    result: null,          // last /api/db/keys response
    selectedKey: null,     // base64 key id
    // Signature of the query whose results are currently shown. A
    // "Run query" with an identical signature keeps the page + the
    // selected entry instead of jumping back to page 0 (task: re-query
    // unchanged conditions -> stay put).
    appliedSig: null,
    // Value-content filter: an array of { cond, kw, op, indent } rows.
    // `op` ("and" | "or") + `indent` form a boolean tree -- indented rows
    // group with the row one step less indented; e.g. A / and B / or C
    // (indents 0 1 0) = (A && B) || C. op/indent are ignored for row 0.
    // When any row has a non-blank keyword the query runs through the
    // streaming /api/db/scan endpoint (paged client-side); blank rows are
    // ignored.
    filters: [],
    // Value pane: cache the last fetched value so the highlight toggle can
    // re-render without a re-fetch. highlight is a sticky preference.
    selectedValue: null,   // last /api/db/value response
    selectedDisplay: '',   // the shown key label for the value head
    highlight: loadHighlightPref(),
    streamMode: false,     // true while showing streamed (value-filtered) results
    scanRows: [],          // accumulated streamed rows (up to 64k)
    scanning: false,       // a stream is in flight
    scanTrunc: false,      // hit the 64k row cap / scan cap
    scanMode: 'text',      // resolved key mode from the stream's meta
    scanMatch: 'exact',
    scanAbort: null,       // AbortController for the in-flight stream
  };

  let streamRenderTimer = null;
  // Abort the in-flight scan (if any). Does NOT clear state.scanAbort --
  // the scan's own finalize step owns that, so a user-triggered Cancel is
  // still recognised (state.scanAbort === its ctrl) and finalizes the UI,
  // while a superseding scan overwrites scanAbort and the old finalize
  // no-ops.
  function abortScan() {
    if (state.scanAbort) {
      try { state.scanAbort.abort(); } catch (e) {}
    }
  }
  // Any filter row with a non-blank keyword makes it a value-filtered
  // (streaming) query.
  function hasActiveFilters() {
    return state.filters.some((f) => (f.kw || '').trim() !== '');
  }

  // Identity of a query's *conditions* (not its page). localTime is
  // included because in time mode it changes how typed bounds are
  // interpreted (local vs UTC), so toggling it is a real change.
  function querySig() {
    return [state.selectedDb, state.mode, state.match,
            state.q, state.lo, state.hi, state.localTime,
            JSON.stringify(state.filters)].join('\u0001');
  }

  // --- skeleton -----------------------------------------------------
  const dbListBody = el('div', { class: 'pane-body' });
  const queryBody  = el('div', { class: 'db-query' });
  const keysBody   = el('div', { class: 'db-keys' });
  const valueBody  = el('div', { class: 'db-value' });

  const dbPane = el('div', { class: 'pane' },
    el('div', { class: 'pane-head' }, el('span', { class: 'title' },
      t('db.databases'))),
    dbListBody);

  const right = el('div', { class: 'db-right' },
    queryBody,
    el('div', { class: 'db-data' }, keysBody, valueBody));

  clear(container).append(el('div', { class: 'db' }, dbPane, right));

  // --- left: database selector --------------------------------------
  function renderDbList() {
    clear(dbListBody);
    const ul = el('ul', { class: 'pl-list' });
    for (const d of state.dbs) {
      const li = el('li', {
        class: 'pl-item' + (d.name === state.selectedDb ? ' selected' : ''),
        onclick: () => selectDb(d.name),
      },
        el('span', { class: 'pl-name', title: d.name }, d.name),
        el('span', { class: 'pl-state' }, String(d.count)));
      if (state.deletable) {
        li.append(el('button', { class: 'db-del',
          title: t('db.drop_this'),
          onclick: (e) => { e.stopPropagation(); dropDatabase(d.name); } },
          '🗑'));
      }
      ul.append(li);
    }
    if (state.dbs.length === 0) {
      ul.append(el('li', { class: 'pl-item',
        style: 'cursor:default;color:var(--fg-dim)' },
        t('db.no_databases')));
    }
    dbListBody.append(ul);
  }

  async function selectDb(name) {
    state.selectedDb = name;
    state.page = 0;
    state.selectedKey = null;
    renderDbList();
    renderQuery();
    clear(valueBody);
    await runQuery();
  }

  // --- right top: query form ----------------------------------------
  function renderQuery() {
    clear(queryBody);
    if (!state.selectedDb) {
      queryBody.append(el('div', { class: 'db-hint' },
        t('db.select_db')));
      return;
    }

    const modeSel = el('select', { onchange: (e) => { state.mode = e.target.value; } },
      ...[['auto', t('db.mode_auto')], ['text', t('db.mode_text')],
          ['number', t('db.mode_number')], ['time', t('db.mode_time')]]
        .map(([v, l]) => el('option', { value: v,
          selected: state.mode === v ? '' : null }, l)));

    const matchSel = el('select', { onchange: (e) => {
      state.match = e.target.value; renderQuery();
    } },
      ...[['exact', t('db.match_exact')], ['range', t('db.match_range')]]
        .map(([v, l]) => el('option', { value: v,
          selected: state.match === v ? '' : null }, l)));

    // Detected-mode hint (only meaningful when mode=auto and we have a result).
    const detected = (state.mode === 'auto' && state.result)
      ? el('span', { class: 'db-detected' }, '→ ' + state.result.mode) : null;

    // Local-time toggle. Only meaningful when the resolved mode is
    // time; hidden otherwise so it doesn't clutter the form.
    const effectiveTime = state.mode === 'time'
      || (state.mode === 'auto' && state.result && state.result.mode === 'time');
    const localCk = el('input', { type: 'checkbox',
      checked: state.localTime ? '' : null,
      onchange: (e) => {
        state.localTime = e.target.checked;
        saveLocalTimePref(state.localTime);   // sticky across queries
        renderKeys();   // re-format displays; no re-query needed
      } });
    const localRow = effectiveTime
      ? el('label', { class: 'db-local-toggle' }, localCk,
          t('db.local_time', { tz: tzOffsetStr(new Date()) }))
      : null;

    const fieldsRow = el('div', { class: 'db-form-row' });
    if (state.match === 'exact') {
      const inp = el('input', { type: 'text', value: state.q,
        placeholder: t('db.exact_ph'),
        oninput: (e) => { state.q = e.target.value; } });
      inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { applyQuery(); } });
      fieldsRow.append(el('label', {}, t('db.key')), inp);
    } else {
      const loIn = el('input', { type: 'text', value: state.lo,
        placeholder: t('db.from_ph'),
        oninput: (e) => { state.lo = e.target.value; } });
      const hiIn = el('input', { type: 'text', value: state.hi,
        placeholder: t('db.to_ph'),
        oninput: (e) => { state.hi = e.target.value; } });
      for (const i of [loIn, hiIn]) {
        i.addEventListener('keydown', (e) => { if (e.key === 'Enter') { applyQuery(); } });
      }
      fieldsRow.append(el('label', {}, t('db.from')), loIn,
        el('label', {}, t('db.to')), hiIn);
    }

    // Run row: primary Run button, plus a Cancel + spinner while a
    // value-filtered scan is streaming.
    const runRow = el('div', { class: 'db-form-row' },
      el('button', { class: 'btn primary', onclick: () => applyQuery() },
        t('db.run_query')),
      state.scanning
        ? el('button', { class: 'btn', onclick: () => abortScan() },
            t('common.cancel'))
        : null,
      state.scanning
        ? el('span', { class: 'db-scanning' }, t('db.scanning'))
        : null);

    queryBody.append(
      el('div', { class: 'db-form-row' },
        el('label', {}, t('db.interpret_as')), modeSel, detected,
        el('label', {}, t('db.match')), matchSel,
        localRow),
      fieldsRow,
      renderValueFilter(),
      runRow);
  }

  // Keep the indent levels a well-formed tree: row 0 at 0, every row at
  // most one deeper than its predecessor (the ← / → buttons + blank-row
  // removal can otherwise leave a gap).
  function normalizeFilterIndents() {
    for (let i = 0; i < state.filters.length; i++) {
      const f = state.filters[i];
      if (i === 0) { f.indent = 0; }
      else {
        const cap = (state.filters[i - 1].indent || 0) + 1;
        f.indent = Math.max(0, Math.min(f.indent || 0, cap));
      }
    }
  }

  // Value-content filter section: rows of { cond, kw, op, indent }. Each
  // row after the first carries an and/or connective + an indent level
  // (adjusted by ← / →) so the rows form a boolean tree. Editing a keyword
  // / dropdown updates state in place WITHOUT re-rendering (keeps input
  // focus); add / remove / indent re-render the form.
  function renderValueFilter() {
    normalizeFilterIndents();
    const rows = state.filters.map((f, i) => {
      const first = (i === 0);
      const indent = f.indent || 0;

      const opSel = first ? null : el('select', { class: 'db-vfilter-op',
        onchange: (e) => { state.filters[i].op = e.target.value; } },
        el('option', { value: 'and', selected: f.op !== 'or' ? '' : null },
          t('db.op_and')),
        el('option', { value: 'or', selected: f.op === 'or' ? '' : null },
          t('db.op_or')));

      const condSel = el('select', {
        onchange: (e) => { state.filters[i].cond = e.target.value; } },
        ...[['includes', t('db.cond_includes')],
            ['excludes', t('db.cond_excludes')],
            ['regex', t('db.cond_regex')],
            ['regex_not', t('db.cond_regex_not')],
            ['regex_line', t('db.cond_regex_line')],
            ['regex_line_not', t('db.cond_regex_line_not')]]
          .map(([v, l]) => el('option',
            { value: v, selected: f.cond === v ? '' : null }, l)));
      const kwIn = el('input', { type: 'text', value: f.kw || '',
        placeholder: t('db.vfilter_ph'),
        oninput: (e) => { state.filters[i].kw = e.target.value; } });
      kwIn.addEventListener('keydown',
        (e) => { if (e.key === 'Enter') { applyQuery(); } });

      // Indent controls (row 0 is pinned at indent 0). → can go at most
      // one deeper than the row above; ← down to 0.
      const canIn = !first && indent < ((state.filters[i - 1].indent || 0) + 1);
      const outBtn = first ? null : el('button', { class: 'db-vfilter-ind',
        title: t('db.outdent'), disabled: indent <= 0 ? '' : null,
        onclick: () => {
          state.filters[i].indent = Math.max(0, indent - 1);
          renderQuery();
        } }, '←');
      const inBtn = first ? null : el('button', { class: 'db-vfilter-ind',
        title: t('db.indent'), disabled: canIn ? null : '',
        onclick: () => { state.filters[i].indent = indent + 1; renderQuery(); } },
        '→');

      const rm = el('button', { class: 'db-vfilter-rm',
        title: t('common.remove'),
        onclick: () => { state.filters.splice(i, 1); renderQuery(); } }, '×');

      return el('div',
        { class: 'db-vfilter-row',
          style: 'margin-left:' + (indent * 22) + 'px' },
        opSel, condSel, kwIn, outBtn, inBtn, rm);
    });

    return el('div', { class: 'db-vfilter' },
      el('div', { class: 'db-vfilter-head' },
        el('span', { class: 'db-vfilter-title' }, t('db.value_filter'))),
      ...rows,
      el('div', { class: 'db-vfilter-actions' },
        el('button', { class: 'btn db-vfilter-add',
          onclick: () => {
            state.filters.push({ cond: 'includes', kw: '', op: 'and',
              indent: 0 });
            renderQuery();
          } },
          t('db.add_value_filter'))));
  }

  async function applyQuery() {
    // "Run query": if the conditions are unchanged from what's shown,
    // keep the current page and selected entry (reconcile below);
    // otherwise it's a new query, so reset to page 0 and drop the
    // selection.
    const same = (querySig() === state.appliedSig);
    if (!same) {
      state.page = 0;
      state.selectedKey = null;
      clear(valueBody);
    }
    await runQuery({ reconcile: same });
  }

  // Convert typed LOCAL times to epoch seconds for the backend (which
  // reads bare numbers as epoch with ms/us/ns auto-scaling). Shared by
  // the server-paged and streaming paths.
  function timeToQuery() {
    const usingLocal = state.localTime
        && (state.mode === 'time'
            || (state.mode === 'auto' && state.result
                && state.result.mode === 'time'));
    return (s) => {
      if (!usingLocal || !s) { return s; }
      const ep = parseLocal(s);
      return ep != null ? String(ep) : s;
    };
  }

  // Dispatch: a value-content filter routes through the streaming scan
  // (client-paged); otherwise the server-paged key query.
  async function runQuery(opts) {
    if (!state.selectedDb) { return; }
    if (hasActiveFilters()) { return runScan(opts); }
    return runKeyQuery(opts);
  }

  async function runKeyQuery(opts) {
    if (!state.selectedDb) { return; }
    // Leaving stream mode: stop any in-flight scan and take ownership so
    // its finalize step no-ops (state.scanAbort no longer matches it).
    abortScan();
    state.scanAbort = null;
    state.streamMode = false;
    state.scanning = false;
    const reconcile = !!(opts && opts.reconcile);
    const toQ = timeToQuery();
    const fetchKeys = (page) => api.dbKeys({
      db: state.selectedDb, mode: state.mode, match: state.match,
      q: toQ(state.q), lo: toQ(state.lo), hi: toQ(state.hi),
      page,
    });
    // Fire the keys query and the list refresh in parallel: another
    // writer (e.g. an rtsp-capture stage) can be adding rows while
    // we browse, and the entry counts in the left list pane should
    // reflect that on every Run query. The list refresh is best-
    // effort -- if it transiently fails we keep the old counts and
    // don't bother the user with a toast for a side-effect refresh.
    let keysRes = null;
    let listRes = null;
    try {
      [keysRes, listRes] = await Promise.all([
        fetchKeys(state.page),
        api.dbList().catch(() => null),
      ]);
    } catch (e) {
      toast(t('db.query_failed', { msg: e.message }), 'error');
    }
    // On an identical re-query the original page may have vanished
    // (rows removed since): clamp to the last available page and
    // refetch so we land on real data rather than an empty page.
    if (reconcile && keysRes
        && typeof keysRes.last_page === 'number'
        && state.page > keysRes.last_page) {
      state.page = keysRes.last_page;
      try { keysRes = await fetchKeys(state.page); }
      catch (e) { toast(t('db.query_failed', { msg: e.message }), 'error'); }
    }
    state.result = keysRes;
    if (listRes && listRes.databases) {
      state.dbs = listRes.databases;
    }
    if (listRes && typeof listRes.deletable === 'boolean') {
      state.deletable = listRes.deletable;
    }
    // Record exactly which conditions these results reflect, so the
    // next identical "Run query" preserves page + selection.
    state.appliedSig = querySig();
    // Keep the previously-selected entry only if it's still present in
    // the (possibly refreshed) page; otherwise drop it and its value.
    if (reconcile && state.selectedKey) {
      const keys = (keysRes && keysRes.keys) ? keysRes.keys : [];
      if (!keys.some((k) => k.key === state.selectedKey)) {
        state.selectedKey = null;
        clear(valueBody);
      }
    }
    // Reflect detected mode + paging in the form, redraw the key
    // list, and update the left pane's entry counts.
    renderQuery();
    renderKeys();
    renderDbList();
  }

  // --- streaming value-filtered scan (client-paged) -----------------

  // Synthesize a `state.result` (the shape renderKeys reads) from the
  // accumulated streamed rows + the current page, so the same key-list /
  // pager code serves both the server-paged and streaming paths.
  function syncStreamResult() {
    const rows = state.scanRows;
    const total = rows.length;
    const lastP = total > 0 ? Math.floor((total - 1) / STREAM_PAGE) : 0;
    const page = Math.min(Math.max(0, state.page), lastP);
    state.page = page;
    const start = page * STREAM_PAGE;
    state.result = {
      db: state.selectedDb,
      mode: state.scanMode,
      match: state.scanMatch,
      page, last_page: lastP, total,
      keys: rows.slice(start, start + STREAM_PAGE),
      has_prev: page > 0, has_next: page < lastP,
      truncated: state.scanTrunc,
      scanning: state.scanning,   // consumed by the key-list head
    };
  }

  // Coalesce the flood of onRow callbacks into at most one re-render per
  // ~120 ms so accumulating up to 64k rows doesn't thrash the DOM.
  function scheduleStreamRender() {
    if (streamRenderTimer) { return; }
    streamRenderTimer = setTimeout(() => {
      streamRenderTimer = null;
      if (state.streamMode) { syncStreamResult(); renderKeys(); }
    }, 120);
  }

  async function runScan(opts) {
    if (!state.selectedDb) { return; }
    const reconcile = !!(opts && opts.reconcile);
    abortScan();
    const ctrl = new AbortController();
    state.scanAbort = ctrl;
    state.streamMode = true;
    state.scanRows = [];
    state.scanning = true;
    state.scanTrunc = false;
    state.scanMode = state.mode === 'auto' ? 'text' : state.mode;
    state.scanMatch = state.match;
    if (!reconcile) {
      state.page = 0;
      state.selectedKey = null;
      clear(valueBody);
    }
    // Best-effort left-pane count refresh (a writer may be adding rows),
    // in parallel with the scan.
    api.dbList().then((d) => {
      if (d && d.databases) { state.dbs = d.databases; }
      if (d && typeof d.deletable === 'boolean') {
        state.deletable = d.deletable;
      }
      renderDbList();
    }).catch(() => {});

    // Drop blank rows, then re-normalize indents of the surviving rows so
    // the tree stays well-formed (a removed middle row could leave a gap).
    const active = state.filters.filter((f) => (f.kw || '').trim() !== '');
    const sentFilters = [];
    active.forEach((f, i) => {
      const cap = i === 0 ? 0 : sentFilters[i - 1].indent + 1;
      sentFilters.push({
        cond: f.cond || 'includes',
        kw: f.kw,
        op: f.op === 'or' ? 'or' : 'and',
        indent: i === 0 ? 0 : Math.max(0, Math.min(f.indent || 0, cap)),
      });
    });
    const toQ = timeToQuery();
    const body = {
      db: state.selectedDb, mode: state.mode, match: state.match,
      q: toQ(state.q), lo: toQ(state.lo), hi: toQ(state.hi),
      filters: sentFilters,
    };
    // Show the scanning state (empty list + Cancel) right away.
    syncStreamResult();
    renderQuery();
    renderKeys();

    let errMsg = null;
    try {
      await api.dbScan(body, {
        onMeta: (m) => {
          if (m.mode) { state.scanMode = m.mode; }
          if (m.match) { state.scanMatch = m.match; }
        },
        onRow: (row) => { state.scanRows.push(row); scheduleStreamRender(); },
        onDone: (d) => {
          state.scanning = false;
          state.scanTrunc = !!d.truncated;
        },
        onError: (msg) => { state.scanning = false; errMsg = msg; },
      }, ctrl.signal);
    } catch (e) {
      if (!e || e.name !== 'AbortError') { errMsg = e.message; }
    } finally {
      // Only the currently-owned scan finalizes the shared UI; a scan
      // that was superseded (newer scan / left stream mode) no-ops.
      if (state.scanAbort === ctrl) {
        state.scanAbort = null;
        state.scanning = false;
        if (streamRenderTimer) {
          clearTimeout(streamRenderTimer);
          streamRenderTimer = null;
        }
        // Drop a selection that didn't survive the (re)scan.
        if (reconcile && state.selectedKey
            && !state.scanRows.some((k) => k.key === state.selectedKey)) {
          state.selectedKey = null;
          clear(valueBody);
        }
        state.appliedSig = querySig();
        syncStreamResult();
        renderQuery();
        renderKeys();
        renderDbList();
        if (errMsg) { toast(t('db.query_failed', { msg: errMsg }), 'error'); }
      }
    }
  }

  // --- right bottom-left: key list + pager --------------------------
  function renderKeys() {
    clear(keysBody);
    const r = state.result;
    const keys = r ? r.keys : [];

    const timeMode = r && r.mode === 'time';
    const list = el('ul', { class: 'db-keylist' });
    for (const k of keys) {
      // In time mode, re-format the server's UTC display in browser-
      // local time when the user asked for it (keeps the UTC display
      // as the row's tooltip for reference).
      let shown = k.display || t('db.empty');
      if (timeMode && state.localTime && typeof k.epoch === 'number') {
        shown = fmtLocal(k.epoch);
      }
      const li = el('li', {
        class: 'db-key' + (k.key === state.selectedKey ? ' selected' : ''),
        title: k.display,
        onclick: () => selectKey(k.key, shown),
      }, el('span', { class: 'db-key-label' }, shown));
      if (state.deletable) {
        li.append(el('button', { class: 'db-del',
          title: t('db.delete_this'),
          onclick: (e) => { e.stopPropagation(); deleteEntry(k.key, shown); } },
          '🗑'));
      }
      list.append(li);
    }
    if (keys.length === 0) {
      list.append(el('li', { class: 'db-key empty' }, t('db.no_matching')));
    }

    const pageNo = r ? r.page : 0;
    const lastP = r && typeof r.last_page === 'number' ? r.last_page : 0;
    const goto_ = (p) => {
      state.page = Math.max(0, Math.min(lastP, p));
      // Stream mode pages client-side over the accumulated rows -- no
      // re-scan; the server-paged path re-queries for the new page.
      if (state.streamMode) { syncStreamResult(); renderKeys(); }
      else { runQuery(); }
    };
    const pageLabel = r
      ? t('db.page_of', { n: pageNo + 1, total: lastP + 1,
          plus: r.truncated ? '+' : '' })
      : t('db.page_n', { n: pageNo + 1 });
    const pager = el('div', { class: 'db-pager' },
      el('button', { class: 'btn', disabled: !(r && r.has_prev),
        title: t('db.first_title'), onclick: () => goto_(0) },
        t('db.first')),
      el('button', { class: 'btn', disabled: !(r && r.has_prev),
        title: t('db.prev_title'), onclick: () => goto_(pageNo - 1) },
        t('db.prev')),
      el('span', { class: 'db-pageno' }, pageLabel),
      el('button', { class: 'btn', disabled: !(r && r.has_next),
        title: t('db.next_title'), onclick: () => goto_(pageNo + 1) },
        t('db.next')),
      el('button', { class: 'btn', disabled: !(r && r.has_next),
        title: t('db.last_title'), onclick: () => goto_(lastP) },
        t('db.last')));

    // Page count + total across all pages (the server reports `total`;
    // a truncated scan makes it a lower bound, flagged with "+").
    const kw = (n) => n + ' key' + (n === 1 ? '' : 's');
    const total = (r && typeof r.total === 'number') ? r.total : keys.length;
    keysBody.append(
      el('div', { class: 'db-keys-head' },
        el('span', {}, t('db.total', { shown: kw(keys.length),
          total: kw(total), plus: (r && r.truncated ? '+' : '') })),
        (r && r.truncated)
          ? el('span', { class: 'db-trunc', title: t('db.truncated_title') },
              t('db.truncated'))
          : null,
        // Live "scanning…" indicator while a value-filtered stream runs.
        (r && r.scanning)
          ? el('span', { class: 'db-scanning' }, t('db.scanning'))
          : null),
      list, pager);
  }

  // --- right bottom-right: value ------------------------------------
  async function selectKey(b64, display) {
    state.selectedKey = b64;
    state.selectedDisplay = display;
    state.selectedValue = null;
    renderKeys();
    clear(valueBody);
    valueBody.append(el('div', { class: 'db-hint' }, t('common.loading')));
    let v;
    try {
      v = await api.dbValue(state.selectedDb, b64);
    } catch (e) {
      clear(valueBody);
      valueBody.append(el('div', { class: 'db-hint' },
        t('db.read_failed', { msg: e.message })));
      return;
    }
    // A newer selection may have landed while this fetch was in flight.
    if (state.selectedKey !== b64) { return; }
    state.selectedValue = v;
    renderValue();
  }

  // The positive value-filter conditions (the ones asserting presence) --
  // these are what "highlight matches" marks in the value pane.
  function activePositiveFilters() {
    return state.filters.filter((f) =>
      (f.kw || '').trim() !== ''
      && (f.cond === 'includes' || f.cond === 'regex'
          || f.cond === 'regex_line'));
  }

  // Merged [start,end) ranges in `text` matched by the positive filters.
  // includes -> case-insensitive substring; regex kinds -> a global
  // RegExp ('m' for the line variant so ^/$ anchor per line). Bad regexes
  // are skipped. Zero-width matches are ignored.
  function matchRanges(text, positives) {
    const ranges = [];
    for (const f of positives) {
      const kw = f.kw;
      if (f.cond === 'includes') {
        const needle = kw.toLowerCase();
        if (!needle) { continue; }
        const hay = text.toLowerCase();
        let i = 0;
        while ((i = hay.indexOf(needle, i)) >= 0) {
          ranges.push([i, i + needle.length]);
          i += needle.length;
        }
      } else {
        let re;
        try {
          re = new RegExp(kw, f.cond === 'regex_line' ? 'gim' : 'gi');
        } catch (e) { continue; }
        let m;
        let guard = 0;
        while ((m = re.exec(text)) !== null) {
          if (m[0].length === 0) { re.lastIndex++; }
          else { ranges.push([m.index, m.index + m[0].length]); }
          if (++guard > 200000) { break; }
        }
      }
    }
    ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const merged = [];
    for (const r of ranges) {
      const last = merged[merged.length - 1];
      if (last && r[0] <= last[1]) { last[1] = Math.max(last[1], r[1]); }
      else { merged.push([r[0], r[1]]); }
    }
    return merged;
  }

  // Fill `pre` with `text`, wrapping the matched ranges in <mark>.
  function appendHighlighted(pre, text, positives) {
    const ranges = matchRanges(text, positives);
    if (ranges.length === 0) { pre.textContent = text; return; }
    let pos = 0;
    for (const [s, e] of ranges) {
      if (s > pos) { pre.append(document.createTextNode(text.slice(pos, s))); }
      pre.append(el('mark', { class: 'db-hl' }, text.slice(s, e)));
      pos = e;
    }
    if (pos < text.length) {
      pre.append(document.createTextNode(text.slice(pos)));
    }
  }

  // Render the cached value (state.selectedValue). Highlighting is offered
  // only for FlexData->JSON values (the hex dump of a binary value doesn't
  // line up with the searched bytes) and only when positive filters exist.
  function renderValue() {
    clear(valueBody);
    const v = state.selectedValue;
    if (!v) { return; }
    if (!v.found) {
      valueBody.append(el('div', { class: 'db-hint' }, t('db.key_not_found')));
      return;
    }
    const enc = v.encoding === 'json' ? t('db.enc_json')
              : t('db.enc_binary');
    const positives = activePositiveFilters();
    const canHl = v.encoding === 'json' && positives.length > 0;
    const hlToggle = canHl
      ? el('label', { class: 'db-hl-toggle', title: t('db.highlight_title') },
          el('input', { type: 'checkbox',
            checked: state.highlight ? '' : null,
            onchange: (e) => {
              state.highlight = e.target.checked;
              saveHighlightPref(state.highlight);
              renderValue();
            } }),
          t('db.highlight'))
      : null;
    // .allow-context-menu: keep the native right-click menu (copy /
    // select-all) on the entry's value text, exempt from the app-wide
    // context-menu suppression.
    const pre = el('pre',
      { class: 'db-dump allow-context-menu ' + v.encoding });
    if (canHl && state.highlight) {
      appendHighlighted(pre, v.text, positives);
    } else {
      pre.textContent = v.text;
    }
    valueBody.append(
      el('div', { class: 'db-value-head' },
        el('span', { class: 'db-vkey', title: state.selectedDisplay },
          state.selectedDisplay),
        hlToggle,
        el('span', { class: 'db-vmeta' }, v.size + ' B · ' + enc)),
      pre);
  }

  // --- mutation: drop entry / database ------------------------------
  // Both are gated server-side on "every pipeline stopped"; the buttons
  // that call these are only rendered when state.deletable is true, but
  // the backend is the authority (a 409 surfaces as a toast).

  // Re-fetch the database list (names + counts + deletable flag) and
  // redraw. If the currently-selected db has vanished, drop the
  // selection and fall back to the first remaining db (or an empty
  // view).
  async function refreshDbList() {
    try {
      const d = await api.dbList();
      state.dbs = (d && d.databases) ? d.databases : [];
      state.deletable = !!(d && d.deletable);
    } catch (e) {
      toast(t('db.list_failed', { msg: e.message }), 'error');
    }
    const gone = state.selectedDb
      && !state.dbs.some((x) => x.name === state.selectedDb);
    if (gone) {
      state.selectedDb = null;
      state.result = null;
      state.selectedKey = null;
      state.appliedSig = null;
      clear(valueBody);
    }
    renderDbList();
    renderQuery();
    renderKeys();
    if (!state.selectedDb && state.dbs.length > 0) {
      await selectDb(state.dbs[0].name);
    }
  }

  function deleteEntry(b64, shown) {
    openModal({
      title: t('db.delete_entry'),
      body: el('div', {},
        el('p', {}, t('db.delete_entry_confirm',
          { db: state.selectedDb })),
        el('pre', { class: 'db-dump' }, shown)),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('common.delete'), kind: 'primary',
          onClick: async (c) => {
            c();
            try {
              await api.dbDeleteKey(state.selectedDb, b64);
              toast(t('db.entry_deleted'), 'ok');
              if (state.selectedKey === b64) {
                state.selectedKey = null;
                clear(valueBody);
              }
              // Re-run the current query (reconcile keeps page/selection
              // where it can) and refresh the left-pane counts.
              await runQuery({ reconcile: true });
            } catch (e) {
              toast(t('db.delete_failed', { msg: e.message }), 'error');
            }
        } },
      ],
    });
  }

  function dropDatabase(name) {
    openModal({
      title: t('db.drop_db'),
      body: el('div', {},
        el('p', {}, t('db.drop_db_confirm', { name }))),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('db.drop_db'), kind: 'primary', onClick: async (c) => {
            c();
            try {
              await api.dbDrop(name);
              toast(t('db.db_dropped'), 'ok');
              await refreshDbList();
            } catch (e) {
              toast(t('db.drop_failed', { msg: e.message }), 'error');
            }
        } },
      ],
    });
  }

  // --- boot ---------------------------------------------------------
  renderQuery();
  renderKeys();
  refreshDbList();
}
