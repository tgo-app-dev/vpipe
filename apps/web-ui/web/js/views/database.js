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
  };

  // Identity of a query's *conditions* (not its page). localTime is
  // included because in time mode it changes how typed bounds are
  // interpreted (local vs UTC), so toggling it is a real change.
  function querySig() {
    return [state.selectedDb, state.mode, state.match,
            state.q, state.lo, state.hi, state.localTime].join('\u0001');
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

    queryBody.append(
      el('div', { class: 'db-form-row' },
        el('label', {}, t('db.interpret_as')), modeSel, detected,
        el('label', {}, t('db.match')), matchSel,
        localRow),
      fieldsRow,
      el('div', { class: 'db-form-row' },
        el('button', { class: 'btn primary', onclick: () => applyQuery() },
          t('db.run_query'))));
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

  async function runQuery(opts) {
    if (!state.selectedDb) { return; }
    const reconcile = !!(opts && opts.reconcile);
    // When the user enters times as LOCAL we convert them to epoch
    // seconds here; the backend interprets bare numbers as epoch (with
    // automatic ms/us/ns scaling), so this round-trips correctly.
    const usingLocal = state.localTime
        && (state.mode === 'time'
            || (state.mode === 'auto' && state.result
                && state.result.mode === 'time'));
    const toQ = (s) => {
      if (!usingLocal || !s) { return s; }
      const ep = parseLocal(s);
      return ep != null ? String(ep) : s;
    };
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
      runQuery();
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
          : null),
      list, pager);
  }

  // --- right bottom-right: value ------------------------------------
  async function selectKey(b64, display) {
    state.selectedKey = b64;
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
    clear(valueBody);
    if (!v.found) {
      valueBody.append(el('div', { class: 'db-hint' }, t('db.key_not_found')));
      return;
    }
    const enc = v.encoding === 'json' ? t('db.enc_json')
              : t('db.enc_binary');
    valueBody.append(
      el('div', { class: 'db-value-head' },
        el('span', { class: 'db-vkey', title: display }, display),
        el('span', { class: 'db-vmeta' }, v.size + ' B · ' + enc)),
      // .allow-context-menu: keep the native right-click menu (copy /
      // select-all) on the entry's value text, exempt from the app-wide
      // context-menu suppression.
      el('pre', { class: 'db-dump allow-context-menu ' + v.encoding },
        v.text));
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
