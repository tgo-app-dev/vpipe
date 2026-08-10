// Shared virtualized + paged directory listing over GET /api/fs/list.
// Used by BOTH the file-browser view and the open/save dialog so they
// share the large-directory optimization: entries arrive in fixed-size
// pages (never one huge response) and only the rows in (or near) the
// viewport are in the DOM, positioned by a fixed row height. A loading
// hourglass is signalled (onLoading) while any fetch is in flight.
//
// The component owns just the list (a pinned mounts block + a virtualized
// scroll region + selection + the keyboard cursor). Interaction policy is
// injected via options, so each consumer keeps its own chrome (toolbar /
// filters / footer).

import { el, clear } from './dom.js';
import { makeIcon } from './icons.js';
import { api } from './api.js';
import { t } from './i18n.js';

export const FS_ROW_H = 26;   // fixed list-row height, px (matches CSS)
const PAGE   = 200;           // entries per /api/fs/list window
const BUFFER = 8;             // extra rows above/below the viewport

export function humanSize(n) {
  if (!(n > 0)) { return '0 B'; }
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(v < 10 ? 1 : 0)) + ' ' + u[i];
}

// Join a directory with a child name in the server's forward-slash
// namespace (works for both the virtual "/" root and native paths).
export function joinPath(dir, name) {
  if (!dir || dir === '/') { return '/' + name; }
  return dir.replace(/[/\\]+$/, '') + '/' + name;
}

// Truncate in the MIDDLE (keep head + tail) so a long mount path keeps
// both its leading "/" and its meaningful tail. Plain LTR -- bidi-safe.
function truncateMiddle(s, max) {
  s = String(s || '');
  if (s.length <= max) { return s; }
  const head = Math.ceil((max - 1) * 0.4);
  const tail = max - 1 - head;
  return s.slice(0, head) + '…' + s.slice(s.length - tail);
}

// createFsList(opts) -> controller.
// opts:
//   multi          (bool)  select several files (else single-select)
//   selectableDirs (bool)  directories can be selected (highlight)
//   dirClickOpens  (bool)  single click on a dir opens it (else selects;
//                          double-click always opens)
//   exts           ([str]) file-extension filter (dot-led); [] = all
//   iconFor        (fn)    entry -> icon name (default folder/file)
//   onDirOpen      (fn)    entry -> void   navigate into a directory
//   onFileActivate (fn)    entry -> void   double-click a file
//   onSelect       (fn)    (names[], entries[]) -> void  selection changed
//   onMount        (fn)    mount -> void   a pinned mount clicked
//   onLoading      (fn)    busy:bool -> void  fetch in flight toggled
//   onNavUp        (fn)    parentPath -> void  Left arrow: leave for the
//                          parent directory (the host navigates, since it
//                          owns whatever else a navigation updates)
//
// controller: { el, load(path), reload(), setExts(exts), getSelection(),
//   getSelectionNames(), setSelection(names), clearSelection(), current(),
//   focus(), destroy() }.
export function createFsList(opts = {}) {
  const multi          = !!opts.multi;
  const selectableDirs = !!opts.selectableDirs;
  const dirClickOpens  = !!opts.dirClickOpens;
  const iconFor = typeof opts.iconFor === 'function'
    ? opts.iconFor : (e) => (e.dir ? 'folder' : 'file');
  const onDirOpen      = opts.onDirOpen || (() => {});
  const onFileActivate = opts.onFileActivate || (() => {});
  const onSelect       = opts.onSelect || (() => {});
  const onMount        = opts.onMount || (() => {});
  const onLoading      = opts.onLoading || (() => {});
  const onNavUp        = opts.onNavUp || (() => {});

  let exts = Array.isArray(opts.exts) ? opts.exts.slice() : [];

  // ---- state ------------------------------------------------------
  let curPath = '';
  let meta = { parent: '', sandboxed: false, mounts: [] };
  const pages   = new Map();   // pageIdx -> entries[]
  const pending = new Map();   // pageIdx -> in-flight fetch promise
  let total = 0, loadGen = 0, raf = 0, busy = 0;
  const selected = new Set();  // selected entry names
  const selInfo  = new Map();  // name -> entry
  // Keyboard cursor: the row the arrow keys move from, as an INDEX into
  // the (virtualized) listing. It is kept apart from the selection
  // because the two can differ -- the dialog cannot select a directory,
  // and a cursor that refused to land on one could never step into it
  // with Right. Where directories ARE selectable the two coincide, so
  // the file browser sees a single highlight.
  let cursor = -1;

  // ---- DOM --------------------------------------------------------
  const mountsEl = el('div', { class: 'fs-mounts' });
  const sizer    = el('div', { class: 'fs-sizer' });
  const scroll   = el('div', { class: 'fs-scroll' }, sizer);
  const emptyEl  = el('div', { class: 'fs-empty' }, t('fs.empty'));
  emptyEl.hidden = true;
  // tabindex: the arrow keys are handled HERE rather than on the
  // document, so the listener dies with this DOM. The file-browser view
  // has no unmount hook, so a document-level one would leak a handler
  // per navigation away and back -- and stale handlers would keep
  // driving destroyed lists.
  const root = el('div', { class: 'fs-vlist', tabindex: '0' },
    mountsEl, scroll, emptyEl);

  scroll.addEventListener('scroll', () => {
    if (raf) { return; }
    raf = requestAnimationFrame(() => { raf = 0; renderWindow(); });
  });
  const ro = new ResizeObserver(() => renderWindow());
  ro.observe(scroll);

  // ---- paging + windowing -----------------------------------------
  const pageOf = (i) => Math.floor(i / PAGE);
  function getEntry(i) {
    const pg = pages.get(pageOf(i));
    return pg ? (pg[i - pageOf(i) * PAGE] || null) : null;
  }
  function setBusy(on) {
    const was = busy > 0;
    busy += on ? 1 : -1;
    if (busy < 0) { busy = 0; }
    if (was !== (busy > 0)) { onLoading(busy > 0); }
  }
  function setTotal(n) {
    total = n;
    sizer.style.height = (n * FS_ROW_H) + 'px';
    scroll.hidden = n === 0;
    emptyEl.hidden = !(n === 0 && !(meta.mounts || []).length);
  }
  const extParam = () => (exts.length ? { exts } : {});

  // Awaiting an in-flight page returns the SAME promise rather than
  // resolving immediately: the keyboard awaits this to select a row it
  // just jumped to, and an early return would hand it a page that is
  // still empty -- so a held-down arrow key would move the highlight
  // and select nothing.
  function ensurePage(pg) {
    if (pg < 0 || pages.has(pg)) { return Promise.resolve(); }
    const inflight = pending.get(pg);
    if (inflight) { return inflight; }
    const gen = loadGen;
    setBusy(true);
    const p = (async () => {
      try {
        const d = await api.fsList(curPath,
          { offset: pg * PAGE, limit: PAGE, ...extParam() });
        if (gen !== loadGen) { return; }     // navigated away
        pages.set(pg, d.entries || []);
        if (typeof d.total === 'number') { setTotal(d.total); }
        renderWindow();
      } catch (e) {
        // Transient: leave the page unloaded; a later scroll re-attempts.
      } finally {
        pending.delete(pg);
        setBusy(false);
      }
    })();
    pending.set(pg, p);
    return p;
  }

  // ---- selection --------------------------------------------------
  function applyHighlight() {
    for (const row of sizer.children) {
      row.classList.toggle('sel', selected.has(row.dataset.name));
      row.classList.toggle('cur', +row.dataset.index === cursor);
    }
  }
  function emitSelect() {
    const names = [...selected];
    onSelect(names, names.map((n) => selInfo.get(n)).filter(Boolean));
  }
  // Make `e` the whole selection. The primitive the keyboard uses: a
  // multi-select list must still REPLACE on an arrow key, or walking
  // down over an already-picked file would silently drop it.
  function selectOnly(e) {
    if (e.dir && !selectableDirs) { applyHighlight(); return; }
    selected.clear();
    selInfo.clear();
    selected.add(e.name);
    selInfo.set(e.name, e);
    applyHighlight();
    emitSelect();
  }
  function selectEntry(e) {
    if (e.dir && !selectableDirs) { return; }
    if (multi && !e.dir) {
      if (selected.has(e.name)) {
        selected.delete(e.name);
        selInfo.delete(e.name);
      } else {
        selected.add(e.name);
        selInfo.set(e.name, e);
      }
      applyHighlight();
      emitSelect();
      return;
    }
    selectOnly(e);
  }

  // ---- keyboard cursor ---------------------------------------------
  const cursorEntry = () => (cursor < 0 ? null : getEntry(cursor));
  // Index of a name among the LOADED pages, or -1. Best-effort by
  // design: finding a name in an unloaded page would mean walking the
  // directory a page at a time, and the callers (rename, mkdir, coming
  // back up out of a directory) all name something the current listing
  // has just been reloaded around.
  function indexOfName(name) {
    for (const [pg, ents] of pages) {
      const i = ents.findIndex((e) => e.name === name);
      if (i >= 0) { return pg * PAGE + i; }
    }
    return -1;
  }
  function scrollIndexIntoView(i) {
    const top = i * FS_ROW_H;
    const vh  = scroll.clientHeight || 0;
    if (top < scroll.scrollTop) { scroll.scrollTop = top; }
    else if (top + FS_ROW_H > scroll.scrollTop + vh) {
      scroll.scrollTop = top + FS_ROW_H - vh;
    }
  }
  // Move the cursor to `i` (clamped), reveal it, and select it. The
  // target row may live in a page that has never been fetched -- a
  // long press of ArrowDown outruns the loaded window -- so the page is
  // awaited, and the result discarded if the cursor moved on meanwhile.
  async function focusIndex(i) {
    if (total === 0) { return; }
    i = Math.max(0, Math.min(total - 1, i));
    cursor = i;
    scrollIndexIntoView(i);
    let e = getEntry(i);
    if (!e) {
      await ensurePage(pageOf(i));
      if (cursor !== i) { return; }
      e = getEntry(i);
    }
    renderWindow();
    if (e) { selectOnly(e); }
    else { applyHighlight(); }   // fetch failed -- move the cursor only
  }

  // Arrow keys act on the list only while it has focus, which a click on
  // any row gives it (the root is tabbable). Enter is deliberately NOT
  // taken: in the dialog it belongs to the modal's confirm action.
  root.addEventListener('keydown', (ev) => {
    if (ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) { return; }
    const page = Math.max(1,
      Math.floor((scroll.clientHeight || 0) / FS_ROW_H) - 1);
    const at = cursor;
    switch (ev.key) {
      case 'ArrowDown':  focusIndex(at < 0 ? 0 : at + 1); break;
      case 'ArrowUp':    focusIndex(at < 0 ? 0 : at - 1); break;
      case 'PageDown':   focusIndex(at < 0 ? 0 : at + page); break;
      case 'PageUp':     focusIndex(at < 0 ? 0 : at - page); break;
      case 'Home':       focusIndex(0); break;
      case 'End':        focusIndex(total - 1); break;
      case 'ArrowLeft':  onNavUp(meta.parent); break;
      case 'ArrowRight': {
        const e = cursorEntry();
        if (e && e.dir) { onDirOpen(e); }
        break;
      }
      default: return;             // not ours -- leave it to the host
    }
    ev.preventDefault();
  });

  // ---- rows -------------------------------------------------------
  function rowFor(e, i) {
    const row = el('div', {
      class: 'fs-row fs-vrow' + (e.dir ? ' dir' : '')
        + (selected.has(e.name) ? ' sel' : '')
        + (cursor === i ? ' cur' : ''),
      style: 'top:' + (i * FS_ROW_H) + 'px',
      'data-name': e.name,
      'data-index': String(i),
    }, makeIcon(iconFor(e), 'sm'), el('span', { class: 'fs-nm' }, e.name),
       e.dir ? null : el('span', { class: 'fs-sz' }, humanSize(e.size || 0)));
    row.addEventListener('click', () => {
      // The click is also where the keyboard picks up from, so the
      // cursor follows it even when the row cannot be selected.
      cursor = i;
      if (e.dir && dirClickOpens) { onDirOpen(e); return; }
      selectEntry(e);
      applyHighlight();
    });
    row.addEventListener('dblclick', () => {
      if (e.dir) { onDirOpen(e); }
      else { selectEntry(e); onFileActivate(e); }
    });
    return row;
  }
  function placeholderRow(i) {
    return el('div', { class: 'fs-row fs-vrow fs-vrow-ph',
      style: 'top:' + (i * FS_ROW_H) + 'px' },
      el('span', { class: 'fs-nm' }, '…'));
  }

  // Render only the rows in (and just around) the viewport, positioned by
  // index; ensure the pages covering that window are loaded.
  function renderWindow() {
    clear(sizer);
    if (total === 0) { return; }
    const vh   = scroll.clientHeight || 0;
    const top  = scroll.scrollTop || 0;
    const first = Math.max(0, Math.floor(top / FS_ROW_H) - BUFFER);
    const last  = Math.min(total - 1,
      Math.ceil((top + vh) / FS_ROW_H) + BUFFER);
    for (let pg = pageOf(first); pg <= pageOf(last); pg++) {
      if (!pages.has(pg)) { ensurePage(pg); }
    }
    for (let i = first; i <= last; i++) {
      const e = getEntry(i);
      sizer.append(e ? rowFor(e, i) : placeholderRow(i));
    }
  }

  function renderMounts() {
    clear(mountsEl);
    const ms = meta.mounts || [];
    mountsEl.hidden = ms.length === 0;
    for (const m of ms) {
      const row = el('div', { class: 'fs-row mount', title: m.path },
        makeIcon('folder', 'sm'), el('span', { class: 'fs-nm' }, m.name),
        el('span', { class: 'fs-mount-path' }, truncateMiddle(m.path, 42)));
      row.addEventListener('click', () => onMount(m));
      mountsEl.append(row);
    }
  }

  // ---- load -------------------------------------------------------
  // Fetch the FIRST page of `path`, size the scroll area from `total`, and
  // render the initial window. Rejects (throws) on a hard failure so the
  // caller can fall back. Returns dir meta, or null if superseded.
  async function load(path) {
    const gen = ++loadGen;
    setBusy(true);
    let d;
    try {
      d = await api.fsList(path, { offset: 0, limit: PAGE, ...extParam() });
    } finally {
      setBusy(false);
    }
    if (gen !== loadGen) { return null; }   // superseded by a later load
    curPath = d.path || path || '';
    meta = { parent: d.parent || '', sandboxed: !!d.sandboxed,
             mounts: d.mounts || [] };
    selected.clear();
    selInfo.clear();
    pages.clear();
    pending.clear();
    cursor = -1;                            // a new directory, no cursor
    pages.set(0, d.entries || []);
    scroll.scrollTop = 0;
    setTotal(typeof d.total === 'number' ? d.total : (d.entries || []).length);
    renderMounts();
    renderWindow();
    emitSelect();
    return { path: curPath, parent: meta.parent, sandboxed: meta.sandboxed,
             total };
  }

  return {
    el: root,
    load,
    reload: () => load(curPath),
    setExts: (e) => {
      exts = Array.isArray(e) ? e.slice() : [];
      return load(curPath);
    },
    getSelection: () =>
      [...selected].map((n) => selInfo.get(n)).filter(Boolean),
    getSelectionNames: () => [...selected],
    // Highlight `names`, and put the cursor on the first of them so the
    // arrow keys carry on from what the caller just pointed at (the new
    // folder, the renamed file, the directory we stepped back out of)
    // rather than from the top of the listing. It also scrolls that row
    // into view, since the caller's whole point is to show it.
    setSelection: (names) => {
      selected.clear();
      selInfo.clear();
      for (const n of (names || [])) { selected.add(n); }
      const i = (names && names.length) ? indexOfName(names[0]) : -1;
      cursor = i;
      if (i >= 0) { scrollIndexIntoView(i); renderWindow(); }
      applyHighlight();
    },
    clearSelection: () => {
      selected.clear();
      selInfo.clear();
      cursor = -1;
      applyHighlight();
      emitSelect();
    },
    current: () => ({ path: curPath, parent: meta.parent,
                      sandboxed: meta.sandboxed, total }),
    focus: () => root.focus({ preventScroll: true }),
    destroy: () => ro.disconnect(),
  };
}
