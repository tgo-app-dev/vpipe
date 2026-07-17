// Shared virtualized + paged directory listing over GET /api/fs/list.
// Used by BOTH the file-browser view and the open/save dialog so they
// share the large-directory optimization: entries arrive in fixed-size
// pages (never one huge response) and only the rows in (or near) the
// viewport are in the DOM, positioned by a fixed row height. A loading
// hourglass is signalled (onLoading) while any fetch is in flight.
//
// The component owns just the list (a pinned mounts block + a virtualized
// scroll region + selection). Interaction policy is injected via options,
// so each consumer keeps its own chrome (toolbar / filters / footer).

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
//
// controller: { el, load(path), reload(), setExts(exts), getSelection(),
//   getSelectionNames(), setSelection(names), clearSelection(), current(),
//   destroy() }.
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

  let exts = Array.isArray(opts.exts) ? opts.exts.slice() : [];

  // ---- state ------------------------------------------------------
  let curPath = '';
  let meta = { parent: '', sandboxed: false, mounts: [] };
  const pages   = new Map();   // pageIdx -> entries[]
  const pending = new Set();   // pageIdx in flight
  let total = 0, loadGen = 0, raf = 0, busy = 0;
  const selected = new Set();  // selected entry names
  const selInfo  = new Map();  // name -> entry

  // ---- DOM --------------------------------------------------------
  const mountsEl = el('div', { class: 'fs-mounts' });
  const sizer    = el('div', { class: 'fs-sizer' });
  const scroll   = el('div', { class: 'fs-scroll' }, sizer);
  const emptyEl  = el('div', { class: 'fs-empty' }, t('fs.empty'));
  emptyEl.hidden = true;
  const root = el('div', { class: 'fs-vlist' }, mountsEl, scroll, emptyEl);

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

  async function ensurePage(pg) {
    if (pg < 0 || pages.has(pg) || pending.has(pg)) { return; }
    const gen = loadGen;
    pending.add(pg);
    setBusy(true);
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
  }

  // ---- selection --------------------------------------------------
  function applyHighlight() {
    for (const row of sizer.children) {
      row.classList.toggle('sel', selected.has(row.dataset.name));
    }
  }
  function emitSelect() {
    const names = [...selected];
    onSelect(names, names.map((n) => selInfo.get(n)).filter(Boolean));
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
    } else {
      selected.clear();
      selInfo.clear();
      selected.add(e.name);
      selInfo.set(e.name, e);
    }
    applyHighlight();
    emitSelect();
  }

  // ---- rows -------------------------------------------------------
  function rowFor(e, i) {
    const row = el('div', {
      class: 'fs-row fs-vrow' + (e.dir ? ' dir' : '')
        + (selected.has(e.name) ? ' sel' : ''),
      style: 'top:' + (i * FS_ROW_H) + 'px',
      'data-name': e.name,
    }, makeIcon(iconFor(e), 'sm'), el('span', { class: 'fs-nm' }, e.name),
       e.dir ? null : el('span', { class: 'fs-sz' }, humanSize(e.size || 0)));
    row.addEventListener('click', () => {
      if (e.dir && dirClickOpens) { onDirOpen(e); return; }
      selectEntry(e);
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
    setSelection: (names) => {
      selected.clear();
      selInfo.clear();
      for (const n of (names || [])) { selected.add(n); }
      applyHighlight();
    },
    clearSelection: () => {
      selected.clear();
      selInfo.clear();
      applyHighlight();
      emitSelect();
    },
    current: () => ({ path: curPath, parent: meta.parent,
                      sandboxed: meta.sandboxed, total }),
    destroy: () => ro.disconnect(),
  };
}
