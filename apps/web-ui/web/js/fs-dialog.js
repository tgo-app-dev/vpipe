// Reusable file open/save dialog that browses the host filesystem via
// GET /api/fs/list. The endpoint speaks the session's namespace: when
// the web-ui filesystem sandbox is active the tree is chroot-like (root
// == "/", real host paths never reach the client); otherwise the paths
// ARE host paths. The dialog is path-namespace agnostic -- it just walks
// whatever `path` / `parent` the server hands back.
//
//   openFsDialog({
//     mode: 'open' | 'save',        // save adds a filename field
//     kind: 'file' | 'dir',         // dir picks the browsed directory
//     multi: false,                 // open+file: select several files
//     filters: [{label, exts}],     // optional extension filters
//     startDir: '/some/dir',        // directory to open in
//     defaultName: 'out.png',       // save: prefill; open: preselect
//     title: '...',                 // optional heading override
//     onPick: (result) => {},       // string, or array when multi
//   })

import { el, clear, openModal, toast } from './dom.js';
import { makeIcon } from './icons.js';
import { api } from './api.js';
import { t } from './i18n.js';

// Category -> extension set. Kept in sync with the backend `path_filter`
// keywords a stage may declare (stage-config.h). Lower-case, dot-led.
const CATEGORY_EXTS = {
  image: ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.ppm',
          '.pgm', '.tiff', '.tif', '.heic'],
  audio: ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.opus',
          '.aiff', '.aif'],
  video: ['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.ts',
          '.flv', '.mpg', '.mpeg'],
  text:  ['.txt', '.md', '.json', '.csv', '.log', '.yaml', '.yml',
          '.xml', '.srt', '.vtt'],
};

// Build a filter list for a backend `path_filter` category keyword.
// Returns [] for an unknown/empty category (dialog shows all files).
export function filterForCategory(cat) {
  const exts = CATEGORY_EXTS[cat];
  if (!exts) { return []; }
  return [{ label: t('fs.filter_' + cat) + ' (' + exts.slice(0, 4).join(' ')
              + '…)', exts }];
}

// Split a path value into { dir, name } for seeding the dialog. A value
// with no separator is treated as a bare filename in the default dir.
export function splitPath(p) {
  const s = String(p || '');
  const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
  if (i < 0) { return { dir: '', name: s }; }
  return { dir: s.slice(0, i) || '/', name: s.slice(i + 1) };
}

// Truncate a long string in the MIDDLE (keep head + tail). Used for a
// mount's real host path so it keeps BOTH its leading "/" and its
// meaningful tail. Plain LTR text -- bidi-safe, unlike the CSS
// direction:rtl left-ellipsis it replaces (which dropped the leading "/").
function truncateMiddle(s, max) {
  s = String(s || '');
  if (s.length <= max) { return s; }
  const head = Math.ceil((max - 1) * 0.4);
  const tail = max - 1 - head;
  return s.slice(0, head) + '…' + s.slice(s.length - tail);
}

function humanSize(n) {
  if (!(n > 0)) { return '0 B'; }
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(v < 10 ? 1 : 0)) + ' ' + u[i];
}

// Join a directory with a child name in the server's forward-slash
// namespace (works for both the virtual "/" root and native paths).
function joinPath(dir, name) {
  if (!dir || dir === '/') { return '/' + name; }
  return dir.replace(/[\/\\]+$/, '') + '/' + name;
}

export function openFsDialog(opts = {}) {
  const mode = opts.mode === 'save' ? 'save' : 'open';
  const kind = opts.kind === 'dir' ? 'dir' : 'file';
  const multi = mode === 'open' && kind === 'file' && !!opts.multi;
  const filters = Array.isArray(opts.filters) ? opts.filters : [];
  const onPick = typeof opts.onPick === 'function' ? opts.onPick : () => {};

  // ---- state --------------------------------------------------------
  let cur = '';               // current directory (server-echoed)
  let sandboxed = false;
  let data = { entries: [], parent: '' };
  const selected = new Set();  // selected file names (relative to cur)
  let filterIdx = filters.length ? 0 : -1;   // -1 == all files
  let close = () => {};

  // ---- DOM ----------------------------------------------------------
  const upBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.up') }, makeIcon('levelup', 'sm'));
  const refreshBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.refresh') }, makeIcon('refresh', 'sm'));
  const pathIn = el('input', { type: 'text', class: 'fs-path',
    autocomplete: 'off', spellcheck: 'false' });
  // Hourglass shown while the backend enumerates the directory (a big
  // folder can take a moment); hidden otherwise.
  const busyEl = el('span', { class: 'fs-busy' }, makeIcon('hourglass', 'sm'));
  busyEl.hidden = true;
  const bar = el('div', { class: 'fs-bar' }, upBtn, refreshBtn, pathIn,
    busyEl);

  const listEl = el('div', { class: 'fs-list' });

  const nameIn = el('input', { type: 'text', class: 'fs-name',
    placeholder: t('fs.filename'), value: opts.defaultName || '' });
  let filterSel = null;
  if (filters.length) {
    filterSel = el('select', { class: 'fs-filter' });
    filters.forEach((f, i) =>
      filterSel.append(el('option', { value: String(i) }, f.label)));
    filterSel.append(el('option', { value: '-1' }, t('fs.all_files')));
    filterSel.value = String(filterIdx);
    filterSel.addEventListener('change', () => {
      filterIdx = parseInt(filterSel.value, 10);
      renderList();
    });
  }
  const footRow = el('div', { class: 'fs-foot-row' });
  if (mode === 'save') {
    footRow.append(el('label', { class: 'fs-name-lbl' }, t('fs.filename')),
      nameIn);
  }
  if (filterSel) { footRow.append(filterSel); }

  const statusEl = el('div', { class: 'fs-status' });
  const body = el('div', { class: 'fs-dialog' }, bar, listEl,
    footRow, statusEl);

  // ---- helpers ------------------------------------------------------
  function matchFilter(name) {
    if (filterIdx < 0 || !filters[filterIdx]) { return true; }
    const lc = name.toLowerCase();
    return filters[filterIdx].exts.some((e) => lc.endsWith(e));
  }

  function commit(result) { onPick(result); close(); }

  function tryConfirm() {
    if (kind === 'dir') { commit(cur); return; }
    if (mode === 'save') {
      const nm = nameIn.value.trim();
      if (!nm) { toast(t('fs.name_required'), 'error'); return; }
      commit(joinPath(cur, nm));
      return;
    }
    // open + file
    if (selected.size === 0) {
      toast(t('fs.select_required'), 'error');
      return;
    }
    const picks = [...selected].map((n) => joinPath(cur, n));
    commit(multi ? picks : picks[0]);
  }

  function renderStatus() {
    clear(statusEl);
    statusEl.append(sandboxed
      ? el('span', { class: 'fs-badge' }, t('fs.sandboxed'))
      : el('span', { class: 'fs-badge native' }, t('fs.native')));
    if (multi && selected.size) {
      statusEl.append(el('span', { class: 'fs-count' },
        t('fs.n_selected', { n: selected.size })));
    }
  }

  function renderList() {
    clear(listEl);
    upBtn.disabled = !data.parent;
    // Granted pass-through "mounts" (--white-list-path) pinned on top so
    // they're reachable from anywhere in the sandbox tree.
    const mounts = data.mounts || [];
    for (const m of mounts) {
      const row = el('div', { class: 'fs-row mount', title: m.path },
        makeIcon('folder', 'sm'),
        el('span', { class: 'fs-nm' }, m.name),
        el('span', { class: 'fs-mount-path' }, truncateMiddle(m.path, 42)));
      row.addEventListener('click', () => navigate(m.path));
      listEl.append(row);
    }
    const entries = data.entries || [];
    let shown = mounts.length;
    for (const e of entries) {
      if (!e.dir && !matchFilter(e.name)) { continue; }
      shown++;
      const isSel = !e.dir && selected.has(e.name);
      const row = el('div',
        { class: 'fs-row' + (e.dir ? ' dir' : '') + (isSel ? ' sel' : '') },
        makeIcon(e.dir ? 'folder' : 'file', 'sm'),
        el('span', { class: 'fs-nm' }, e.name),
        e.dir ? null
              : el('span', { class: 'fs-sz' }, humanSize(e.size || 0)));
      if (e.dir) {
        row.addEventListener('click', () => navigate(joinPath(cur, e.name)));
      } else {
        row.addEventListener('click', () => {
          if (multi) {
            if (selected.has(e.name)) { selected.delete(e.name); }
            else { selected.add(e.name); }
          } else {
            selected.clear();
            selected.add(e.name);
            if (mode === 'save') { nameIn.value = e.name; }
          }
          renderList();
          renderStatus();
        });
        row.addEventListener('dblclick', () => {
          selected.clear();
          selected.add(e.name);
          tryConfirm();
        });
      }
      listEl.append(row);
    }
    if (!shown) {
      listEl.append(el('div', { class: 'fs-empty' }, t('fs.empty')));
    }
    renderStatus();
  }

  async function navigate(path) {
    busyEl.hidden = false;          // hourglass on while the backend works
    try {
      const d = await api.fsList(path);
      data = d || { entries: [], parent: '' };
      cur = (d && typeof d.path === 'string') ? d.path : (path || '/');
      sandboxed = !!(d && d.sandboxed);
      selected.clear();
      pathIn.value = cur;
      renderList();
      busyEl.hidden = true;
    } catch (e) {
      toast(t('fs.list_failed', { msg: e.message }), 'error');
      // Fall back to the session default directory once, so a bad seed
      // path (e.g. a not-yet-created save dir) still opens. The retry owns
      // the hourglass from here; only clear it when there's no retry.
      if (path) { navigate(''); }
      else { busyEl.hidden = true; }
    }
  }

  // Refresh: re-list the current directory to pick up host-side changes.
  refreshBtn.addEventListener('click', () => navigate(cur));

  // Enter in the path box navigates; keep it from bubbling to the
  // modal's global Enter=confirm handler.
  pathIn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      ev.stopPropagation();
      navigate(pathIn.value.trim());
    }
  });
  upBtn.addEventListener('click', () => {
    if (data.parent || data.parent === '') {
      if (data.parent) { navigate(data.parent); }
    }
  });

  const title = opts.title
    || (mode === 'save' ? t('fs.save_title')
        : kind === 'dir' ? t('fs.pick_folder') : t('fs.open_title'));
  const confirmLabel = mode === 'save' ? t('common.save')
    : kind === 'dir' ? t('fs.select_folder') : t('common.open');

  close = openModal({
    title,
    className: 'fs-modal',
    body,
    actions: [
      { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
      { label: confirmLabel, kind: 'primary', onClick: () => tryConfirm() },
    ],
  });

  navigate(opts.startDir || '');
  setTimeout(() => { (mode === 'save' ? nameIn : pathIn).focus(); }, 0);
}
