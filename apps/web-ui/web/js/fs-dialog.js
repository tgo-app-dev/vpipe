// Reusable file open/save dialog that browses the host filesystem via
// GET /api/fs/list. The endpoint speaks the session's namespace: when
// the web-ui filesystem sandbox is active the tree is chroot-like (root
// == "/", real host paths never reach the client); otherwise the paths
// ARE host paths. The dialog is path-namespace agnostic -- it just walks
// whatever `path` / `parent` the server hands back.
//
// The directory listing itself (paging + virtualization + loading) is the
// shared createFsList component (fs-list.js), so the dialog gets the same
// large-directory handling as the file-browser view; extension filters are
// applied server-side (via the component's exts) so paging stays correct.
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
import { t } from './i18n.js';
import { api } from './api.js';
import { createFsList, joinPath } from './fs-list.js';

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

export function openFsDialog(opts = {}) {
  const mode = opts.mode === 'save' ? 'save' : 'open';
  const kind = opts.kind === 'dir' ? 'dir' : 'file';
  const multi = mode === 'open' && kind === 'file' && !!opts.multi;
  const filters = Array.isArray(opts.filters) ? opts.filters : [];
  const onPick = typeof opts.onPick === 'function' ? opts.onPick : () => {};

  // ---- state --------------------------------------------------------
  let cur = '';               // current directory (server-echoed)
  let parent = '';
  let sandboxed = false;
  let selCount = 0;
  let filterIdx = filters.length ? 0 : -1;   // -1 == all files
  let close = () => {};
  // True while a nested prompt/confirm (New folder, Overwrite?) is
  // stacked on top. It neutralizes THIS dialog's Enter/Escape so the
  // keys act on the overlay alone (both share document-level handlers).
  let overlayOpen = false;

  const activeExts = () =>
    (filterIdx >= 0 && filters[filterIdx]) ? filters[filterIdx].exts : [];

  // ---- DOM ----------------------------------------------------------
  const upBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.up') }, makeIcon('levelup', 'sm'));
  const refreshBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.refresh') }, makeIcon('refresh', 'sm'));
  // "New folder" -- only where creating one is meaningful: a save
  // dialog or an explicit destination-folder pick (not open+file).
  const canMkdir = mode === 'save' || kind === 'dir';
  const mkdirBtn = canMkdir
    ? el('button', { class: 'btn ghost mini fs-mkdir', type: 'button',
        title: t('fs.new_folder') }, makeIcon('folder', 'sm'),
        el('span', { class: 'fs-mk-lbl' }, t('fs.new_folder')))
    : null;
  const pathIn = el('input', { type: 'text', class: 'fs-path',
    autocomplete: 'off', spellcheck: 'false' });
  // Hourglass shown while the backend enumerates a directory (a big folder,
  // or a lazily-loaded page while scrolling). Driven by the list's onLoading.
  const busyEl = el('span', { class: 'fs-busy' }, makeIcon('hourglass', 'sm'));
  busyEl.hidden = true;
  const bar = el('div', { class: 'fs-bar' }, upBtn, refreshBtn, mkdirBtn,
    pathIn, busyEl);

  const nameIn = el('input', { type: 'text', class: 'fs-name',
    placeholder: t('fs.filename'), value: opts.defaultName || '' });

  // ---- shared virtualized/paged list --------------------------------
  const list = createFsList({
    multi,
    selectableDirs: false,     // the dialog never "selects" a directory
    dirClickOpens: true,       // single click on a directory navigates in
    exts: activeExts(),
    onDirOpen: (e) => navigate(joinPath(cur, e.name)),
    onFileActivate: (e) => {
      // Double-click a file = confirm it directly.
      if (mode === 'save') { nameIn.value = e.name; tryConfirm(); return; }
      if (kind === 'dir') { commit(cur); return; }
      commit(multi ? [joinPath(cur, e.name)] : joinPath(cur, e.name));
    },
    onSelect: (names, entries) => {
      selCount = names.length;
      if (mode === 'save' && entries.length === 1 && !entries[0].dir) {
        nameIn.value = entries[0].name;
      }
      renderStatus();
    },
    onMount: (m) => navigate(m.path),
    onLoading: (b) => { busyEl.hidden = !b; },
  });
  const listBox = el('div', { class: 'fs-list' }, list.el);

  let filterSel = null;
  if (filters.length) {
    filterSel = el('select', { class: 'fs-filter' });
    filters.forEach((f, i) =>
      filterSel.append(el('option', { value: String(i) }, f.label)));
    filterSel.append(el('option', { value: '-1' }, t('fs.all_files')));
    filterSel.value = String(filterIdx);
    filterSel.addEventListener('change', () => {
      filterIdx = parseInt(filterSel.value, 10);
      list.setExts(activeExts());   // reloads the current dir server-filtered
    });
  }
  const footRow = el('div', { class: 'fs-foot-row' });
  if (mode === 'save') {
    footRow.append(el('label', { class: 'fs-name-lbl' }, t('fs.filename')),
      nameIn);
  }
  if (filterSel) { footRow.append(filterSel); }

  const statusEl = el('div', { class: 'fs-status' });
  const body = el('div', { class: 'fs-dialog' }, bar, listBox,
    footRow, statusEl);

  // ---- helpers ------------------------------------------------------
  function commit(result) { onPick(result); close(); }

  async function tryConfirm() {
    if (overlayOpen) { return; }   // a prompt/confirm owns the keyboard
    if (kind === 'dir') { commit(cur); return; }
    if (mode === 'save') {
      const nm = nameIn.value.trim();
      if (!nm) { toast(t('fs.name_required'), 'error'); return; }
      // Confirm before clobbering an existing file of the same name. The
      // in-view list is paged (may not hold every entry), so ask the
      // server for the whole listing of `cur` -- omitting offset/limit
      // returns it un-windowed. A same-named directory is left for the
      // backend/stage to reject; if the probe fails we don't block saving.
      let clash = false;
      try {
        const all = await api.fsList(cur);
        clash = (all.entries || []).some((e) => !e.dir && e.name === nm);
      } catch (e) { /* can't tell -- fall through and let the save proceed */ }
      if (clash) {
        openConfirm({
          title: t('fs.overwrite_title'),
          message: t('fs.overwrite_msg', { name: nm }),
          confirmLabel: t('common.overwrite'),
          onOk: () => commit(joinPath(cur, nm)),
        });
        return;
      }
      commit(joinPath(cur, nm));
      return;
    }
    // open + file
    const names = list.getSelectionNames();
    if (names.length === 0) {
      toast(t('fs.select_required'), 'error');
      return;
    }
    const picks = names.map((n) => joinPath(cur, n));
    commit(multi ? picks : picks[0]);
  }

  // A single-line text prompt stacked over the dialog (New folder).
  // `onOk(value, closePrompt)` performs the action and closes the prompt
  // itself on success, so a failed create can keep it open.
  function openPrompt({ title, label, confirmLabel, onOk }) {
    overlayOpen = true;
    const input = el('input', { type: 'text', class: 'fs-name',
      autocomplete: 'off', spellcheck: 'false' });
    const pbody = el('div', { class: 'fs-prompt' },
      el('label', { class: 'fs-name-lbl' }, label), input);
    const done = (c) => { overlayOpen = false; c(); };
    openModal({
      title, className: 'fs-prompt-modal', body: pbody,
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => done(c) },
        { label: confirmLabel, kind: 'primary', onClick: (c) => {
            const v = input.value.trim();
            if (!v) { return; }
            onOk(v, () => done(c));
          } },
      ],
    });
    setTimeout(() => input.focus(), 0);
  }

  // A yes/no confirmation stacked over the dialog (Overwrite?). Same
  // overlayOpen guard as openPrompt.
  function openConfirm({ title, message, confirmLabel, onOk }) {
    overlayOpen = true;
    const done = (c) => { overlayOpen = false; c(); };
    openModal({
      title, className: 'fs-confirm-modal',
      body: el('div', { class: 'fs-confirm' }, message),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => done(c) },
        { label: confirmLabel, kind: 'primary',
          onClick: (c) => { done(c); onOk(); } },
      ],
    });
  }

  // Create `name` under the current directory, then step into it (or the
  // already-present folder) so the user can save there straight away.
  async function doMkdir(name, closePrompt) {
    try {
      const res = await api.fsMkdir(cur, name);
      closePrompt();
      navigate(res && res.path ? res.path : joinPath(cur, name));
    } catch (e) {
      toast(t('fs.mkdir_failed', { msg: e.message }), 'error');
    }
  }

  function renderStatus() {
    clear(statusEl);
    statusEl.append(sandboxed
      ? el('span', { class: 'fs-badge' }, t('fs.sandboxed'))
      : el('span', { class: 'fs-badge native' }, t('fs.native')));
    if (multi && selCount) {
      statusEl.append(el('span', { class: 'fs-count' },
        t('fs.n_selected', { n: selCount })));
    }
  }

  async function navigate(path) {
    try {
      const info = await list.load(path);
      if (!info) { return; }          // superseded by a later navigate
      cur = info.path;
      parent = info.parent;
      sandboxed = info.sandboxed;
      pathIn.value = cur;
      upBtn.disabled = !parent;
      renderStatus();
    } catch (e) {
      toast(t('fs.list_failed', { msg: e.message }), 'error');
      // Fall back to the session default directory once, so a bad seed
      // path (e.g. a not-yet-created save dir) still opens.
      if (path) { navigate(''); }
    }
  }

  refreshBtn.addEventListener('click', () => navigate(cur));

  if (mkdirBtn) {
    mkdirBtn.addEventListener('click', () => {
      if (overlayOpen) { return; }
      openPrompt({
        title: t('fs.new_folder'),
        label: t('fs.folder_name'),
        confirmLabel: t('common.create'),
        onOk: (name, closeP) => doMkdir(name, closeP),
      });
    });
  }

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
    if (parent) { navigate(parent); }
  });

  const title = opts.title
    || (mode === 'save' ? t('fs.save_title')
        : kind === 'dir' ? t('fs.pick_folder') : t('fs.open_title'));
  const confirmLabel = mode === 'save' ? t('common.save')
    : kind === 'dir' ? t('fs.select_folder') : t('common.open');

  // list.destroy() (drops the ResizeObserver) must run on EVERY close path
  // -- confirm, Cancel, Escape, and backdrop click. openModal routes
  // Escape/backdrop through the cancel action, so destroying there + in the
  // dialog's own close() (used by commit) covers them all; destroy() is
  // idempotent so the overlap is harmless.
  const closeModal = openModal({
    title,
    className: 'fs-modal',
    body,
    actions: [
      // While a nested prompt/confirm is up, swallow this dialog's own
      // Escape so it dismisses only the overlay -- both share the document
      // keydown handler, so Escape would otherwise close this dialog too.
      // Otherwise destroy the list (drops its ResizeObserver) before close.
      { label: t('common.cancel'), cancel: true,
        onClick: (c) => {
          if (overlayOpen) { return; }
          list.destroy();
          c();
        } },
      { label: confirmLabel, kind: 'primary', onClick: () => tryConfirm() },
    ],
  });
  close = () => { list.destroy(); closeModal(); };

  navigate(opts.startDir || '');
  setTimeout(() => { (mode === 'save' ? nameIn : pathIn).focus(); }, 0);
}
