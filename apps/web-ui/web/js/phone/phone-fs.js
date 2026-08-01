// One-column filesystem walker for the phone, over the same
// /api/fs/list the desktop dialog and file browser use -- so it speaks
// the session's VIRTUAL namespace and is correct whether or not the
// filesystem sandbox is on.
//
// The desktop equivalents are a three-column workspace and a two-pane
// dialog; neither shape survives a phone. What is left when you take the
// panes away is a stack: one directory at a time, tap a folder to
// descend, tap the path bar's arrow to come back up. That is the whole
// component, and it is shared by everything here that needs to point at
// a file -- loading a pipeline, saving one, filling a path config field,
// and the Files view.
//
// Path joining is inlined rather than imported from fs-list.js: it is
// two lines, and that module carries the desktop's virtualized list.

import { el, clear, openModal } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t } from '../i18n.js';
import { iconFor } from '../fs-kinds.js';

export function joinPath(dir, name) {
  if (!dir || dir === '/') { return '/' + name; }
  return dir.replace(/[/\\]+$/, '') + '/' + name;
}

export function dirOf(p) {
  const s = String(p || '');
  const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
  return i < 0 ? '' : (s.slice(0, i) || '/');
}

export function baseOf(p) {
  const s = String(p || '');
  const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
  return i < 0 ? s : s.slice(i + 1);
}

// Mount a browser into `host`.
//   start      directory to open first ('' = the server's default root)
//   exts       ['.vpipeline', ...] -- files kept; directories always show
//   onFile(p, entry)   a file was tapped
//   onDir(p)           the listing moved (also fires for the first load)
//   pickDirs   true -> tapping a directory row SELECTS it (onFile) rather
//              than descending; the path bar's arrow still navigates
// Returns { go(dir), path() }.
export function mountFsBrowser(host, opts = {}) {
  const onFile = opts.onFile || (() => {});
  const onDir = opts.onDir || (() => {});
  const pathEl = el('div', { class: 'ph-sheet-path' });
  const listEl = el('div', { class: 'ph-sheet-list' });
  clear(host).append(pathEl, listEl);
  let cur = '';
  // Only the newest listing may paint: a fast tapper can leave two
  // fetches in flight, and the slower one must not win.
  let gen = 0;

  async function go(dir) {
    const mine = ++gen;
    clear(listEl).append(
      el('div', { class: 'ph-sheet-hint' }, t('common.loading')));
    let d;
    try {
      d = await api.fsList(dir, opts.exts ? { exts: opts.exts } : {});
    } catch (e) {
      if (mine !== gen) { return; }
      clear(listEl).append(el('div', { class: 'ph-sheet-hint' },
        t('fs.list_failed', { msg: e.message })));
      return;
    }
    if (mine !== gen) { return; }
    cur = d.path || dir || '';
    clear(pathEl).append(
      d.parent
        ? el('button', { class: 'ph-up', title: t('fs.up'),
                         onclick: () => go(d.parent) }, '↑')
        : null,
      el('span', { class: 'ph-sheet-pathtext' }, cur || '/'));
    clear(listEl);
    for (const m of (d.mounts || [])) {
      listEl.append(el('button', { class: 'ph-sheet-item',
        onclick: () => go(m.path) },
        makeIcon('folder', 'sm'),
        el('span', { class: 'ph-sheet-name' }, m.name)));
    }
    const entries = d.entries || [];
    if (!entries.length && !(d.mounts || []).length) {
      listEl.append(el('div', { class: 'ph-sheet-hint' }, t('fs.empty')));
    }
    for (const e of entries) {
      const full = joinPath(cur, e.name);
      listEl.append(el('button', { class: 'ph-sheet-item',
        onclick: () => {
          if (e.dir && !opts.pickDirs) { go(full); }
          else { onFile(full, e); }
        } },
        makeIcon(iconFor(e), 'sm'),
        el('span', { class: 'ph-sheet-name' }, e.name),
        e.dir && opts.pickDirs
          ? el('button', { class: 'ph-into', title: t('common.open'),
              onclick: (ev) => { ev.stopPropagation(); go(full); } }, '›')
          : null));
    }
    onDir(cur);
  }

  go(opts.start || '');
  return { go, path: () => cur };
}

// The walker in a modal, for "point at one file/folder and come back".
//   title, start, exts, pickDirs   as above
//   onPick(path)                   chosen; the sheet closes first
//   nameField                      show a filename box (save flows); its
//                                  value is joined onto the directory
//   defaultName                    seed for that box
export function openFsSheet(opts = {}) {
  const host = el('div', {});
  const nameIn = opts.nameField
    ? el('input', { type: 'text', class: 'ph-input',
        value: opts.defaultName || '', autocapitalize: 'off',
        autocorrect: 'off', spellcheck: 'false' })
    : null;
  let browser = null;
  const finish = (p) => {
    if (!p) { return; }
    close();
    opts.onPick(p);
  };
  const actions = [
    { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
  ];
  if (nameIn) {
    actions.push({ label: opts.confirmLabel || t('common.save'),
      kind: 'primary',
      onClick: () => {
        const n = nameIn.value.trim();
        if (n) { finish(joinPath(browser.path(), n)); }
      } });
  } else if (opts.pickDirs) {
    actions.push({ label: opts.confirmLabel || t('fs.select_folder'),
      kind: 'primary', onClick: () => finish(browser.path()) });
  }
  const close = openModal({
    title: opts.title || t('fs.open_title'),
    className: 'ph-modal',
    body: el('div', {}, host,
      nameIn
        ? el('div', {},
            el('div', { class: 'ph-sheet-label' }, t('fs.filename')), nameIn)
        : null),
    actions,
  });
  browser = mountFsBrowser(host, {
    start: opts.start, exts: opts.exts, pickDirs: opts.pickDirs,
    onFile: (p, e) => {
      // With a filename box open, tapping a file fills the box rather
      // than committing -- the confirm button is the commit, and an
      // accidental tap must not overwrite the wrong file.
      if (nameIn && !e.dir) { nameIn.value = baseOf(p); return; }
      if (opts.pickDirs && e.dir) { finish(p); return; }
      if (!e.dir) { finish(p); }
    },
  });
  return close;
}
