// File browser view: a three-column workspace over the session's
// filesystem namespace (sandbox chroot-like "/"-root, or native paths --
// same namespace the file open/save dialog speaks). The columns are
// split 2 : 3 : 3 as { folder tree, directory listing, preview }, under a
// top toolbar (up one level / refresh / rename / new folder).
//
// The tree is the jump-around navigator (lazy, expand-on-demand); the
// listing shows the current directory's entries (dirs + files); the
// preview renders the selected file -- image, one screen of text, or an
// audio / video player. Selecting an entry never leaves the view; only
// a directory drill-in (double-click / tree click) changes the listing.
//
// All bytes come from the backend: GET /api/fs/list (one directory),
// GET /api/fs/file (raw file bytes, Range-aware) via api.fsFileUrl /
// api.fsText, and POST /api/fs/{mkdir,rename} for the two mutating
// toolbar actions.

import { el, clear, openModal, toast } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t } from '../i18n.js';
import { createFsList, humanSize, joinPath } from '../fs-list.js';

// Extension -> preview category. Lower-case, no leading dot. Anything
// not listed has no inline preview (the pane offers a download instead).
const IMAGE = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'svg',
  'tiff', 'tif', 'heic', 'ico']);
const AUDIO = new Set(['wav', 'mp3', 'flac', 'aac', 'm4a', 'ogg', 'opus',
  'aiff', 'aif']);
const VIDEO = new Set(['mp4', 'm4v', 'mov', 'webm', 'mkv', 'avi', 'ts',
  'flv', 'mpg', 'mpeg']);
const TEXT = new Set(['txt', 'md', 'log', 'csv', 'srt', 'vtt', 'json',
  'xml', 'yaml', 'yml', 'html', 'htm', 'js', 'css', 'c', 'cc', 'h', 'hpp',
  'py', 'sh', 'ini', 'conf', 'toml', 'ts', 'tsx', 'jsx']);

// One screen of text: bytes fetched for a text preview (Range-capped so a
// huge file is never pulled whole).
const TEXT_PREVIEW_BYTES = 64 * 1024;

// The list pane's paging + virtualization live in the shared createFsList
// component (fs-list.js), used by the open/save dialog too. This view keeps
// one live component across (re)mounts and destroys the prior one, since the
// view has no unmount hook (prevents a leaked ResizeObserver per nav switch).
let fbList = null;

function extOf(name) {
  const i = String(name || '').lastIndexOf('.');
  return i >= 0 ? name.slice(i + 1).toLowerCase() : '';
}

function categorize(name) {
  const e = extOf(name);
  if (IMAGE.has(e)) { return 'image'; }
  if (AUDIO.has(e)) { return 'audio'; }
  if (VIDEO.has(e)) { return 'video'; }
  if (TEXT.has(e))  { return 'text'; }
  return null;
}

function iconFor(entry) {
  if (entry.dir) { return 'folder'; }
  const c = categorize(entry.name);
  return c === 'image' ? 'image'
    : c === 'audio' ? 'audio'
    : c === 'video' ? 'video' : 'file';
}

// ---- server forward-slash path helpers (namespace-agnostic) ----------
// joinPath / humanSize are shared from fs-list.js.

function isPrefix(base, full) {
  if (!base || base === '/') { return true; }
  const b = base.replace(/\/+$/, '');
  return full === b || full.startsWith(b + '/');
}

function relSegments(base, full) {
  const b = (base === '/' || !base) ? '' : base.replace(/\/+$/, '');
  const rest = (b && full.startsWith(b)) ? full.slice(b.length) : full;
  return rest.split('/').filter(Boolean);
}

// ---------------------------------------------------------------------

export function mountFileBrowser(container) {
  clear(container);
  const rootEl = el('div', { class: 'fb' });
  container.append(rootEl);

  // ---- state ------------------------------------------------------
  let curDir  = '';
  let curParent = '';           // parent of curDir ('' == at a root)
  let selectedName = null;      // highlighted list entry (file or dir)
  let selectedIsDir = false;
  let sandboxed = false;
  let treeRoots = [];           // top-level tree nodes (home + mounts)
  let currentNode = null;       // tree node matching curDir, if revealed

  // ---- toolbar ----------------------------------------------------
  const upBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.up') }, makeIcon('levelup', 'sm'));
  const refreshBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fs.refresh') }, makeIcon('refresh', 'sm'));
  const renameBtn = el('button', { class: 'btn ghost mini', type: 'button',
    title: t('fb.rename') }, makeIcon('edit', 'sm'),
    el('span', {}, t('fb.rename')));
  const newFolderBtn = el('button', { class: 'btn ghost mini',
    type: 'button', title: t('fb.new_folder') },
    makeIcon('folder-plus', 'sm'), el('span', {}, t('fb.new_folder')));
  const pathIn = el('input', { type: 'text', class: 'fb-path',
    autocomplete: 'off', spellcheck: 'false' });
  const badge = el('span', { class: 'fs-badge' });
  const toolbar = el('div', { class: 'fb-toolbar' },
    upBtn, refreshBtn, pathIn, badge,
    el('span', { class: 'fb-tb-gap' }),
    renameBtn, newFolderBtn);

  // ---- panes ------------------------------------------------------
  const treeBody = el('div', { class: 'pane-body fb-tree' });
  const prevBody = el('div', { class: 'pane-body fb-preview' });

  // The list pane is the shared paged + virtualized component. A hourglass
  // in the "Files" pane header reflects its loading state (page 0 fetch, or
  // a lazily-loaded page while scrolling a big directory).
  const filesBusy = el('span', { class: 'fs-busy' },
    makeIcon('hourglass', 'sm'));
  filesBusy.hidden = true;
  if (fbList) { fbList.destroy(); }
  const list = createFsList({
    multi: false,
    selectableDirs: true,      // a directory can be selected (for rename)
    dirClickOpens: false,      // single-click selects; double-click opens
    iconFor,
    onDirOpen: (e) => enterDir(joinPath(curDir, e.name)),
    onFileActivate: (e) => showPreview(e),
    onSelect: (names, entries) => {
      const e = entries.length === 1 ? entries[0] : null;
      selectedName  = e ? e.name : null;
      selectedIsDir = e ? !!e.dir : false;
      updateToolbar();
      if (e && !e.dir) { showPreview(e); }
    },
    onMount: (m) => enterDir(m.path),
    onLoading: (b) => { filesBusy.hidden = !b; },
  });
  fbList = list;
  const listBody = el('div', { class: 'pane-body fb-list' }, list.el);

  const bodyEl = el('div', { class: 'fb-body' },
    el('div', { class: 'pane fb-col-tree' },
      el('div', { class: 'pane-head' },
        el('span', { class: 'title' }, t('fb.folders'))), treeBody),
    el('div', { class: 'pane fb-col-list' },
      el('div', { class: 'pane-head' },
        el('span', { class: 'title' }, t('fb.files')), filesBusy), listBody),
    el('div', { class: 'pane fb-col-preview' },
      el('div', { class: 'pane-head' },
        el('span', { class: 'title' }, t('fb.preview'))), prevBody));

  rootEl.append(toolbar, bodyEl);

  // ---- toolbar / badge state -------------------------------------
  function updateToolbar() {
    upBtn.disabled = !curParent;
    renameBtn.disabled = !selectedName;
  }
  function renderBadge() {
    clear(badge);
    badge.className = 'fs-badge' + (sandboxed ? '' : ' native');
    badge.textContent = sandboxed ? t('fs.sandboxed') : t('fs.native');
  }

  // ---- preview ----------------------------------------------------
  function renderPreviewEmpty() {
    clear(prevBody);
    prevBody.append(el('div', { class: 'fb-preview-empty' },
      t('fb.pick_to_preview')));
  }

  function metaLine(e) {
    return el('div', { class: 'fb-file-meta' },
      e.name + '  ·  ' + humanSize(e.size || 0));
  }

  function showUnpreviewable(e, url) {
    clear(prevBody);
    prevBody.append(el('div', { class: 'fb-preview-empty' },
      metaLine(e),
      el('div', { class: 'fb-no-preview' }, t('fb.no_preview')),
      el('a', { class: 'btn', href: url, download: e.name },
        makeIcon('load', 'sm'), el('span', {}, t('fb.download')))));
  }

  function showPreview(e) {
    const vpath = joinPath(curDir, e.name);
    const url = api.fsFileUrl(vpath);
    const cat = categorize(e.name);
    clear(prevBody);

    if (cat === 'image') {
      const img = el('img', { class: 'fb-img', alt: e.name, src: url });
      img.addEventListener('error', () => showUnpreviewable(e, url));
      prevBody.append(el('div', { class: 'fb-preview-inner' },
        metaLine(e), el('div', { class: 'fb-media-wrap' }, img)));
    } else if (cat === 'video') {
      const v = el('video', { class: 'fb-video', controls: true,
        preload: 'metadata', src: url });
      v.addEventListener('error', () => showUnpreviewable(e, url));
      prevBody.append(el('div', { class: 'fb-preview-inner' },
        metaLine(e), el('div', { class: 'fb-media-wrap' }, v)));
    } else if (cat === 'audio') {
      const a = el('audio', { class: 'fb-audio', controls: true,
        preload: 'metadata', src: url });
      a.addEventListener('error', () => showUnpreviewable(e, url));
      prevBody.append(el('div', { class: 'fb-preview-inner audio' },
        metaLine(e), el('div', { class: 'fb-media-wrap audio' }, a)));
    } else if (cat === 'text') {
      const pre = el('pre', { class: 'fb-text allow-context-menu' },
        t('common.loading'));
      const note = el('div', { class: 'fb-note', hidden: true });
      prevBody.append(el('div', { class: 'fb-preview-inner' },
        metaLine(e), note, pre));
      api.fsText(vpath, TEXT_PREVIEW_BYTES).then((res) => {
        clear(pre).append(document.createTextNode(res.text));
        if (res.truncated) {
          note.hidden = false;
          note.textContent = t('fb.truncated',
            { n: humanSize(TEXT_PREVIEW_BYTES) });
        }
      }).catch((err) => {
        clear(pre).append(document.createTextNode(
          t('fb.preview_failed', { msg: err.message })));
      });
    } else {
      showUnpreviewable(e, url);
    }
  }

  // ---- directory tree (lazy) --------------------------------------
  function makeNode(vpath, name, depth) {
    const twisty = el('span', { class: 'fb-twisty' }, '▸');
    const nameEl = el('span', { class: 'fb-tnm' }, name);
    const row = el('div', { class: 'fb-tnode',
      style: 'padding-left:' + (depth * 12 + 6) + 'px' },
      twisty, makeIcon('folder', 'sm'), nameEl);
    const childBox = el('div', { class: 'fb-tchildren' });
    childBox.hidden = true;
    const node = { vpath, name, depth, expanded: false, loaded: false,
      children: null, row, childBox, twisty };
    twisty.addEventListener('click', (ev) => {
      ev.stopPropagation();
      toggleNode(node);
    });
    row.addEventListener('click', () => enterDir(vpath, node));
    return node;
  }

  function seedChildren(node, listing) {
    clear(node.childBox);
    node.children = [];
    const dirs = (listing.entries || []).filter((e) => e.dir);
    for (const e of dirs) {
      const child = makeNode(joinPath(node.vpath, e.name), e.name,
        node.depth + 1);
      node.children.push(child);
      node.childBox.append(child.row, child.childBox);
    }
    node.loaded = true;
    node.twisty.classList.toggle('leaf', dirs.length === 0);
  }

  async function ensureExpanded(node) {
    if (!node.loaded) {
      node.twisty.textContent = '···';   // ···
      try {
        // dirs_only: the tree never needs files, so a large directory
        // costs only its (few) sub-directories, not thousands of entries.
        seedChildren(node, await api.fsList(node.vpath, { dirsOnly: true }));
      } catch (err) {
        toast(t('fb.op_failed', { msg: err.message }), 'error');
        return;
      }
    }
    node.expanded = true;
    node.childBox.hidden = false;
    node.twisty.textContent = node.children.length ? '▾' : '';
  }

  function collapseNode(node) {
    node.expanded = false;
    node.childBox.hidden = true;
    if (node.children && node.children.length) {
      node.twisty.textContent = '▸';
    }
  }

  function toggleNode(node) {
    if (node.expanded) { collapseNode(node); } else { ensureExpanded(node); }
  }

  function setCurrentNode(node) {
    if (currentNode && currentNode.row) {
      currentNode.row.classList.remove('cur');
    }
    currentNode = node;
    if (node && node.row) {
      node.row.classList.add('cur');
      node.row.scrollIntoView({ block: 'nearest' });
    }
  }

  // Expand the tree from whichever root contains `vpath` down to it, so
  // the current directory is revealed + highlighted. Best-effort: stops
  // (and highlights the deepest match) if a segment isn't present.
  async function revealPath(vpath) {
    let node = treeRoots.find((r) => isPrefix(r.vpath, vpath))
      || treeRoots[0];
    if (!node) { return; }
    await ensureExpanded(node);
    for (const seg of relSegments(node.vpath, vpath)) {
      if (!node.children) { break; }
      const child = node.children.find((c) => c.name === seg);
      if (!child) { break; }
      await ensureExpanded(child);
      node = child;
    }
    setCurrentNode(node);
  }

  // Build the tree from a dirs-only root listing (cheap even at a huge
  // root). Returns the resolved root path, or null on failure.
  async function buildTree() {
    clear(treeBody);
    treeRoots = [];
    currentNode = null;
    let d;
    try {
      d = await api.fsList('', { dirsOnly: true });
    } catch (err) {
      treeBody.append(el('div', { class: 'fs-empty' },
        t('fb.op_failed', { msg: err.message })));
      return null;
    }
    sandboxed = !!d.sandboxed;
    const rootPath = d.path || '/';
    const home = makeNode(rootPath, sandboxed ? '/' : (rootPath || '/'), 0);
    treeRoots.push(home);
    treeBody.append(home.row, home.childBox);
    // Seed the home root from this same (dirs-only) listing, open.
    seedChildren(home, d);
    home.expanded = true;
    home.childBox.hidden = false;
    home.twisty.textContent = home.children.length ? '▾' : '';
    // Mounts become sibling roots so paths outside the sandbox home are
    // still reachable / revealable in the tree.
    for (const m of (d.mounts || [])) {
      const mn = makeNode(m.path, m.name, 0);
      treeRoots.push(mn);
      treeBody.append(mn.row, mn.childBox);
    }
    return rootPath;
  }

  // ---- navigation -------------------------------------------------
  // Load `vpath` into the listing pane via the shared component (fetches
  // the first page, sizes the scroll area, renders the initial window;
  // further pages load lazily on scroll). Returns whether it loaded.
  async function loadListing(vpath) {
    let info;
    try {
      info = await list.load(vpath);
    } catch (err) {
      toast(t('fb.op_failed', { msg: err.message }), 'error');
      return false;
    }
    if (!info) { return false; }             // superseded by a later load
    curDir = info.path;
    curParent = info.parent;
    sandboxed = info.sandboxed;
    // list.load fired onSelect([]) -> selectedName/isDir already reset.
    pathIn.value = curDir;
    renderPreviewEmpty();
    renderBadge();
    updateToolbar();
    return true;
  }

  // Navigate the listing into `vpath` and reveal it in the tree. `node`,
  // when given (a tree-row click), is highlighted directly.
  async function enterDir(vpath, node) {
    const ok = await loadListing(vpath);
    if (!ok) { return; }
    if (node) { setCurrentNode(node); }
    else { revealPath(curDir); }
  }

  // ---- toolbar actions --------------------------------------------
  upBtn.addEventListener('click', () => {
    if (curParent) { enterDir(curParent); }
  });

  refreshBtn.addEventListener('click', async () => {
    await loadListing(curDir);
    if (currentNode && currentNode.loaded) {
      seedChildren(currentNode,
        await api.fsList(currentNode.vpath, { dirsOnly: true })
          .catch(() => ({ entries: [] })));
      if (currentNode.expanded) { currentNode.childBox.hidden = false; }
    }
  });

  pathIn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      enterDir(pathIn.value.trim());
    }
  });

  function promptName(title, initial, confirmLabel, onOk) {
    const input = el('input', { type: 'text', value: initial || '' });
    openModal({
      title,
      body: el('div', { class: 'fb-prompt' }, input),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: confirmLabel, kind: 'primary', onClick: (c) => {
          const v = input.value.trim();
          if (!v) { return; }
          c();
          onOk(v);
        } },
      ],
    });
    setTimeout(() => { input.focus(); input.select(); }, 0);
  }

  newFolderBtn.addEventListener('click', () => {
    promptName(t('fb.new_folder'), '', t('common.create'), async (name) => {
      try {
        await api.fsMkdir(curDir, name);
        toast(t('fb.folder_created'), 'ok');
        await loadListing(curDir);
        selectedName = name;
        selectedIsDir = true;
        list.setSelection([name]);   // highlight the new folder if in-window
        updateToolbar();
        // Reflect the new subdirectory in the tree.
        if (currentNode && currentNode.loaded) {
          seedChildren(currentNode,
            await api.fsList(currentNode.vpath, { dirsOnly: true })
              .catch(() => ({ entries: [] })));
          if (currentNode.expanded) { currentNode.childBox.hidden = false; }
        }
      } catch (err) {
        toast(t('fb.op_failed', { msg: err.message }), 'error');
      }
    });
  });

  renameBtn.addEventListener('click', () => {
    if (!selectedName) { toast(t('fb.select_item'), 'error'); return; }
    const name = selectedName;
    const wasDir = selectedIsDir;
    promptName(t('fb.rename'), name, t('common.rename'), async (to) => {
      if (to === name) { return; }
      try {
        await api.fsRename(joinPath(curDir, name), to);
        toast(t('fb.renamed'), 'ok');
        await loadListing(curDir);
        selectedName = to;
        selectedIsDir = wasDir;
        list.setSelection([to]);     // keep the renamed item highlighted
        updateToolbar();
        if (wasDir && currentNode && currentNode.loaded) {
          seedChildren(currentNode,
            await api.fsList(currentNode.vpath, { dirsOnly: true })
              .catch(() => ({ entries: [] })));
          if (currentNode.expanded) { currentNode.childBox.hidden = false; }
        }
      } catch (err) {
        toast(t('fb.op_failed', { msg: err.message }), 'error');
      }
    });
  });

  // ---- boot -------------------------------------------------------
  (async () => {
    const rootPath = await buildTree();
    if (rootPath !== null) {
      await loadListing(rootPath || '');
      setCurrentNode(treeRoots[0] || null);
    }
  })();
}
