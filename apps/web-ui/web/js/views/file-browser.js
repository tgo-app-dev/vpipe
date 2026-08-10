// File browser view: a three-column workspace over the session's
// filesystem namespace (sandbox chroot-like "/"-root, or native paths --
// same namespace the file open/save dialog speaks). The columns are
// { folder tree, directory listing, preview }, split 2 : 3 : 3 until the
// user drags a handle (remembered), under a top toolbar (up one level /
// refresh / rename / new folder).
//
// The tree is the jump-around navigator (lazy, expand-on-demand); the
// listing shows the current directory's entries (dirs + files); the
// preview renders the selected file -- image, one screen of text, or an
// audio / video player. Selecting an entry never leaves the view; only
// a directory drill-in (double-click / tree click / Right arrow) changes
// the listing.
//
// The keyboard drives the listing once it has focus: Up/Down (and
// PageUp/PageDown/Home/End) walk the entries, Left leaves for the parent
// directory and Right steps into the highlighted one. The mechanics live
// in the shared list component; this view supplies only what "out" and
// "into" mean here, since both also move the tree.
//
// All bytes come from the backend: GET /api/fs/list (one directory),
// GET /api/fs/file (raw file bytes, Range-aware) via api.fsFileUrl /
// api.fsText, and POST /api/fs/{mkdir,rename} for the two mutating
// toolbar actions.

import { el, clear, openModal, toast } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t } from '../i18n.js';
import { createFsList, joinPath } from '../fs-list.js';
// Extension -> preview category, shared with the phone browser so the
// two can't disagree about what previews.
import { iconFor } from '../fs-kinds.js';
import { createFsPreview } from '../fs-preview.js';

// The list pane's paging + virtualization live in the shared createFsList
// component (fs-list.js), used by the open/save dialog too. This view keeps
// one live component across (re)mounts and destroys the prior one, since the
// view has no unmount hook (prevents a leaked ResizeObserver per nav switch).
let fbList = null;

// ---- server forward-slash path helpers (namespace-agnostic) ----------
// joinPath is shared from fs-list.js.

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

// Last segment of a server path ('' at a root), so stepping OUT of a
// directory can land the cursor back on the one we came from.
function baseName(p) {
  const segs = String(p || '').split('/').filter(Boolean);
  return segs.length ? segs[segs.length - 1] : '';
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
    // Left / Right walk the hierarchy from the keyboard: out to the
    // parent, and into the highlighted directory (the component calls
    // onDirOpen for that, so Right and a double-click take one path).
    onNavUp: () => goUp(),
  });
  fbList = list;
  const listBody = el('div', { class: 'pane-body fb-list' }, list.el);

  const treePane = el('div', { class: 'pane fb-col-tree' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('fb.folders'))), treeBody);
  const listPane = el('div', { class: 'pane fb-col-list' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('fb.files')), filesBusy), listBody);
  const prevPane = el('div', { class: 'pane fb-col-preview' },
    el('div', { class: 'pane-head' },
      el('span', { class: 'title' }, t('fb.preview'))), prevBody);

  // ---- column widths ----------------------------------------------
  // The three panes are grid tracks in `fr`, with a drag handle in a
  // fixed track between each pair. A drag moves width BETWEEN the two
  // panes it sits between and leaves the third alone, which is what
  // makes the gesture local: widening the preview should not also
  // reshuffle the tree.
  const FB_COLS_KEY = 'vpipe_fb_cols';
  const FB_COLS_DEF = [2, 3, 3];      // the long-standing 2 : 3 : 3
  const COL_MIN = 120;                // px a pane may not shrink past
  let cols = FB_COLS_DEF.slice();
  try {
    const saved = JSON.parse(localStorage.getItem(FB_COLS_KEY));
    if (Array.isArray(saved) && saved.length === 3
        && saved.every((n) => Number.isFinite(n) && n > 0)) {
      cols = saved.map(Number);
    }
  } catch (e) { /* no storage, or a stale value -- keep the default */ }

  const panes = [treePane, listPane, prevPane];
  const handles = [0, 1].map((i) => el('div', {
    class: 'fb-split', 'data-i': String(i), role: 'separator',
    'aria-orientation': 'vertical', title: t('fb.resize_cols') }));
  const bodyEl = el('div', { class: 'fb-body' },
    treePane, handles[0], listPane, handles[1], prevPane);

  function applyCols() {
    bodyEl.style.gridTemplateColumns =
      cols[0] + 'fr var(--fb-split-w) ' + cols[1]
      + 'fr var(--fb-split-w) ' + cols[2] + 'fr';
  }
  function saveCols() {
    try { localStorage.setItem(FB_COLS_KEY, JSON.stringify(cols)); }
    catch (e) { /* storage blocked -- the drag still applies this session */ }
  }
  applyCols();

  // Drag redistributes the PIXELS of the adjacent pair, then converts
  // back to fr against the fr they already share. Going through pixels
  // is what makes the pane track the pointer 1:1 whatever the fr units
  // happen to be worth.
  for (const h of handles) {
    h.addEventListener('pointerdown', (ev) => {
      const i = +h.dataset.i;
      const w0 = panes[i].getBoundingClientRect().width;
      const w1 = panes[i + 1].getBoundingClientRect().width;
      const span = w0 + w1;
      const frSum = cols[i] + cols[i + 1];
      if (span <= 0) { return; }
      const x0 = ev.clientX;
      h.setPointerCapture(ev.pointerId);
      h.classList.add('dragging');
      ev.preventDefault();
      const move = (e) => {
        const want = Math.min(Math.max(w0 + (e.clientX - x0), COL_MIN),
          span - COL_MIN);
        cols[i] = frSum * (want / span);
        cols[i + 1] = frSum - cols[i];
        applyCols();
      };
      const end = (e) => {
        h.removeEventListener('pointermove', move);
        h.removeEventListener('pointerup', end);
        h.removeEventListener('pointercancel', end);
        h.classList.remove('dragging');
        try { h.releasePointerCapture(e.pointerId); } catch (er) {}
        cols = cols.map((n) => +n.toFixed(4));
        saveCols();
      };
      h.addEventListener('pointermove', move);
      h.addEventListener('pointerup', end);
      h.addEventListener('pointercancel', end);
    });
    // Double-click restores 2 : 3 : 3 -- the way back from a drag that
    // squeezed a pane to its minimum, without hunting for the handle.
    h.addEventListener('dblclick', () => {
      cols = FB_COLS_DEF.slice();
      applyCols();
      saveCols();
    });
  }

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
  // The rendering itself is the shared component (fs-preview.js), which
  // the open/save dialog's side panel uses too.
  const preview = createFsPreview();
  prevBody.append(preview.el);
  const renderPreviewEmpty = () => preview.empty();
  const showPreview = (e) => preview.show(curDir, e);

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
  // when given (a tree-row click), is highlighted directly. `focusName`
  // is an entry to land the selection + keyboard cursor on afterwards.
  async function enterDir(vpath, node, focusName) {
    const ok = await loadListing(vpath);
    if (!ok) { return; }
    if (focusName) {
      list.setSelection([focusName]);
      selectedName = focusName;
      selectedIsDir = true;      // only ever the directory we came from
      updateToolbar();
    }
    if (node) { setCurrentNode(node); }
    else { revealPath(curDir); }
  }

  // Out to the parent, landing on the directory just left so Left and
  // Right are each other's undo. Shared by the toolbar button and the
  // Left arrow key.
  function goUp() {
    if (!curParent) { return; }
    enterDir(curParent, null, baseName(curDir));
  }

  // ---- toolbar actions --------------------------------------------
  upBtn.addEventListener('click', goUp);

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
      // Give the listing the keyboard straight away: the view has no
      // other default focus, and arrow navigation that needs a click to
      // wake up is a feature nobody finds.
      list.focus();
    }
  })();
}
