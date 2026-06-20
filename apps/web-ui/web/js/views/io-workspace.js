// User I/O workspace: a splittable pane tree. It opens with a single
// text I/O pane and lets the user split any pane vertically (left /
// right) or horizontally (top / bottom) via a low-profile "⋯" menu in
// each pane's header. A freshly split-off pane starts empty and can be
// assigned a view type; the supported types are HLS video and the
// Session Log.
//
// There is structurally only ever ONE text I/O pane (the anchor):
// splits always create EMPTY panes, and the empty-pane chooser offers
// only HLS / Session Log, so the "one text I/O view session-wide" rule
// holds. HLS and log panes may be created freely, so there can be many.
//
// Splits and closes are done as in-place DOM surgery (wrap / unwrap),
// never a full re-render, so an existing pane's live content (the
// streaming console, a playing video) keeps running across the change.

import { el, clear } from '../dom.js';
import { mountUserIo } from './user-io.js';
import { mountHlsVideo } from './hls-video.js';
import { mountLog } from './log.js';
import { t } from '../i18n.js';

// The composed workspace persists for the life of the page (until the
// connection is lost -> a reload re-imports this module and resets it).
// We cache the live DOM so the split layout, the streaming console, and
// any playing video all survive switching to another nav view and back
// -- the alternative (rebuilding on every mount) is what reset the view
// to a single default pane.
let wsSingleton = null;

export function mountIoWorkspace(container) {
  if (wsSingleton) {
    // Re-attach the existing live workspace; do NOT rebuild it (that
    // would drop the layout + restart the console / video).
    clear(container);
    container.append(wsSingleton);
    return;
  }

  clear(container);
  const ws = el('div', { class: 'io-workspace' });
  container.append(ws);
  wsSingleton = ws;

  let root = null;          // root tree node (leaf or split)
  let menu = null;          // { el, onDoc, onKey } open dropdown, if any

  // ---- dropdown menu ----------------------------------------------
  function closeMenu() {
    if (!menu) { return; }
    menu.el.remove();
    document.removeEventListener('mousedown', menu.onDoc);
    document.removeEventListener('keydown', menu.onKey);
    menu = null;
  }

  function openMenu(anchor, items) {
    closeMenu();
    const m = el('div', { class: 'io-menu' });
    for (const it of items) {
      m.append(el('button', {
        class: 'io-menu-item' + (it.danger ? ' danger' : ''),
        onclick: () => { closeMenu(); it.onClick(); },
      }, it.label));
    }
    ws.append(m);
    // Position under the anchor, right-aligned to it, in ws-local
    // coordinates (the menu is a child of ws so it's clipped by
    // nothing and removed automatically when the view unmounts).
    const wr = ws.getBoundingClientRect();
    const ar = anchor.getBoundingClientRect();
    let left = ar.right - wr.left - m.offsetWidth;
    if (left < 4) { left = 4; }
    m.style.left = left + 'px';
    m.style.top = (ar.bottom - wr.top + 4) + 'px';
    const onDoc = (e) => {
      if (!m.contains(e.target) && e.target !== anchor) { closeMenu(); }
    };
    const onKey = (e) => { if (e.key === 'Escape') { closeMenu(); } };
    menu = { el: m, onDoc, onKey };
    // Defer so the click that opened the menu doesn't immediately
    // close it.
    setTimeout(() => {
      if (!menu) { return; }
      document.addEventListener('mousedown', onDoc);
      document.addEventListener('keydown', onKey);
    }, 0);
  }

  // ---- leaf (a single pane) ---------------------------------------
  function makeLeaf(view, opts) {
    const titleEl = el('span', { class: 'title' });
    const actions = el('span', { class: 'io-pane-actions' });
    const menuBtn = el('button', { class: 'io-pane-menu',
      title: t('io.split_options') }, '⋯');
    const headEl = el('div', { class: 'io-pane-head' },
      titleEl, el('span', { class: 'grow' }), actions, menuBtn);
    const bodyEl = el('div', { class: 'io-pane-body' });
    const paneEl = el('div', { class: 'io-pane' }, headEl, bodyEl);

    const leaf = { kind: 'leaf', view: null, el: paneEl, titleEl,
                   actions, bodyEl, parent: null, cleanup: null };
    menuBtn.addEventListener('click', () => paneMenu(leaf, menuBtn));
    mountView(leaf, view, opts);
    return leaf;
  }

  function setTitle(leaf, t) { leaf.titleEl.textContent = t; }

  function mountView(leaf, view, opts) {
    if (leaf.cleanup) {
      try { leaf.cleanup(); } catch (e) {}
      leaf.cleanup = null;
    }
    clear(leaf.bodyEl);
    clear(leaf.actions);
    leaf.view = view;
    if (view === 'text') {
      setTitle(leaf, t('nav.io'));
      leaf.cleanup = mountUserIo(leaf.bodyEl, leaf.actions);
    } else if (view === 'hls') {
      setTitle(leaf, t('io.hls'));
      leaf.cleanup = mountHlsVideo(leaf.bodyEl, leaf.actions, {
        stream: opts && opts.stream,
        onTitle: (t) => setTitle(leaf, t),
      });
    } else if (view === 'log') {
      setTitle(leaf, t('io.session_log'));
      leaf.cleanup = mountLog(leaf.bodyEl, leaf.actions);
    } else {
      setTitle(leaf, t('io.new_view'));
      mountEmpty(leaf);
    }
  }

  function mountEmpty(leaf) {
    // Text I/O is a singleton and already present. HLS and Session Log
    // may be added freely.
    leaf.bodyEl.append(el('div', { class: 'io-empty' },
      el('div', { class: 'io-empty-title' }, t('io.add_view')),
      el('div', { class: 'io-empty-choices' },
        el('button', { class: 'btn primary',
          onclick: () => mountView(leaf, 'hls') }, t('io.hls')),
        el('button', { class: 'btn',
          onclick: () => mountView(leaf, 'log') }, t('io.session_log'))),
      el('div', { class: 'io-empty-hint' },
        t('io.more_soon'))));
  }

  function paneMenu(leaf, anchor) {
    const items = [
      { label: t('io.split_v'),
        onClick: () => splitLeaf(leaf, 'v') },
      { label: t('io.split_h'),
        onClick: () => splitLeaf(leaf, 'h') },
    ];
    // The anchor text I/O pane is the session-wide singleton and is
    // never closeable; every other pane is.
    if (leaf.view !== 'text') {
      items.push({ label: t('io.close_pane'), danger: true,
        onClick: () => closeLeaf(leaf) });
    }
    openMenu(anchor, items);
  }

  // ---- split / close (in-place DOM surgery) -----------------------
  // dir 'v' -> vertical divider, panes side by side (existing left,
  // new right). dir 'h' -> horizontal divider, panes stacked (existing
  // top, new bottom).
  function splitLeaf(leaf, dir) {
    const parent = leaf.parent;
    const empty = makeLeaf('empty');
    const splitEl = el('div', { class: 'io-split dir-' + dir });
    const host = leaf.el.parentNode;     // ws or a parent split's el
    host.replaceChild(splitEl, leaf.el);
    splitEl.append(leaf.el, empty.el);   // existing first (left / top)

    const split = { kind: 'split', dir, el: splitEl,
                    a: leaf, b: empty, parent };
    leaf.parent = split;
    empty.parent = split;
    if (parent) {
      if (parent.a === leaf) { parent.a = split; } else { parent.b = split; }
    } else {
      root = split;
    }
  }

  function closeLeaf(leaf) {
    const parent = leaf.parent;
    if (!parent) { return; }             // the only pane stays
    const sibling = (parent.a === leaf) ? parent.b : parent.a;
    const grand = parent.parent;
    const host = parent.el.parentNode;
    host.replaceChild(sibling.el, parent.el);
    sibling.parent = grand;
    if (grand) {
      if (grand.a === parent) { grand.a = sibling; }
      else { grand.b = sibling; }
    } else {
      root = sibling;
    }
    if (leaf.cleanup) {
      try { leaf.cleanup(); } catch (e) {}
    }
  }

  // ---- boot --------------------------------------------------------
  root = makeLeaf('text');
  ws.append(root.el);
}
