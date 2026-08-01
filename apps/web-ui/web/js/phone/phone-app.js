// Phone shell: the reduced UI served to a phone browser.
//
// It reuses the desktop page skeleton (index.html) rather than a second
// document -- the topbar, #view host, modal root and toast root are all
// already there -- and changes what fills it: the left navigation strip
// and the system status bar are dropped, and the nav moves into a drawer
// behind a top-right menu button. That keeps ONE page, one asset bundle,
// and one place where a view's mount contract is defined.
//
// Three views only, for now: the read-only pipeline browser, text I/O,
// and Preview. Everything else the dashboard offers (profiler, database,
// files, composer, settings) stays desktop-only and is reached by
// switching layouts from the drawer -- which is also the escape hatch
// when the phone/desktop detection guesses wrong.
//
// A view module gets { body, actions, setTitle } and returns a cleanup
// function. `cache: true` on a view keeps its mounted DOM alive across
// view switches instead of tearing it down -- that is what lets the text
// I/O console keep its transcript (and its poll) while the user looks at
// a pipeline. Views that hold an expensive live resource (Preview's
// video channel) deliberately do NOT cache: they persist their choice
// instead and resume it on the way back, so nothing streams to a phone
// that isn't looking at it.

import { el, clear, openModal } from '../dom.js';
import { makeIcon } from '../icons.js';
import { t, onLocaleChange } from '../i18n.js';
import { setUiMode } from '../ui-mode.js';
import { mountPhonePipelines } from './phone-pipelines.js';
import { mountPhoneIo } from './phone-io.js';
import { mountPhonePreview } from './phone-preview.js';
import { mountPhoneFiles } from './phone-files.js';
import { mountPhoneSystem } from './phone-system.js';

const VIEWS = [
  { id: 'pipelines', labelKey: 'nav.pipelines', icon: 'pipeline',
    mount: mountPhonePipelines },
  { id: 'io', labelKey: 'nav.io', icon: 'io',
    mount: mountPhoneIo, cache: true },
  { id: 'preview', labelKey: 'phone.preview', icon: 'video',
    mount: mountPhonePreview },
  // Browsing the server's filesystem is already here for load and save;
  // this is the same walker with a preview on the end.
  { id: 'files', labelKey: 'nav.files', icon: 'files',
    mount: mountPhoneFiles },
  // The phone shell drops the desktop's status bar, so this is the only
  // place its live monitors appear here.
  { id: 'system', labelKey: 'phone.system', icon: 'profiler',
    mount: mountPhoneSystem },
];

export function mountPhoneApp() {
  document.head.append(
    el('link', { rel: 'stylesheet', href: '/css/phone.css' }));

  const app = document.getElementById('app');
  const topbar = document.getElementById('topbar');
  const view = document.getElementById('view');
  const nav = document.getElementById('nav');
  const statusbar = document.getElementById('statusbar');
  // The class drives every phone-only rule in phone.css; the nav strip
  // and status bar are hidden by it (their elements stay so the desktop
  // markup is untouched).
  app.classList.add('phone');
  if (nav) { clear(nav); }
  if (statusbar) { clear(statusbar); }

  // ---- top bar ------------------------------------------------------
  const menuBtn = el('button', {
    class: 'ph-menu-btn', 'aria-label': t('phone.menu'),
    title: t('phone.menu'), onclick: () => openDrawer(),
  }, makeIcon('menu', 'sm'));
  clear(topbar).append(
    el('span', { class: 'brand' }, 'VPIPE'),
    el('span', { class: 'grow' }),
    menuBtn);

  // ---- view host ----------------------------------------------------
  // One pane per view: a header (title + the view's own control strip)
  // over the body it fills. Cached views keep their pane here between
  // visits, so re-selecting one re-attaches live DOM instead of
  // rebuilding it.
  const panes = new Map();
  let current = null;

  function makePane(v) {
    const titleEl = el('span', { class: 'ph-view-title' }, t(v.labelKey));
    const actions = el('div', { class: 'ph-view-actions' });
    const body = el('div', { class: 'ph-view-body' });
    const root = el('div', { class: 'ph-view' },
      el('div', { class: 'ph-view-head' }, titleEl, actions),
      body);
    const setTitle = (s) => {
      titleEl.textContent = s || t(v.labelKey);
    };
    const cleanup = v.mount({ body, actions, setTitle }) || null;
    return { view: v, root, cleanup };
  }

  function dropPane(pane) {
    if (!pane) { return; }
    pane.root.remove();
    if (pane.view.cache) { return; }   // kept for the next visit
    if (pane.cleanup) {
      try { pane.cleanup(); } catch (e) { /* a bad view can't block */ }
    }
    panes.delete(pane.view.id);
  }

  function select(v) {
    if (current && current.view.id === v.id) { closeDrawer(); return; }
    dropPane(current);
    let pane = panes.get(v.id);
    if (!pane) {
      pane = makePane(v);
      panes.set(v.id, pane);
    }
    current = pane;
    clear(view).append(pane.root);
    history.replaceState(null, '', '#' + v.id);
    closeDrawer();
  }

  // ---- drawer -------------------------------------------------------
  // A right-hand sheet over a scrim. Anchored top-right because that is
  // where the button that opens it lives, so it unfolds from under the
  // thumb that just pressed it.
  let drawer = null;

  function closeDrawer() {
    if (!drawer) { return; }
    drawer.back.remove();
    document.removeEventListener('keydown', drawer.onKey);
    drawer = null;
  }

  // Close the drawer BEFORE the dialog: the drawer's scrim sits above
  // the modal layer, so a dialog opened underneath it would be visible
  // but untappable.
  function confirmDesktop() {
    closeDrawer();
    openModal({
      title: t('phone.desktop_site'),
      body: el('div', {},
        el('p', {}, t('phone.desktop_hint')),
        el('p', {}, t('phone.desktop_confirm'))),
      actions: [
        { label: t('common.cancel'), cancel: true, onClick: (c) => c() },
        { label: t('phone.desktop_switch'), kind: 'primary',
          onClick: (c) => { c(); setUiMode('desktop'); location.reload(); } },
      ],
    });
  }

  function openDrawer() {
    if (drawer) { closeDrawer(); return; }
    const panel = el('div', { class: 'ph-drawer' });
    panel.append(el('div', { class: 'ph-drawer-title' }, t('phone.views')));
    for (const v of VIEWS) {
      const active = current && current.view.id === v.id;
      panel.append(el('button', {
        class: 'ph-drawer-item' + (active ? ' active' : ''),
        onclick: () => select(v),
      }, makeIcon(v.icon, 'sm'), el('span', {}, t(v.labelKey))));
    }
    panel.append(el('div', { class: 'ph-drawer-sep' }));
    // The way out of a wrong detection -- and the only route to the
    // views this shell doesn't carry. Deliberately the smallest target
    // in the drawer and the only one that asks twice: it is a one-way
    // door on a phone (the desktop shell has no drawer to come back
    // through, only a top-bar button), it reloads the page, and it sits
    // a thumb's width from the view buttons above it.
    panel.append(el('button', {
      class: 'ph-drawer-link', onclick: () => confirmDesktop(),
    }, t('phone.desktop_site')));

    const back = el('div', { class: 'ph-drawer-back' }, panel);
    back.addEventListener('click', (e) => {
      if (e.target === back) { closeDrawer(); }
    });
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); closeDrawer(); }
    };
    document.addEventListener('keydown', onKey);
    drawer = { back, onKey };
    (document.getElementById('modal-root') || document.body).append(back);
  }

  // ---- boot ---------------------------------------------------------
  const initial = VIEWS.find((v) => '#' + v.id === location.hash) || VIEWS[0];
  select(initial);

  // A language change re-mounts EVERY pane, cached ones included: each
  // view reads t() at mount time, so a cached pane would otherwise keep
  // the old language until it happened to be rebuilt.
  onLocaleChange(() => {
    const keep = current ? current.view : initial;
    for (const pane of [...panes.values()]) {
      pane.root.remove();
      if (pane.cleanup) { try { pane.cleanup(); } catch (e) {} }
    }
    panes.clear();
    current = null;
    menuBtn.title = t('phone.menu');
    menuBtn.setAttribute('aria-label', t('phone.menu'));
    select(keep);
  });
}
