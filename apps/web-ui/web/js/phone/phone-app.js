// Phone shell: the reduced UI served to a phone browser.
//
// It reuses the desktop page skeleton (index.html) rather than a second
// document -- the topbar, #view host, modal root and toast root are all
// already there -- and changes what fills it: the left navigation strip
// and the system status bar are dropped, and the nav moves into a drawer
// behind a top-right menu button. That keeps ONE page, one asset bundle,
// and one place where a view's mount contract is defined.
//
// A reduced set of views: the pipeline browser, text I/O, the two panels
// the stages ship (Preview and Compare Images), a file browser and the
// live system monitors. What the dashboard has and this does not --
// profiler, database, composer, settings, and the graph CANVAS -- stays
// desktop-only, reached by switching layouts from the drawer, which is
// also the escape hatch when the phone/desktop detection guesses wrong.
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
import { t, onLocaleChange, getLocale, setLocale, locales }
  from '../i18n.js';
import { api } from '../api.js';
import { startProgressPoll } from '../status-bar.js';
import { setUiMode } from '../ui-mode.js';
import { mountPhonePipelines } from './phone-pipelines.js';
import { mountPhoneIo } from './phone-io.js';
import { stageViewMounter } from './phone-stage-view.js';
import { mountPhoneFiles } from './phone-files.js';
import { mountPhoneSystem } from './phone-system.js';
import { mountViewportDebug } from './phone-debug.js';

const VIEWS = [
  { id: 'pipelines', labelKey: 'nav.pipelines', icon: 'pipeline',
    mount: mountPhonePipelines },
  { id: 'io', labelKey: 'nav.io', icon: 'io',
    mount: mountPhoneIo, cache: true },
  // Panels the STAGES ship with themselves; the phone side of each is
  // just the shared host pointed at its registered view id.
  { id: 'preview', labelKey: 'phone.preview', icon: 'video',
    mount: stageViewMounter('preview', 'phone.preview') },
  { id: 'compare', labelKey: 'phone.compare', icon: 'image',
    mount: stageViewMounter('compare-image', 'phone.compare') },
  // Browsing the server's filesystem is already here for load and save;
  // this is the same walker with a preview on the end.
  { id: 'files', labelKey: 'nav.files', icon: 'files',
    mount: mountPhoneFiles },
  // The phone shell drops the desktop's status bar, so this is the only
  // place its live monitors appear here.
  { id: 'system', labelKey: 'phone.system', icon: 'profiler',
    mount: mountPhoneSystem },
];

// Publish the VISIBLE height of the window as --vvh, and flag whether a
// soft keyboard is up.
//
// A phone keyboard does not shrink the layout viewport. iOS leaves
// `100dvh` at its full height and simply COVERS the bottom of it, so a
// column laid out to fill the screen puts its input row underneath the
// keyboard, out of reach and out of sight -- and no amount of flexing
// helps, because as far as the layout is concerned nothing changed.
//
// The visual viewport is the part still on screen. Sizing the shell to
// that instead is the whole fix: every phone view is already a flex
// column whose middle section takes the slack, so the header keeps its
// size, the scrolling region gives up the difference, and whatever sits
// at the bottom (the text I/O compose box, a stage's Apply button)
// stays above the keyboard. Nothing per-view is needed.
function trackViewport(app) {
  const vv = window.visualViewport;
  if (!vv) { return; }        // no API -> --vvh unset, rules fall back
  const root = document.documentElement;
  // Is the caret in a text field? This is the condition the soft
  // keyboard actually follows, which makes it the primary signal: no
  // measurement survives a ROTATION performed with the keyboard up. A
  // rotation invalidates the height baseline, the only height left to
  // learn from is the keyboard-shrunken one, and from then until the
  // field is blurred the keyboard reads as absent.
  const typing = () => {
    const e = document.activeElement;
    if (!e) { return false; }
    if (e.isContentEditable) { return true; }
    const tag = e.tagName;
    if (tag === 'TEXTAREA') { return true; }
    if (tag !== 'INPUT') { return false; }
    // Buttons, checkboxes and friends take focus without a keyboard.
    return !/^(button|checkbox|radio|range|color|file|submit|reset|image)$/i
        .test(e.type || 'text');
  };

  // Geometry stays as a SECOND opinion, for a keyboard raised without a
  // focused field (a browser's own find bar, an accessibility panel).
  // Its baseline is the tallest the visible area has been -- comparing
  // against window.innerHeight does not work, because with
  // interactive-widget=resizes-content the layout viewport shrinks by
  // the same amount as the visual one and the difference is zero
  // (measured: inner=358, vv h=358, keyboard plainly up).
  let tallest = 0;
  let baseWidth = 0;
  const apply = () => {
    // A rotation changes what "tallest" means, so the baseline restarts
    // with the width.
    if (window.innerWidth !== baseWidth) {
      baseWidth = window.innerWidth;
      tallest = 0;
    }
    const inField = typing();
    // Learn the baseline ONLY when no field is focused. Otherwise a
    // rotation with the keyboard up teaches it the shrunken height and
    // it never recovers -- which is precisely the bug above.
    if (!inField) { tallest = Math.max(tallest, vv.height); }
    root.style.setProperty('--vvh', Math.round(vv.height) + 'px');
    // A URL bar sliding away also shrinks the visible area, by far less
    // than a keyboard does; ~120px tells the two apart without having to
    // know a device's keyboard height.
    app.classList.toggle('kbd-open',
        inField || (tallest > 0 && (tallest - vv.height) > 120));
  };
  // Focus can change with no viewport event at all (an external
  // keyboard, or dismissing the soft one while the field keeps focus),
  // so the flag is re-evaluated on focus moves too.
  document.addEventListener('focusin', apply, true);
  document.addEventListener('focusout', () => setTimeout(apply, 0), true);
  // The scroll iOS applies at focus time is STALE by the time the
  // layout has caught up, and nothing re-evaluates it.
  //
  // At focus the page is still full height, the field really is behind
  // the keyboard, and the browser scrolls to lift it -- correct, then.
  // A frame later --vvh shrinks everything to the visible height and
  // the field is in view with no scroll needed, but the scroll already
  // applied stays put, holding the UI above the screen for as long as
  // the keyboard is up. Measured: scrollY 337 -- exactly the keyboard's
  // height -- against content only 358 tall.
  //
  // So it is reset ONCE PER GEOMETRY CHANGE, and never in response to a
  // scroll. That distinction is the whole safety argument: an earlier
  // version compensated continuously and the UI shook, because the
  // browser was re-deciding against a page that kept moving under it.
  // Nothing here reacts to the browser reacting to us -- a keyboard
  // opening fires a handful of resizes, each worth at most two
  // corrections, and then it stops.
  const unscroll = () => {
    if (window.scrollY !== 0) { window.scrollTo(0, 0); }
  };
  vv.addEventListener('resize', () => {
    apply();
    unscroll();
    // ...and once more after the keyboard's animation settles, which
    // finishes without an event of its own.
    setTimeout(unscroll, 250);
  });
  // Scroll only re-reads the numbers; it must not correct anything.
  vv.addEventListener('scroll', apply);
  apply();
}

// Refuse the gestures that would scroll the PAGE while a keyboard is up.
//
// The layout viewport stays at its full height (695 on the device this
// was measured on) while only the visible part shrinks (358), so the
// browser will happily scroll the page down into the 337px of nothing
// below the UI. No stylesheet prevents it: the scrolling box is the
// layout viewport itself, which no height set here resizes, and
// `overflow: hidden` on the root propagates to that viewport in theory
// while iOS scrolls it anyway in practice.
//
// So the gesture is refused instead -- and ONLY the gestures that would
// move the page. A touch that begins inside something which can really
// scroll (the console, a list, a long config form) is left completely
// alone, found by walking up for a scroll container with room left to
// move rather than by tagging every scroller in the app by hand. Two
// fingers are left alone too, or this would take pinch-zoom with it.
//
// Armed only while the keyboard is up, because that is the only time
// the page HAS anywhere to scroll -- with the keyboard down the visible
// area and the layout viewport are the same size.
function lockPageScroll(app) {
  const canScroll = (node) => {
    for (let e = node; e && e !== document.body; e = e.parentElement) {
      if (!(e instanceof Element)) { continue; }
      const oy = getComputedStyle(e).overflowY;
      if ((oy === 'auto' || oy === 'scroll')
          && e.scrollHeight > e.clientHeight) {
        return true;
      }
    }
    return false;
  };
  // passive: false -- the whole point is to be able to refuse it.
  document.addEventListener('touchmove', (e) => {
    if (!app.classList.contains('kbd-open')) { return; }
    if (e.touches && e.touches.length > 1) { return; }   // pinch-zoom
    if (canScroll(e.target)) { return; }                 // a real scroller
    e.preventDefault();
  }, { passive: false });
}

// Two viewport keys the phone shell asks for, both of which have to be
// set on the meta tag rather than in CSS.
//
// interactive-widget=resizes-content -- "when a soft keyboard opens,
// shrink the LAYOUT viewport instead of covering it". That is exactly
// the behaviour this whole file is otherwise emulating by hand with
// --vvh: where it is honoured, the page height simply becomes correct
// and stays correct, with nothing to track. Chrome/Android implements
// it; browsers that do not, ignore the key, and the --vvh path carries
// them. Unconditional, because it asks for the right thing everywhere.
//
// maximum-scale=1 -- stop iOS zooming the page in when a field under
// 16px takes focus. The usual "fix" is to set every input to 16px, but
// that makes each field read as a different, louder kind of text than
// the labels around it, which is what the 13px sizing exists to avoid.
//
// maximum-scale is iOS-ONLY, and deliberately without user-scalable=no:
//
//   - iOS has ignored scale restrictions for USER-initiated zoom since
//     iOS 10, so pinch-to-zoom keeps working and this costs no
//     accessibility. user-scalable=no would add nothing here and would
//     be a real restriction if that ever changed.
//   - Android Chrome does not auto-zoom on focus, so there is nothing
//     to fix there -- but it DOES honour maximum-scale, so applying it
//     would take pinch-zoom away for no benefit at all.
function tuneViewport() {
  const meta = document.querySelector('meta[name="viewport"]');
  if (!meta) { return; }
  const add = (key, value) => {
    if (new RegExp('\\b' + key + '\\s*=').test(meta.content)) { return; }
    meta.content = meta.content + ', ' + key + '=' + value;
  };
  add('interactive-widget', 'resizes-content');
  const ua = navigator.userAgent || '';
  const ios = /iPhone|iPod|iPad/.test(ua)
      || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (ios) { add('maximum-scale', '1'); }
}

export function mountPhoneApp() {
  document.head.append(
    el('link', { rel: 'stylesheet', href: '/css/phone.css' }));
  tuneViewport();

  const app = document.getElementById('app');
  const topbar = document.getElementById('topbar');
  const view = document.getElementById('view');
  const nav = document.getElementById('nav');
  const statusbar = document.getElementById('statusbar');
  // The class drives every phone-only rule in phone.css; the nav strip
  // and status bar are hidden by it (their elements stay so the desktop
  // markup is untouched).
  app.classList.add('phone');
  // Also on the ROOT: html and body have to be styled too (see the page
  // pinning in phone.css), and they are outside #app.
  document.documentElement.classList.add('phone-ui');
  trackViewport(app);
  lockPageScroll(app);
  // ?debug=vv -- an on-screen readout of the viewport numbers. The
  // keyboard-layout failures on iOS are indistinguishable by eye, and a
  // phone has nowhere to print them otherwise.
  try {
    if (new URLSearchParams(location.search).get('debug') === 'vv') {
      mountViewportDebug();
    }
  } catch (e) { /* no URLSearchParams -- skip the panel */ }
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

  // ---- live progress line -------------------------------------------
  // A thin bar directly under the menu-button row, carrying its own
  // description. It is a sibling of the topbar rather than a child so it
  // spans the full width and leaves the bar's height (and its
  // safe-area padding) alone -- and because #app is a flex column, that
  // position IS "just below the row" in every view. The phone shell has
  // no status bar, so without this a long-running report is invisible
  // here.
  //
  // Hidden entirely while nothing is reporting: an always-present empty
  // rail would spend a whole line of a phone screen saying nothing.
  const progFill = el('div', { class: 'ph-prog-fill' });
  const progText = el('span', { class: 'ph-prog-text' });
  const progBar = el('div', { class: 'ph-prog', hidden: true },
    progFill, progText);
  if (topbar && topbar.parentNode) {
    topbar.parentNode.insertBefore(progBar, topbar.nextSibling);
  }
  // Not captured: this shell is mounted once and lives as long as the
  // page, so the poll's lifetime is the page's too (as the desktop's
  // restart watcher is). There is no teardown to hang a stop on.
  startProgressPoll((items) => {
    if (!items.length) {
      progBar.hidden = true;
      progBar.classList.remove('indet');
      return;
    }
    // The most recently updated report, which is what `seq` is bumped
    // for -- the same pick the desktop's status cell makes. Concurrent
    // reports are counted rather than listed: there is one line here,
    // and a phone has no room for a panel to expand into.
    let top = items[0];
    for (const it of items) {
      if ((it.seq || 0) > (top.seq || 0)) { top = it; }
    }
    const frac = top.total
      ? Math.max(0, Math.min(1, top.done / top.total)) : null;
    progBar.hidden = false;
    // total 0 means the reporter cannot say how much there is; a bar
    // pinned at 0% would read as "stuck", so it sweeps instead.
    progBar.classList.toggle('indet', frac === null);
    progFill.style.width = frac === null ? '' : (frac * 100).toFixed(1) + '%';
    const pct = frac === null ? '' : Math.round(frac * 100) + '% ';
    const more = items.length > 1 ? ' +' + (items.length - 1) : '';
    progText.textContent =
      pct + (top.detail || top.desc || '') + more;
  });

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

  function currentLanguageLabel() {
    const cur = getLocale();
    const l = locales().find((x) => x.tag === cur);
    // The self-name ("简体中文"), not a translation of it: someone
    // looking for their own language recognises it written its own way.
    return l ? l.label : cur;
  }

  // Same two steps as the desktop's picker: set the client locale (whose
  // listener re-mounts every pane below), then tell the BACKEND, so
  // server-produced messages match what the client is now showing. An
  // older backend without the route just leaves those in English.
  function openLanguageSheet() {
    closeDrawer();                 // its scrim sits above the modal layer
    const list = el('div', { class: 'ph-sheet-list' });
    const close = openModal({
      title: t('settings.language'),
      body: list,
      actions: [{ label: t('common.cancel'), cancel: true,
                  onClick: (c) => c() }],
    });
    const cur = getLocale();
    for (const l of locales()) {
      list.append(el('button', {
        class: 'ph-sheet-item' + (l.tag === cur ? ' active' : ''),
        onclick: () => {
          close();
          if (l.tag === cur) { return; }
          setLocale(l.tag);
          api.setLanguage(l.tag).catch(() => { /* older backend */ });
        },
      }, el('span', { class: 'ph-sheet-name' }, l.label)));
    }
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
    // Language. The phone shell has no Settings view to put this in, and
    // it is the one preference that is useless where it currently lives:
    // a reader who needs another language cannot be asked to switch to
    // the desktop layout to find the switch. Shown as a row rather than
    // the desktop's segmented control -- three self-named languages do
    // not fit side by side at a tappable size -- with the current one as
    // its value, so the drawer says what is set without being opened
    // twice.
    panel.append(el('button', {
      class: 'ph-drawer-item sub', onclick: () => openLanguageSheet(),
    },
      el('span', {}, t('settings.language')),
      el('span', { class: 'ph-drawer-value' }, currentLanguageLabel())));

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
