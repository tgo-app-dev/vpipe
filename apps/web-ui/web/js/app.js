// vpipe web-ui bootstrap: builds the navigation strip and swaps views.

import { el, clear, openModal } from './dom.js';
import { makeIcon } from './icons.js';
import { api, setAuthPrompt } from './api.js';
import { getTheme, applyTheme } from './theme.js';
import { t, getLocale, onLocaleChange } from './i18n.js';
import { mountPipelineManager } from './views/pipeline-manager.js';
import { mountIoWorkspace } from './views/io-workspace.js';
import { mountDatabase } from './views/database.js';
import { mountProfiler } from './views/profiler.js';
import { mountSettings } from './views/settings.js';
import { mountStatusBar } from './status-bar.js';

// Apply the persisted theme as early as possible (index.html's inline
// script already set it on first paint; this keeps it consistent).
applyTheme(getTheme());

// Reflect the persisted UI locale on <html lang>, and best-effort push
// it to the backend so server-produced messages match this client.
document.documentElement.setAttribute('lang', getLocale());
api.setLanguage(getLocale()).catch(() => { /* older backend -- ignore */ });

// Remote servers gate /api/* on an access key printed in their console.
// When the backend returns 401, api.js calls this to collect the key.
setAuthPrompt(() => new Promise((resolve) => {
  const input = el('input', { type: 'text', maxlength: '8',
    placeholder: t('auth.key_ph') });
  openModal({
    title: t('auth.title'),
    body: el('div', {},
      el('p', {}, t('auth.body')),
      el('label', { class: 'fl' }, t('auth.key')), input),
    actions: [
      { label: t('common.cancel'), cancel: true,
        onClick: (c) => { c(); resolve(null); } },
      { label: t('auth.connect'), kind: 'primary',
        onClick: (c) => { const v = input.value.trim(); c(); resolve(v || null); } },
    ],
  });
  setTimeout(() => input.focus(), 0);
}));

// Auto-refresh a localhost page when the server is restarted. We poll
// /api/health for its per-process `instance` token and hard-reload when
// it changes -- a fresh server means the open page may hold stale data
// (pipeline list, configs, ...). Gated to localhost: a remote server
// re-keys on restart, so its clients hit a 401 and refresh via the
// post-authentication reload in api.js instead.
function isLocalHost() {
  const h = location.hostname;
  return h === 'localhost' || h === '127.0.0.1'
    || h === '::1' || h === '[::1]';
}

function watchServerInstance() {
  if (!isLocalHost()) { return; }
  const POLL_MS = 2000;
  let seen = null;
  async function tick() {
    try {
      const h = await api.health();
      const inst = h && h.instance;
      if (inst) {
        if (seen === null) { seen = inst; }
        else if (inst !== seen) { location.reload(); return; }
      }
    } catch (e) { /* server down / transient -- keep polling */ }
    setTimeout(tick, POLL_MS);
  }
  setTimeout(tick, POLL_MS);
}

// On connect, mirror the server's startup permission report (the one
// printed to its console) in a dialog. The blocking probes can still be
// running for a moment after the page loads, so retry while not ready.
// Dismiss with the button, Enter, or Escape (handled by openModal).
async function showStartupChecks() {
  let doc = null;
  for (let i = 0; i < 8; i++) {
    try { doc = await api.startupChecks(); }
    catch (e) { return; }   // server gone / unauthorized -- skip silently
    if (doc && doc.ready) { break; }
    doc = null;
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (!doc || !Array.isArray(doc.checks) || doc.checks.length === 0) {
    return;
  }
  const list = el('div', { class: 'startup-checks' });
  for (const c of doc.checks) {
    const warn = c.status === 'warn';
    list.append(el('div', { class: 'sc-row ' + (warn ? 'warn' : 'ok') },
      el('span', { class: 'sc-badge' },
        warn ? t('startup.warn') : t('startup.ok')),
      el('span', { class: 'sc-name' }, c.name || ''),
      el('span', { class: 'sc-detail' }, c.detail || '')));
    if (warn && Array.isArray(c.hints) && c.hints.length) {
      const h = el('div', { class: 'sc-hints' });
      for (const t of c.hints) { h.append(el('div', {}, t)); }
      list.append(h);
    }
  }
  const intro = el('p', { class: 'sc-intro' }, doc.has_warnings
    ? t('startup.needs_attention')
    : t('startup.all_passed'));
  openModal({
    title: t('startup.title'),
    body: el('div', {}, intro, list),
    actions: [{ label: t('common.dismiss'), kind: 'primary',
                onClick: (c) => c() }],
  });
}

function placeholder(title) {
  return (container) => {
    clear(container).append(el('div', { class: 'placeholder' },
      el('h2', {}, title),
      el('p', {}, t('app.coming_soon'))));
  };
}

// `labelKey` is an i18n catalogue key resolved through t() at render
// time, so the nav re-labels itself when the UI language changes.
const VIEWS = [
  { id: 'pipelines', labelKey: 'nav.pipelines', icon: 'pipeline',
    mount: mountPipelineManager },
  { id: 'profiler', labelKey: 'nav.profiler', icon: 'profiler',
    mount: mountProfiler },
  { id: 'io', labelKey: 'nav.io', icon: 'io',
    mount: mountIoWorkspace },
  { id: 'db', labelKey: 'nav.database', icon: 'database',
    mount: mountDatabase },
];
const SETTINGS = { id: 'settings', labelKey: 'nav.settings', icon: 'settings',
  mount: mountSettings };

function main() {
  // Suppress the browser's right-click menu across the whole UI: it
  // pops over panes (notably the live HLS video) and interrupts
  // dashboard activity. Elements that opt in via `.allow-context-menu`
  // (text displays / inputs where copy / paste / select-all / spellcheck
  // are useful) keep the native menu.
  document.addEventListener('contextmenu', (e) => {
    if (e.target.closest && e.target.closest('.allow-context-menu')) {
      return;
    }
    e.preventDefault();
  });

  const nav = document.getElementById('nav');
  const view = document.getElementById('view');
  let current = null;

  function select(v) {
    current = v;
    renderNav();
    v.mount(view);
    history.replaceState(null, '', '#' + v.id);
  }

  function navButton(v) {
    const active = current && current.id === v.id;
    return el('button', {
      class: 'nav-btn' + (active ? ' active' : ''),
      onclick: () => select(v),
    }, makeIcon(v.icon), el('span', { class: 'label' }, t(v.labelKey)));
  }

  // Rebuild the nav strip (labels come from t(), so this re-localizes).
  function renderNav() {
    clear(nav);
    for (const v of VIEWS) { nav.append(navButton(v)); }
    nav.append(el('div', { class: 'spacer' }));
    nav.append(navButton(SETTINGS));
  }

  const all = [...VIEWS, SETTINGS];
  const initial = all.find((v) => '#' + v.id === location.hash) || VIEWS[0];
  select(initial);

  // Re-render the nav and re-mount the current view when the UI language
  // changes (each view reads t() at mount time).
  onLocaleChange(() => { if (current) { select(current); } });

  // Bottom system status bar -- polls /api/system/status every 1 s
  // and survives the view-switching lifecycle (no mount/unmount, it
  // lives as long as the page).
  const statusbar = document.getElementById('statusbar');
  if (statusbar) { mountStatusBar(statusbar); }

  // Localhost: refresh automatically when the backend is restarted.
  watchServerInstance();

  // Mirror the server's startup permission report in a dialog on connect.
  showStartupChecks();
}

main();
