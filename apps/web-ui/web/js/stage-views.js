// Host side of the stage-provided GUI views.
//
// A stage can ship its own panel: the JavaScript, CSS and message
// catalogue are embedded in libvpipe next to the stage's C++, and the
// backend advertises them at GET /api/ui/views (see
// ui/ui-view-registry.h). This app's entire job is to
//   * ask what exists,
//   * dynamically import() the module the backend names,
//   * hand it the widget helpers and a message channel, and
//   * forward FlexData between it and its C++ backend
// -- never to know what any particular view does. Adding a panel to a
// stage therefore touches no file in apps/web-ui.
//
// The panel registries in composer.js / io-workspace.js keep their
// built-in app-level views (pipelines, database, files, ...) and APPEND
// whatever this module discovers.

import { api, getAuthKey } from './api.js';
import { el, clear } from './dom.js';
import { makeIcon } from './icons.js';
import { t, addStrings } from './i18n.js';
import { openViewChannel } from '/ui/sdk/view-channel.js';

// Discovery is a one-shot per page load: the set of registered views is
// fixed for the lifetime of the backend process, and app.js already
// reloads the page when the server restarts.
let discovery = null;

// Views the backend registered, as [{id, stage_type, module, styles,
// label_key, icon}]. Never rejects -- an older backend without the
// route, or an unreachable one, simply contributes no views.
export function listStageViews() {
  if (!discovery) {
    discovery = api.uiViews()
      .then((r) => (r && Array.isArray(r.views)) ? r.views : [])
      .catch(() => []);
  }
  return discovery;
}

// Modules + stylesheets already brought in, keyed by URL.
const modules = new Map();
const sheets = new Set();

function injectStyles(href) {
  if (!href || sheets.has(href)) { return; }
  sheets.add(href);
  document.head.append(
    el('link', { rel: 'stylesheet', href }));
}

// Import a view's module once, register its message catalogue, and
// attach its stylesheet. Subsequent mounts reuse the same module.
function loadModule(view) {
  if (!modules.has(view.module)) {
    modules.set(view.module, import(view.module).then((mod) => {
      // A view ships its own strings; merging them into the host
      // catalogue is what lets t() resolve the view's keys (and the
      // panel label the backend named in label_key).
      if (mod && mod.strings) { addStrings(mod.strings); }
      injectStyles(view.styles);
      return mod;
    }));
  }
  return modules.get(view.module);
}

// What a view module receives as its third mount() argument. Everything
// it can reach is here -- it imports nothing from this app.
function viewContext(view, cfg, ctx) {
  const c = ctx || {};
  return {
    // The app's widget helpers, passed rather than imported so a view
    // never depends on this app's module layout.
    host: { el, clear, makeIcon, t },
    onTitle: c.onTitle || (() => {}),
    // Opaque per-panel blob the host persists with the layout.
    config: cfg || {},
    onConfigChange: c.onConfigChange || (() => {}),
    // A channel to this view's C++ backend, pre-bound to its id.
    openChannel: (handlers) =>
      openViewChannel(view.id, handlers, getAuthKey()),
  };
}

// Adapt a registry entry to the panel-registry mount signature
// (body, actions, config, ctx) -> cleanup.
//
// import() is async while a panel mount is synchronous, so the returned
// cleanup closes over the in-flight load: unmounting before the module
// arrives cancels it rather than mounting into a detached body.
export function stageViewMount(view) {
  return (body, actions, cfg, ctx) => {
    let cleanup = null;
    let dead = false;
    loadModule(view).then((mod) => {
      if (dead) { return; }
      if (!mod || typeof mod.mount !== 'function') {
        throw new Error('view module exports no mount()');
      }
      cleanup = mod.mount(body, actions, viewContext(view, cfg, ctx)) || null;
    }).catch((e) => {
      if (dead) { return; }
      clear(body).append(el('div', { class: 'preview-hint' },
        t('views.load_failed', { id: view.id, msg: e.message })));
    });
    return () => {
      dead = true;
      if (cleanup) {
        try { cleanup(); } catch (e) {}
        cleanup = null;
      }
    };
  };
}

// Registry entries in the shape composer.js / io-workspace.js consume.
// `labelKey` resolves through the same t() as a built-in view because
// the module's own strings were merged at load time.
export async function stageViewPanelTypes() {
  const views = await listStageViews();
  return views.map((v) => ({
    type: v.id,
    labelKey: v.label_key || ('stage.' + v.id + '.name'),
    icon: v.icon || 'io',
    stageType: v.stage_type || '',
    mount: stageViewMount(v),
  }));
}
