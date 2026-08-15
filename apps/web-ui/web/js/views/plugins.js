// Plugins view: what is available under the work directory's plugins/,
// what this process has loaded, and the two actions a session may take.
//
// THE ONE THING TO KNOW BEFORE READING THE BUTTONS: there is no unload.
// vpipe never dlclose's a plugin -- the stage registry holds raw factory
// pointers into the dylib, the composer reads StageSpec* out of its
// static storage, and live stages hold vtables there. So "Disable" stops
// the composer offering a plugin's stages; it does not reclaim it. The
// UI says that in those words rather than implying a symmetry with Load
// that the host does not have.

import { el, clear, toast } from '../dom.js';
import { t } from '../i18n.js';
import { api } from '../api.js';

export function mountPlugins(container) {
  clear(container);

  const wrap = el('div', { class: 'settings' });
  wrap.append(el('div', { class: 'pane-head' },
    el('span', { class: 'title' }, t('plugins.title'))));

  const body = el('div', { class: 'settings-body' });
  wrap.append(body);
  container.append(wrap);

  render(body);
}

async function render(body) {
  clear(body);
  let d = null;
  try {
    d = await api.plugins();
  } catch (e) {
    body.append(el('div', { class: 'tb-empty' }, String(e)));
    return;
  }

  // Where the convention says plugins live, always shown -- when the
  // list is empty this path IS the answer to "so where do I put one".
  const rootRow = el('div', { class: 'setting' });
  rootRow.append(el('div', { class: 'setting-label' }, t('plugins.root')));
  rootRow.append(el('div', { class: 'setting-desc mono' }, d.root || '?'));
  if (!d.root_exists) {
    rootRow.append(el('div', { class: 'setting-desc' },
      t('plugins.root_missing')));
  }
  body.append(rootRow);

  body.append(el('div', { class: 'setting-desc' }, t('plugins.no_unload')));

  // ---- loaded --------------------------------------------------------
  body.append(el('div', { class: 'tb-cat' },
    t('plugins.loaded') + ' (' + (d.loaded || []).length + ')'));
  if (!(d.loaded || []).length) {
    body.append(el('div', { class: 'tb-empty' }, t('plugins.none_loaded')));
  }
  for (const p of d.loaded || []) {
    body.append(loadedRow(p, body));
  }

  // ---- available but not loaded --------------------------------------
  const avail = (d.available || []).filter((a) => !a.loaded);
  body.append(el('div', { class: 'tb-cat' },
    t('plugins.available') + ' (' + avail.length + ')'));
  if (!avail.length) {
    body.append(el('div', { class: 'tb-empty' },
      d.root_exists ? t('plugins.none_available')
                    : t('plugins.root_missing')));
  }
  for (const a of avail) {
    const row = el('div', { class: 'setting' });
    row.append(el('div', { class: 'setting-label mono' }, a.file));
    const btn = el('button', { class: 'seg' }, t('plugins.load'));
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      try {
        await api.pluginLoad(a.path);
        toast(t('plugins.loaded_ok'));
      } catch (e) {
        // The host logged the specific reason (ABI mismatch, missing
        // entry point, throwing register); surface it rather than a
        // generic failure.
        toast(String((e && e.message) || e), 'error');
      }
      render(body);
    });
    row.append(btn);
    body.append(row);
  }
}

function loadedRow(p, body) {
  const row = el('div', { class: 'setting' });
  const label = el('div', { class: 'setting-label' });
  label.append(el('span', { class: 'mono' }, p.name));
  if (p.version) { label.append(el('span', { class: 'setting-unit' }, ' v' + p.version)); }
  if (!p.enabled) {
    // Both facts at once: switched off, and still in memory. Either
    // alone would mislead.
    label.append(el('span', { class: 'plugin-off' },
      ' ' + t('plugins.disabled_resident')));
  }
  row.append(label);

  const meta = [];
  if (p.vendor) { meta.push(p.vendor); }
  if (p.license) { meta.push(p.license); }
  meta.push(t('plugins.stages') + ': ' + (p.stage_count || 0));
  row.append(el('div', { class: 'setting-desc' }, meta.join(' · ')));
  if (p.description) {
    row.append(el('div', { class: 'setting-desc' }, p.description));
  }
  row.append(el('div', { class: 'setting-desc mono' }, p.path));

  const btn = el('button', { class: 'seg' },
    p.enabled ? t('plugins.disable') : t('plugins.enable'));
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await api.pluginEnabled(p.name, !p.enabled);
    } catch (e) {
      toast(String((e && e.message) || e), 'error');
    }
    render(body);
  });
  row.append(btn);
  return row;
}
