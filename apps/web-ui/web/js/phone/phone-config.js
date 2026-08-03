// Editable stage-configuration fields for the phone.
//
// One field per schema entry (`/api/pipelines/{id}/stages/{sid}/config`
// returns `schema: [{key, type, required, doc, default, current, present,
// is_path, suggest_db, ...}]`). Each returns a `read()` whose result goes
// into the PUT body, or `undefined` for "leave this key out".
//
// UNSET is a real value, distinct from empty, and the distinction is what
// makes mutually-exclusive keys work (chrono takes frequency_hz OR
// period_ms, and sending both is an error) -- so a blank box must be able
// to mean "the stage's default" and not "the empty string". The desktop
// resolves that per type and this follows the same rule, because the
// backend it posts to is the same:
//
//   bool / string / text   tri-state. An omitted key and a key present as
//                          "" are different configs, so `unset` is
//                          tracked explicitly and moved with the ⌫
//                          button rather than inferred from emptiness.
//   int / uint / real      blank == unset. There is no empty number.
//   array / object / any   blank == unset. An empty box is not JSON.
//
// The schema default fills the placeholder throughout, so a blank field
// still says what it will resolve to.

import { el, clear, openModal } from '../dom.js';
import { api, MODEL_REGISTRY_DB } from '../api.js';
import { t, tOr } from '../i18n.js';
import { openFsSheet, dirOf, baseOf } from './phone-fs.js';

const NUMERIC = new Set(['int', 'uint', 'real']);
const JSONISH = new Set(['array', 'object', 'any']);

function fmtDefault(v) {
  if (v === null || v === undefined) { return ''; }
  if (typeof v === 'string') { return v; }
  if (typeof v === 'object') {
    try { return JSON.stringify(v); } catch (e) { return ''; }
  }
  return String(v);
}

// Build one field. Returns { el, read, key }.
export function configField(f, opts = {}) {
  const ro = !!opts.readOnly;
  const type = f.type || 'any';
  const present = f.present !== false;
  const defTxt = fmtDefault(f.default);
  // Tracks the tri-state for the types that have one; for the rest it is
  // recomputed from emptiness at read() time and this is only the seed.
  let unset = !present;

  const label = el('div', { class: 'ph-f-label' },
    el('span', { class: 'ph-f-key' }, f.key),
    f.required ? el('span', { class: 'ph-f-req' }, '*') : null,
    el('span', { class: 'ph-f-type' }, type));
  const row = el('div', { class: 'ph-f-row' });
  const wrap = el('div', { class: 'ph-field' }, label, row);
  const doc = tOr('cfg.' + (opts.stageType || '') + '.' + f.key, f.doc || '');
  if (doc) { wrap.append(el('div', { class: 'ph-f-doc' }, doc)); }

  let read;

  if (type === 'bool') {
    // Three states, so a two-state checkbox won't do: a select says
    // which of them is live without a second control.
    const sel = el('select', { class: 'ph-input ph-select', disabled: ro });
    sel.append(
      el('option', { value: 'unset' },
         t('phone.cfg_default', { v: defTxt === 'true' ? 'true' : 'false' })),
      el('option', { value: 'true' }, 'true'),
      el('option', { value: 'false' }, 'false'));
    sel.value = unset ? 'unset' : (f.current ? 'true' : 'false');
    row.append(sel);
    read = () => (sel.value === 'unset' ? undefined : sel.value === 'true');

  } else if (JSONISH.has(type)) {
    const ta = el('textarea', { class: 'ph-input ph-textarea',
      rows: '3', placeholder: defTxt, readonly: ro,
      autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' });
    ta.value = present ? fmtDefault(f.current) : '';
    row.append(ta);
    read = () => {
      const s = ta.value.trim();
      if (!s) { return undefined; }
      try { return JSON.parse(s); }
      catch (e) { throw new Error(t('phone.cfg_bad_json', { key: f.key })); }
    };

  } else if (NUMERIC.has(type)) {
    const inp = el('input', { type: 'number', class: 'ph-input',
      placeholder: defTxt, readonly: ro,
      // Bring up the right soft keyboard: a decimal point only where a
      // real number can use one.
      inputmode: type === 'real' ? 'decimal' : 'numeric',
      step: type === 'real' ? 'any' : '1',
      min: type === 'uint' ? '0' : null });
    inp.value = present ? fmtDefault(f.current) : '';
    row.append(inp);
    read = () => {
      const s = inp.value.trim();
      if (!s) { return undefined; }
      const n = type === 'real' ? parseFloat(s) : parseInt(s, 10);
      if (!Number.isFinite(n)) {
        throw new Error(t('phone.cfg_bad_number', { key: f.key }));
      }
      return n;
    };

  } else {
    // string / text, plus anything unrecognised.
    const multi = type === 'text';
    const inp = multi
      ? el('textarea', { class: 'ph-input ph-textarea', rows: '3',
          placeholder: defTxt, readonly: ro })
      : el('input', { type: 'text', class: 'ph-input', placeholder: defTxt,
          readonly: ro, autocapitalize: 'off', autocorrect: 'off',
          spellcheck: 'false' });
    inp.value = present ? String(f.current == null ? '' : f.current) : '';
    const markSet = () => {
      if (unset) { unset = false; wrap.classList.remove('unset'); }
    };
    inp.addEventListener('input', markSet);
    row.append(inp);

    // A picker for the two field flavours that are painful to type on a
    // phone and that the stage already tells us how to browse for.
    if (!ro && f.is_path) {
      row.append(el('button', { class: 'ph-f-btn',
        onclick: () => {
          const kind = f.path_kind || '';
          openFsSheet({
            title: t('phone.pick_path'),
            start: dirOf(inp.value) || '',
            pickDirs: kind === 'dir' || kind === 'directory',
            exts: extsFromFilter(f.path_filter),
            nameField: !!f.path_write && kind !== 'dir',
            defaultName: f.path_write ? baseOf(inp.value) : '',
            onPick: (p) => { inp.value = p; markSet(); },
          });
        } }, '…'));
    } else if (!ro && f.suggest_db === MODEL_REGISTRY_DB) {
      row.append(el('button', { class: 'ph-f-btn',
        onclick: () => openModelPicker(f, (key) => {
          inp.value = key; markSet();
        }) }, '⌄'));
    }

    // The explicit way back to "unset" for a tri-state type: clearing
    // the box alone can't say it, since "" is itself a value.
    if (!ro) {
      row.append(el('button', { class: 'ph-f-btn',
        title: t('phone.cfg_unset_title'),
        onclick: () => {
          unset = true;
          inp.value = '';
          wrap.classList.add('unset');
        } }, '⌫'));
    }
    if (unset) { wrap.classList.add('unset'); }
    read = () => (unset ? undefined : inp.value);
  }

  return { el: wrap, read, key: f.key };
}

// A stage's `path_filter` is a display string like "*.png;*.jpg"; the
// listing API wants dot-led extensions. An unparseable filter yields no
// restriction, which shows too much rather than too little.
function extsFromFilter(filter) {
  if (!filter) { return null; }
  const out = [];
  for (const part of String(filter).split(/[;,\s]+/)) {
    const m = /\.([A-Za-z0-9_]+)$/.exec(part.trim());
    if (m) { out.push('.' + m[1].toLowerCase()); }
  }
  return out.length ? out : null;
}

// Installed models for a model-registry `suggest_db` field. Filtered by the
// field's model_type allow-list only -- the desktop additionally hides a
// supplement whose parent doesn't match the model picked in a SIBLING
// field of the same form, which needs cross-field state this compact
// editor doesn't carry. The value written is the registry key, exactly
// as the desktop writes it.
async function openModelPicker(f, onPick) {
  const list = el('div', { class: 'ph-sheet-list' },
    el('div', { class: 'ph-sheet-hint' }, t('common.loading')));
  const close = openModal({
    title: t('phone.pick_model'),
    className: 'ph-modal',
    body: list,
    actions: [{ label: t('common.cancel'), cancel: true,
                onClick: (c) => c() }],
  });
  let data;
  try { data = await api.modelsInstalled(); }
  catch (e) {
    clear(list).append(el('div', { class: 'ph-sheet-hint' }, e.message));
    return;
  }
  const allowed = (f.suggest_db_type || '')
    .split(',').map((s) => s.trim()).filter(Boolean);
  const models = ((data && data.models) || []).filter(
    (m) => !allowed.length || allowed.includes(m.model_type));
  clear(list);
  if (!models.length) {
    list.append(el('div', { class: 'ph-sheet-hint' },
      t('phone.no_models')));
    return;
  }
  for (const m of models) {
    const sub = [m.model_type, m.param_class, m.variant]
      .filter(Boolean).join(' · ');
    list.append(el('button', { class: 'ph-sheet-item col',
      onclick: () => { close(); onPick(m.key); } },
      el('span', { class: 'ph-sheet-name' }, m.name || m.key),
      sub ? el('span', { class: 'ph-sheet-sub' }, sub) : null));
  }
}

// Collect a PUT body from built fields. Throws with a user-facing
// message when a field can't be parsed, and does so BEFORE anything is
// sent -- a config that is half-applied is worse than one refused.
export function readConfig(fields) {
  const cfg = {};
  for (const f of fields) {
    const v = f.read();
    if (v === undefined) { continue; }      // intentionally unset
    cfg[f.key] = v;
  }
  return cfg;
}
