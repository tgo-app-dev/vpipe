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
//
// AUTO-APPLY. `opts.onCommit` is called the moment a field fully
// determines its own value -- a blur after an edit, a tri-state flip, a
// picked path or model -- so an edit reaches the stage without a trip to
// the Apply button, exactly as on the desktop. The JSON textareas
// (array/object/any) are the one exclusion: a half-typed blob would
// throw on every blur, so those commit only via Apply. Free-text and
// number boxes wait for `change`, which fires on blur-AFTER-EDIT, so
// tabbing past an untouched field costs nothing.

import { el, clear, openModal } from '../dom.js';
import { api, MODEL_REGISTRY_DB } from '../api.js';
import { t, tOr } from '../i18n.js';
import { openFsSheet, dirOf, baseOf } from './phone-fs.js';
import { compatibleModels } from '../model-filter.js';

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

// Build one field. Returns { el, read, key, value }.
//
// `value()` is the field's RAW text, which is what the model picker's
// parent-compatibility filter reads off the sibling fields of the same
// form (see openModelPicker). It is deliberately not `read()`: that
// answers "what goes in the PUT body" and returns undefined for unset,
// where this only has to say what is currently typed.
export function configField(f, opts = {}) {
  const ro = !!opts.readOnly;
  const type = f.type || 'any';
  const present = f.present !== false;
  const defTxt = fmtDefault(f.default);
  const commit = () => { if (opts.onCommit) { opts.onCommit(); } };
  // The element holding the field's text, for value(). Left null for the
  // types whose value can never name a model (bool).
  let valueEl = null;
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
  // What the pickers at the bottom drive. Whether a field can be BROWSED
  // for is a property of the field (`is_path`, `suggest_db`), not of its
  // type -- so the branches below record their input here and the pickers
  // attach once, after the type dispatch. Doing it inside the string
  // branch (as this did) silently skipped every array/any path field:
  // load-image's `url` is `any` + is_path and got no picker at all on
  // this shell, while the desktop offered one. Left null for the types no
  // picker applies to.
  //   el     the element holding the value
  //   multi  the value is a JSON array -- a pick APPENDS rather than
  //          replaces, matching the desktop's array path fields
  //   mark   tell the field it is no longer "unset"
  //   before insert the picker button ahead of this one (keeps ⌫ last)
  let pick = null;

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
    // Picking an option IS the whole edit -- there is no blur to wait for.
    sel.addEventListener('change', commit);
    row.append(sel);
    read = () => (sel.value === 'unset' ? undefined : sel.value === 'true');

  } else if (JSONISH.has(type)) {
    const ta = el('textarea', { class: 'ph-input ph-textarea',
      rows: '3', placeholder: defTxt, readonly: ro,
      autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' });
    ta.value = present ? fmtDefault(f.current) : '';
    row.append(ta);
    valueEl = ta;
    // No blur-commit here: a JSON box is half-typed for most of the time
    // it is focused, and committing on every blur would just throw. A
    // PICKED path still commits -- it writes a complete array itself.
    pick = { el: ta, multi: true, mark: () => {}, before: null };
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
    inp.addEventListener('change', commit);
    row.append(inp);
    valueEl = inp;
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
    // `change` fires on blur-after-edit (and on the keyboard's Done), so
    // an untouched field the user merely scrolled past commits nothing.
    inp.addEventListener('change', commit);
    row.append(inp);
    valueEl = inp;

    // The explicit way back to "unset" for a tri-state type: clearing
    // the box alone can't say it, since "" is itself a value.
    let unsetBtn = null;
    if (!ro) {
      unsetBtn = el('button', { class: 'ph-f-btn',
        title: t('phone.cfg_unset_title'),
        onclick: () => {
          unset = true;
          inp.value = '';
          wrap.classList.add('unset');
          // Clearing is a complete edit on its own -- no blur follows a
          // button tap, so this is the only chance to commit it.
          commit();
        } }, '⌫');
      row.append(unsetBtn);
    }
    if (unset) { wrap.classList.add('unset'); }
    pick = { el: inp, multi: false, mark: markSet, before: unsetBtn };
    read = () => (unset ? undefined : inp.value);
  }

  // ---- pickers -----------------------------------------------------
  // The two field flavours that are painful to type on a phone and that
  // the stage already tells us how to browse for. Attached by FIELD
  // property, so an array/any path field gets one too.
  if (pick && !ro) {
    const addBtn = (glyph, onclick) => {
      const b = el('button', { class: 'ph-f-btn', onclick }, glyph);
      if (pick.before) { row.insertBefore(b, pick.before); }
      else { row.append(b); }
    };
    // The last path already in the field, so the sheet opens where the
    // previous pick left off rather than at the sandbox root.
    const lastPath = () => {
      const cur = (pick.el.value || '').trim();
      if (!cur) { return ''; }
      if (!pick.multi) { return cur; }
      try {
        const v = JSON.parse(cur);
        const arr = Array.isArray(v) ? v : [v];
        return arr.length ? String(arr[arr.length - 1]) : '';
      } catch (e) { return cur; }
    };
    // A multi field APPENDS: openFsSheet picks one path at a time, so
    // tapping the button repeatedly is how a list gets built. Anything
    // unparseable in the box is kept as the first element rather than
    // discarded -- a half-typed entry is still the user's.
    const setPath = (p) => {
      if (!pick.multi) {
        pick.el.value = p;
      } else {
        let arr = [];
        const cur = (pick.el.value || '').trim();
        if (cur) {
          try {
            const v = JSON.parse(cur);
            arr = Array.isArray(v) ? v : [v];
          } catch (e) { arr = [cur]; }
        }
        arr.push(p);
        pick.el.value = JSON.stringify(arr, null, 2);
      }
      pick.mark();
      // A pick is a deliberate value commit, and a programmatic set fires
      // no `change` -- so say so here. This is also what commits the JSON
      // array path fields, which have no blur-commit of their own.
      commit();
    };
    if (f.is_path) {
      const kind = f.path_kind || '';
      addBtn('…', () => {
        openFsSheet({
          title: t('phone.pick_path'),
          start: dirOf(lastPath()) || '',
          pickDirs: kind === 'dir' || kind === 'directory',
          exts: extsFromFilter(f.path_filter),
          nameField: !!f.path_write && kind !== 'dir',
          defaultName: f.path_write ? baseOf(lastPath()) : '',
          onPick: setPath,
        });
      });
    } else if (f.suggest_db === MODEL_REGISTRY_DB) {
      addBtn('⌄', () => openModelPicker(f, setPath, opts.peerValues));
    }
  }

  return {
    el: wrap,
    read,
    key: f.key,
    value: () => (valueEl ? String(valueEl.value ?? '').trim() : ''),
  };
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

// Installed models for a model-registry `suggest_db` field, filtered by
// the SAME three rules the desktop's model browser applies -- a phone
// offering a choice the stage would reject is worse than a desktop doing
// it, because there is less room to notice the mistake afterwards:
//
//   (a) the field's model_type allow-list (`suggest_db_type`, which may
//       name several types comma-separated). A field that pins NO type
//       still isn't unfiltered: it shows plain models only, hiding
//       datasets and bare supplements, which are never a valid value for
//       a model field.
//   (b) the field's required I/O modalities (`need_inputs` /
//       `need_outputs`): the model's inputs must cover every required
//       input and its outputs every required output. This is what keeps
//       text-chat's LM field to text->text models.
//   (c) parent compatibility: a supplement (vision tower, LoRA) appears
//       only when the model chosen in a SIBLING field of this form is
//       one it attaches to. `peerValues` is what carries that cross-field
//       state -- the desktop reads it back off the DOM by field id, which
//       this shell has no equivalent of, so the caller passes a reader.
//
// The value written is the registry key, exactly as the desktop writes it.
async function openModelPicker(f, onPick, peerValues) {
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
  const all = (data && data.models) || [];

  // WHAT THIS STAGE FIELD WILL ACCEPT: its suggest_db_type allow-list,
  // its required I/O modalities, and parent compatibility against the
  // installed models the form's OTHER fields already name. Shared with
  // the desktop browser -- see model-filter.js, where a copy of this
  // rule drifted into hiding a supplement's own siblings.
  const models = compatibleModels(all, f, peerValues ? peerValues() : []);
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
