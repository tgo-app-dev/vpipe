// Narrowing the installed-model list in a picker.
//
// This is the OPERATOR's filter and sits on top of the compatibility
// rules, which are the FIELD's and are not negotiable: a stage's
// suggest_db_type allow-list and its need_inputs / need_outputs decide
// what may be offered at all, and nothing here can widen that. What
// this does is make a list that is already correct findable, which
// stops being the same thing somewhere around a couple of dozen
// installed models.
//
// Pure functions in their own module for the reason port-compat.js is:
// two pickers (the desktop browser and the phone sheet) ask the same
// question, and a rule copied into both drifts.

// The modality vocabulary, in the order chips are drawn. Matches the
// catalogue's -- see stages/model-catalog.h.
export const MODALITIES = ['text', 'image', 'audio', 'video'];

// Everything about a model that a picker row SHOWS, lowercased for
// substring matching. Keeping the two in step is the point: anything
// readable in a row should also be typeable, and a field that is
// searchable but invisible is a hidden feature.
export function modelHaystack(m) {
  return [m.key, m.variant, m.model_type, m.family, m.param_class, m.name]
    .filter(Boolean).join(' ').toLowerCase();
}

// The modalities actually present under `key` ("inputs" / "outputs")
// across `models`, in MODALITIES order.
//
// A picker offers a chip only for these. A chip for a modality nothing
// has can only ever empty the list, which reads as a broken filter
// rather than as an absent modality.
export function modalitiesPresent(models, key) {
  const seen = new Set();
  for (const m of models) {
    for (const x of (m[key] || [])) { seen.add(x); }
  }
  return MODALITIES.filter((x) => seen.has(x));
}

// Does `m` survive the filter?
//
// `query` is a lowercased substring (empty passes everything). `wantIn`
// and `wantOut` are Sets of modalities, and the semantics are AND --
// selecting `image` and `audio` asks for a model that takes BOTH, which
// is the same rule need_inputs already uses. OR would be the wrong
// default here: the reason to pick two is to describe one model, not to
// widen the net.
export function modelMatches(m, { query = '', wantIn, wantOut } = {}) {
  if (query && !modelHaystack(m).includes(query)) { return false; }
  const has = (arr, x) => (arr || []).includes(x);
  for (const x of (wantIn || [])) {
    if (!has(m.inputs, x)) { return false; }
  }
  for (const x of (wantOut || [])) {
    if (!has(m.outputs, x)) { return false; }
  }
  return true;
}

// ---------------------------------------------------------------------
// COMPATIBILITY -- the FIELD's rules, not the operator's.
//
// Everything above narrows a list that is already correct. This decides
// what may be offered at all, and it lives here for the reason the rest
// does: the desktop browser and the phone sheet ask the same question,
// and it WAS copied into both, where it drifted into the bug below.

// Does the field accept `m` at all? Its suggest_db_type allow-list (an
// untyped field shows plain models only -- no datasets, no bare
// supplements) and its required I/O modalities.
export function fieldAccepts(m, field) {
  const csv = (v) => (v || '').split(',').map((s) => s.trim()).filter(Boolean);
  const allowed = csv(field.suggest_db_type);
  const needIn = csv(field.need_inputs);
  const needOut = csv(field.need_outputs);
  const ioOk = needIn.every((x) => (m.inputs || []).includes(x))
    && needOut.every((x) => (m.outputs || []).includes(x));
  return (allowed.length
    ? allowed.includes(m.model_type)
    : (m.category || 'model') === 'model') && ioOk;
}

// Is `m` compatible with what the form's OTHER fields already name?
//
// `picked` are the installed models those fields resolved to. The rule
// hides what is known to be incompatible, never what is merely
// unconfirmed -- so an empty form offers everything.
//
// A SUPPLEMENT IS NOT A PARENT, and this is where the rule went wrong.
// Nothing in the catalogue attaches to a LoRA or a VDN branch, so a
// chosen supplement treated as a candidate parent matches nothing and
// vetoes its own siblings: picking the VDN branch on
// minimax-h3-model-config hid every H3 Turbo LoRA, because the branch's
// model_type is minimax-h3-vdn while the LoRAs attach to
// minimax-h3-fl2va.
//
// What a chosen supplement IS is evidence about the parent -- it names
// what it attaches to. On a form whose parent is chosen on ANOTHER stage
// that is the only evidence there is, which is every H3 config form: the
// DiT comes from model-select. Two supplements naming the same parent
// are siblings, and a Krea-2 LoRA beside an H3 branch is still hidden.
export function parentAccepts(m, picked) {
  if (!m.parent_model_type) { return true; }        // not a supplement
  const list = picked || [];
  const parents = list.filter((p) => !p.parent_model_type);
  const siblings = list.filter((p) => p.parent_model_type);
  const classOk = (a, b) => !a || !b || a.toLowerCase() === b.toLowerCase();
  // A real parent, when the form has one.
  if (parents.some((p) => p.model_type === m.parent_model_type
      // Case-insensitive: a registry record written before the catalog
      // switched "e4b" -> "E4B" still matches the tower's "E4B".
      && classOk(m.parent_param_class, p.param_class))) {
    return true;
  }
  // Else a sibling's word for it.
  if (siblings.length) {
    return siblings.some((s) => s.parent_model_type === m.parent_model_type
      && classOk(m.parent_param_class, s.parent_param_class));
  }
  return !parents.length;                           // nothing chosen yet
}

// The installed models a field may offer, given what the form's other
// fields name. `chosen` is those fields' raw values; a value matches a
// model by registry key or by hf_path.
export function compatibleModels(all, field, chosen) {
  const want = (chosen instanceof Set) ? chosen : new Set(chosen || []);
  const picked = (all || []).filter(
    (m) => want.has(m.key) || want.has(m.hf_path));
  return (all || []).filter(
    (m) => fieldAccepts(m, field) && parentAccepts(m, picked));
}
