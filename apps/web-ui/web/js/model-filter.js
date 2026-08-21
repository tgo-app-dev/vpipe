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
