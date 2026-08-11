// Whether one stage's output port can feed another's input port.
//
// The rule is the RUNTIME's, mirrored here: payload types must agree
// unless either side is untyped (pipeline-runtime.cc's edge check), and
// payload tags are a finer constraint on top with OR semantics
// (port_tags_compatible in pipeline/stage-spec.cc). The backend stays
// authoritative -- it re-checks on every connect and returns the error
// that matters. This copy exists only so an editor can say "these two do
// not fit" before the round trip, and so the graph canvas and the
// narrow-view connection editor cannot drift apart about what they
// offer.

// A comma-separated tag list -> trimmed, non-empty tags.
export function parseTags(s) {
  return (s || '').split(',').map((x) => x.trim()).filter(Boolean);
}

// Untyped ("any", or absent) on either side is permitted -- that is what
// the runtime does for legacy / untyped-passthrough stages.
export function typesCompatible(a, b) {
  return !a || !b || a === 'any' || b === 'any' || a === b;
}

// OR semantics: untagged on either side accepts anything, otherwise the
// two sets must intersect. Both arguments are ARRAYS (parseTags output).
export function tagsCompatible(a, b) {
  return !a.length || !b.length || a.some((x) => b.includes(x));
}

// A stage's ports as indexed slots, the DECLARED ones widened to the
// wired count.
//
// The two sources disagree on purpose. `node.iports` / `node.oports` are
// what the pipeline currently has -- a stage authored with three inputs
// wired reports three, even when its type declares seven -- while the
// spec carries the names, docs and types of every slot the type has.
// Connecting to a slot that does not exist yet is exactly what the
// backend's append path is for, so the editor has to offer them; showing
// only the live ports would hide most of a stage's inputs, and showing
// only the declared ones would hide an extra live connection.
//
// `spec` may be null (a type with no registered spec), in which case the
// live ports are all there is to say.
export function portSlots(node, spec, kind) {
  const live = (kind === 'in' ? node.iports : node.oports) || [];
  const decl = (spec ? (kind === 'in' ? spec.iports : spec.oports) : null)
               || [];
  const n = Math.max(decl.length, live.length);
  const slots = [];
  for (let i = 0; i < n; i++) {
    const d = decl[i] || null;
    const l = live[i] || null;
    slots.push({
      index: i,
      name: (d && d.name) || '',
      type: (d && d.type) || (l && l.type) || 'any',
      tags: parseTags((l && l.tags !== undefined) ? l.tags : (d && d.tags)),
      doc:  (d && d.doc) || '',
      live: l !== null,
    });
  }
  return slots;
}

// Why `out` cannot feed `in`, or '' when it can. The string is a REASON
// rather than a bool so the editor can show it against the offending
// candidate instead of just greying it out.
export function slotMismatch(out, inp) {
  if (!typesCompatible(out.type, inp.type)) { return 'type'; }
  if (!tagsCompatible(out.tags, inp.tags)) { return 'tags'; }
  return '';
}
