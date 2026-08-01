// Composer dock trees: the geometry behind splitting a DOCKED panel.
//
// A dock region used to be a flat strip -- an array of panels sharing one
// fixed direction, so a left-docked panel could only ever be split vertically
// (another row in the same column). Letting any docked panel split EITHER way
// (as the User I/O workspace does) makes each region a binary tree instead:
//
//   leaf   { kind:'leaf',  panel }
//   split  { kind:'split', dir:'h'|'v', ratio, a, b }
//
// `dir` names the DIVIDER, matching the User I/O workspace's convention:
//   'v' -> vertical divider,   a | b   side by side
//   'h' -> horizontal divider, a / b   stacked
// `ratio` is a's share of the parent rectangle along the split axis.
//
// Regions are the four sides plus `center` -- the area a MAXIMIZED panel
// fills. Treating center as just another dock region is what lets a maximized
// panel be split too; a center tree holding a single leaf IS "maximized".
//
// Everything here is pure geometry + tree surgery on plain objects, with no
// DOM and no imports, so it can be exercised directly (see the jsc harness in
// the review notes) rather than only through a browser.

export const REGIONS = ['left', 'right', 'top', 'bottom', 'center'];

// The direction a region's FIRST split takes by default, i.e. the way its
// legacy flat strip used to stack: side columns grow downward, top/bottom
// strips grow sideways.
export function defaultDir(region) {
  return (region === 'left' || region === 'right') ? 'h' : 'v';
}

export const leafOf = (panel) => ({ kind: 'leaf', panel });

// ---- queries -------------------------------------------------------------

export function forEachLeaf(node, fn) {
  if (!node) { return; }
  if (node.kind === 'leaf') { fn(node); return; }
  forEachLeaf(node.a, fn);
  forEachLeaf(node.b, fn);
}

export function leafCount(node) {
  let n = 0;
  forEachLeaf(node, () => { ++n; });
  return n;
}

export function panels(node) {
  const out = [];
  forEachLeaf(node, (l) => out.push(l.panel));
  return out;
}

export function findLeaf(node, panel) {
  let hit = null;
  forEachLeaf(node, (l) => { if (l.panel === panel) { hit = l; } });
  return hit;
}

// ---- surgery -------------------------------------------------------------

// Replace `target`'s leaf with a split holding it plus `panel`. The new pane
// goes SECOND (right / below), matching the workspace's "existing first".
// Returns the (possibly new) root.
export function splitAt(root, target, dir, panel) {
  const leaf = findLeaf(root, target);
  if (!leaf) { return root; }
  const moved = { kind: 'leaf', panel: leaf.panel };
  // Mutate the found node in place into a split so any parent link stays
  // valid without a rebuild.
  leaf.kind = 'split';
  leaf.dir = dir;
  leaf.ratio = 0.5;
  leaf.a = moved;
  leaf.b = leafOf(panel);
  delete leaf.panel;
  return root;
}

// Drop `panel`; its sibling takes the whole rectangle. Returns the new root,
// or null when the region is now empty.
export function removeLeaf(root, panel) {
  if (!root) { return null; }
  if (root.kind === 'leaf') { return root.panel === panel ? null : root; }
  const prune = (node) => {
    if (node.kind === 'leaf') { return node.panel === panel ? null : node; }
    const a = prune(node.a);
    const b = prune(node.b);
    if (a === null) { return b; }
    if (b === null) { return a; }
    node.a = a; node.b = b;
    return node;
  };
  return prune(root);
}

// Append a panel to a region that may be empty, splitting the LAST leaf along
// `dir` -- the behaviour the old flat strip had when a panel was docked to a
// side that already held some.
export function appendLeaf(root, panel, dir) {
  if (!root) { return leafOf(panel); }
  let node = root;
  while (node.kind === 'split') { node = node.b; }
  return splitAt(root, node.panel, dir, panel);
}

// ---- geometry ------------------------------------------------------------

const clamp01 = (v) => (v < 0.05 ? 0.05 : (v > 0.95 ? 0.95 : v));

// Walk the tree over a rectangle, reporting where each panel lands.
// `rect` is {x, y, w, h}. Returns [{panel, x, y, w, h}].
export function layout(node, rect) {
  const out = [];
  const walk = (n, x, y, w, h) => {
    if (!n) { return; }
    if (n.kind === 'leaf') { out.push({ panel: n.panel, x, y, w, h }); return; }
    const r = clamp01(typeof n.ratio === 'number' ? n.ratio : 0.5);
    if (n.dir === 'v') {                 // vertical divider: a | b
      const aw = w * r;
      walk(n.a, x, y, aw, h);
      walk(n.b, x + aw, y, w - aw, h);
    } else {                             // horizontal divider: a / b
      const ah = h * r;
      walk(n.a, x, y, w, ah);
      walk(n.b, x, y + ah, w, h - ah);
    }
  };
  walk(node, rect.x, rect.y, rect.w, rect.h);
  return out;
}

// The draggable divider of every split, in the same coordinate space.
// Returns [{node, dir, x, y, w, h, span}] where `span` is the parent extent
// along the split axis (what a drag delta is divided by to get a ratio).
export function dividers(node, rect) {
  const out = [];
  const walk = (n, x, y, w, h) => {
    if (!n || n.kind !== 'split') { return; }
    const r = clamp01(typeof n.ratio === 'number' ? n.ratio : 0.5);
    if (n.dir === 'v') {
      const aw = w * r;
      out.push({ node: n, dir: 'v', x: x + aw, y, w: 0, h, span: w });
      walk(n.a, x, y, aw, h);
      walk(n.b, x + aw, y, w - aw, h);
    } else {
      const ah = h * r;
      out.push({ node: n, dir: 'h', x, y: y + ah, w, h: 0, span: h });
      walk(n.a, x, y, w, ah);
      walk(n.b, x, y + ah, w, h - ah);
    }
  };
  walk(node, rect.x, rect.y, rect.w, rect.h);
  return out;
}

// ---- serialization -------------------------------------------------------

// `cfgOf(panel)` returns the panel's persisted blob; the tree adds only its
// own shape around it.
export function toJSON(node, cfgOf) {
  if (!node) { return null; }
  if (node.kind === 'leaf') { return { ...cfgOf(node.panel) }; }
  return { split: node.dir,
           ratio: +clamp01(node.ratio).toFixed(4),
           a: toJSON(node.a, cfgOf),
           b: toJSON(node.b, cfgOf) };
}

// `mk(spec)` builds a panel from a persisted blob (or returns null, e.g. an
// unknown panel type); a branch whose children both fail collapses away.
export function fromJSON(json, mk) {
  if (!json || typeof json !== 'object') { return null; }
  if (json.split === 'h' || json.split === 'v') {
    const a = fromJSON(json.a, mk);
    const b = fromJSON(json.b, mk);
    if (!a) { return b; }
    if (!b) { return a; }
    const r = Number(json.ratio);
    return { kind: 'split', dir: json.split,
             ratio: Number.isFinite(r) ? clamp01(r) : 0.5, a, b };
  }
  const p = mk(json);
  return p ? leafOf(p) : null;
}

// Rebuild a tree from the pre-split flat strip format: panels in order along
// `dir`, sized by their `extent` weights. Layouts saved before splitting
// existed (including any stored in a pipeline's `aux`) load through here.
export function fromLegacy(specs, dir, mk) {
  const made = [];
  for (const sp of (specs || [])) {
    const p = mk(sp);
    if (p) {
      made.push({ panel: p, w: Math.max(0.05, Number(sp.extent) || 1) });
    }
  }
  if (!made.length) { return null; }
  // Right-fold so the first panel keeps the first slice: each split gives `a`
  // its own weight against the total of everything after it.
  let node = leafOf(made[made.length - 1].panel);
  let tail = made[made.length - 1].w;
  for (let i = made.length - 2; i >= 0; --i) {
    const total = made[i].w + tail;
    node = { kind: 'split', dir, ratio: clamp01(made[i].w / total),
             a: leafOf(made[i].panel), b: node };
    tail = total;
  }
  return node;
}
