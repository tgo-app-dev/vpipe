// Linearized DAG layout -- the `git log --graph --oneline` rendering of
// a stage graph.
//
// One row per stage, in dependency order, with the connections drawn in
// a gutter down the left. It exists because that form SURVIVES A NARROW
// COLUMN: it needs one text line per node and a few pixels of lane per
// concurrent branch, however deep the graph gets, where the canvas in
// graph.js needs 176px per node plus room to route around it.
//
// Shared by the phone shell (which has no room for a canvas at all) and
// by the desktop pipeline editor (which falls back to it when its pane
// is dragged too narrow to draw one) -- so the two cannot disagree about
// what a pipeline's shape looks like.

// Gutter geometry. ROW_H is fixed on purpose: the lane art is drawn per
// row in its own SVG, and lanes only line up across rows if every row is
// the same height (hence the single-line, ellipsised row text).
export const ROW_H    = 46;
const LANE_W   = 14;
const LANE_X0  = 11;
// Lanes past this are clamped onto the last column rather than widening
// the gutter without bound -- on a phone, a graph that wide is unreadable
// either way, and the rows must keep their text.
export const MAX_LANES = 7;


export function laneX(i) {
  return LANE_X0 + Math.min(i, MAX_LANES - 1) * LANE_W;
}

// Dependency order, stable in the backend's node order (Kahn). An
// already-ordered graph comes back unchanged, so the rows match the
// order the pipeline file lists its stages in whenever that is already
// topological. A cycle -- which the runtime rejects at launch, but which
// must not hang or drop stages here -- leaves its members to the tail.
export function topoOrder(nodes, edges) {
  const idx = new Map();
  nodes.forEach((n, i) => idx.set(n.id, i));
  const indeg = nodes.map(() => 0);
  const outs = nodes.map(() => []);
  for (const e of edges) {
    const a = idx.get(e.from), b = idx.get(e.to);
    if (a === undefined || b === undefined) { continue; }
    outs[a].push(b);
    indeg[b] += 1;
  }
  const placed = nodes.map(() => false);
  const order = [];
  for (;;) {
    let progress = false;
    for (let i = 0; i < nodes.length; i++) {
      if (placed[i] || indeg[i] > 0) { continue; }
      placed[i] = true;
      progress = true;
      order.push(i);
      for (const j of outs[i]) { indeg[j] -= 1; }
    }
    if (!progress) { break; }
  }
  for (let i = 0; i < nodes.length; i++) {
    if (!placed[i]) { order.push(i); }
  }
  return order.map((i) => nodes[i]);
}

// Assign each row a LANE, the way git lays commits out.
//
// A lane is a pending connection: once a stage is drawn, every consumer
// it feeds reserves a lane that runs down the gutter until that consumer
// gets its own row. A stage sits on the LEFTMOST lane that was waiting
// for it; the other lanes waiting for it merge in and are released; its
// first consumer inherits its lane and the rest fan out to the right.
// That is the whole of it -- the rest is drawing.
//
// Consumers are taken nearest-first so the closest one inherits the lane
// and the long connections end up on the outside, which is what stops
// them from cutting across the middle of the list.
export function assignLanes(rows, edges) {
  const rowOf = new Map();
  rows.forEach((n, i) => rowOf.set(n.id, i));
  const outs = new Map();
  for (const e of edges) {
    const a = rowOf.get(e.from), b = rowOf.get(e.to);
    // A back edge would break "producer above consumer" and there is no
    // lane discipline that survives it. Skip it in the gutter; the
    // stage detail still lists the connection, so nothing is hidden.
    if (a === undefined || b === undefined || b <= a) { continue; }
    if (!outs.has(e.from)) { outs.set(e.from, []); }
    outs.get(e.from).push(e);
  }
  for (const list of outs.values()) {
    list.sort((x, y) => rowOf.get(x.to) - rowOf.get(y.to));
  }

  const lanes = [];               // lane -> stage id it is waiting for
  const firstFree = () => {
    const i = lanes.indexOf(null);
    if (i >= 0) { return i; }
    lanes.push(null);
    return lanes.length - 1;
  };

  const out = [];
  let width = 1;
  for (let r = 0; r < rows.length; r++) {
    const id = rows[r].id;
    const landing = [];
    const through = [];
    for (let i = 0; i < lanes.length; i++) {
      if (lanes[i] === id) { landing.push(i); }
      else if (lanes[i] !== null) { through.push(i); }
    }
    let col;
    if (landing.length) {
      col = landing[0];
      for (const i of landing.slice(1)) { lanes[i] = null; }
    } else {
      col = firstFree();          // a source: nothing was waiting for it
    }
    lanes[col] = null;            // the arriving connections end here

    const branches = [];
    for (const e of (outs.get(id) || [])) {
      const lane = branches.length === 0 ? col : firstFree();
      lanes[lane] = e.to;
      branches.push(lane);
    }
    out.push({ node: rows[r], col, merges: landing.slice(1), through,
               branches, fromAbove: landing.length > 0 });
    width = Math.max(width, lanes.length, col + 1);
    // Trailing free lanes are dropped so the gutter narrows again once a
    // fan-out has been consumed.
    while (lanes.length && lanes[lanes.length - 1] === null) { lanes.pop(); }
  }
  return { rows: out, width: Math.min(width, MAX_LANES) };
}

// The lane art for one row, as an SVG the width of the gutter.
export function laneArt(d, width, danger) {
  const w = laneX(width - 1) + LANE_X0;
  const mid = ROW_H / 2;
  const xc = laneX(d.col);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'lane-art');
  svg.setAttribute('width', String(w));
  svg.setAttribute('height', String(ROW_H));
  svg.setAttribute('viewBox', `0 0 ${w} ${ROW_H}`);
  const path = (dstr, cls) => {
    const p = document.createElementNS(svg.namespaceURI, 'path');
    p.setAttribute('d', dstr);
    p.setAttribute('class', cls || 'lane');
    svg.append(p);
  };

  for (const i of d.through) {
    const x = laneX(i);
    path(`M ${x} 0 V ${ROW_H}`);
  }
  if (d.fromAbove) { path(`M ${xc} 0 V ${mid}`); }
  for (const i of d.merges) {                    // joins into the node
    const x = laneX(i);
    path(`M ${x} 0 C ${x} ${mid * 0.55} ${xc} ${mid * 0.45} ${xc} ${mid}`);
  }
  for (const i of d.branches) {                  // fans out below it
    const x = laneX(i);
    if (i === d.col) { path(`M ${xc} ${mid} V ${ROW_H}`); continue; }
    const h = ROW_H - mid;
    path(`M ${xc} ${mid} C ${xc} ${mid + h * 0.45} `
       + `${x} ${mid + h * 0.55} ${x} ${ROW_H}`);
  }
  const dot = document.createElementNS(svg.namespaceURI, 'circle');
  dot.setAttribute('cx', String(xc));
  dot.setAttribute('cy', String(mid));
  dot.setAttribute('r', '4.5');
  dot.setAttribute('class', 'lane-dot' + (danger ? ' bad' : ''));
  svg.append(dot);
  return svg;
}


// Pixel width of a gutter holding `width` lanes -- the same arithmetic
// laneArt() sizes its SVG with, exported so a caller can lay out the
// column beside it without re-deriving it.
export function gutterWidth(width) {
  return laneX(width - 1) + LANE_X0;
}
