// Layered-DAG layout + SVG renderer for the stage graph (dagre/netron
// style), with drag-to-pan and zoom.
//
// Pipeline:
//   1. Layer assignment   -- longest-path rank from any source.
//   2. Dummy-node insertion -- every edge spanning >1 layer becomes a
//      chain of unit-length edges through a per-layer "dummy" node.
//      That single trick converts the routing problem into a sequence
//      of adjacent-column hops: long edges get a dedicated y-slot in
//      every column they cross, so they never hide behind a real node
//      and never pile on top of each other unless the crossing-min
//      pass leaves them stacked.
//   3. Crossing minimisation -- alternating barycentre sweeps reorder
//      each layer's items (real and dummy) to minimise edge crossings.
//   4. Y assignment       -- real nodes get their full height; dummies
//      get a small "lane" slot with tighter spacing.
//   5. Edge rendering     -- a smooth spline through (source port ->
//      dummies' centres -> destination port) with horizontal tangents.
//
// Pan/zoom uses the SVG viewBox over fixed "world" coordinates. The
// view state {k, cx, cy} lives in opts.view and is mutated in place so
// it survives the frequent re-renders (selection changes); callers
// reset it (pass a fresh {}) only on pipeline switch, which refits.

import { el, svgEl } from './dom.js';
import { tOr } from './i18n.js';

const NODE_W    = 176;
const COL_GAP   = 74;
const ROW_GAP   = 34;
// Minimum clearance enforced when de-overlapping a DROPPED box. Kept small
// (a hair, not a full ROW_GAP) so a box only shifts when it would actually
// touch a neighbour -- otherwise a small nudge into the empty gap above a
// node reads as "occupied" and snaps the box right back to its old slot.
const OVERLAP_GAP = 6;
const HEADER_H  = 38;
const PORT_TOP  = HEADER_H + 8;
const PORT_GAP  = 18;
const PORT_R    = 4.5;
const PAD       = 28;
const DUMMY_H   = 10;    // routing slot height for through-going edges
const DUMMY_GAP = 6;     // tighter spacing when an adjacent item is a dummy
const ROW_PITCH = 120;   // nominal row height for drop-pin distance caps
const SWEEPS    = 16;    // barycentre crossing-min iterations
const MIN_K     = 0.08;
const MAX_K     = 4;
const FIT_MAX_K = 1;     // Fit never enlarges past 1:1 (only shrinks).

// Box height from the rendered port counts. `inCount` is the number of
// input slots actually drawn (spec-declared, not just the wired ones);
// `addStub` reserves one extra row for the dashed "+ add input" stub
// when it's shown (legacy untyped stages only).
function nodeHeight(inCount, outCount, addStub) {
  const iCount = inCount + (addStub ? 1 : 0);
  const ports = Math.max(iCount, outCount, 1);
  return PORT_TOP + ports * PORT_GAP + 8;
}

export function shortType(t) {
  if (!t || t === 'any') { return 'any'; }
  let s = t;
  const k = s.lastIndexOf('::');
  if (k >= 0) { s = s.slice(k + 2); }
  return s.replace(/Payload$/, '').replace(/Beat$/, '');
}

function truncate(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// `metrics` resolves per-node rendered-port counts:
//   inCount(n)  -> number of input slots drawn (spec-declared)
//   addStub(n)  -> whether the "+ add input" stub adds a row
//
// `seed` (optional) is the PREVIOUS layout's `pos` map (id -> {x,y,...}).
// When present the layout is PLACEMENT-AWARE: columns (depths) are still
// assigned by longest-path rank, but within each column nodes keep their
// current vertical order (sorted by seed Y) instead of being reshuffled
// by crossing-minimisation, and each column is shifted by the median of
// its nodes' Y deltas -- so a re-layout moves stages as little as
// possible. New nodes and routing dummies (which have no seed) are
// slotted near their neighbours by propagating the seed-Y field across
// a few barycentre passes. Without a seed (first layout of a pipeline)
// it falls back to crossing-min from JSON order.
//
// `pins` (optional) is a Map id -> {col, y} of USER-placed positions (a
// stage dropped from the toolbox at a chosen grid cell). A pin sets a
// rank FLOOR for its node (so the column the user chose is honoured, even
// though a brand-new stage has no edges to rank it) and seeds its Y, so
// the drop position survives the auto-layout. Topology can only push a
// pinned node to a LATER column, never earlier.
function computeLayout(graph, metrics, seed, pins) {
  // 1. Build node table + per-node iport-edge list (rank input).
  const nodes = new Map();
  for (const n of graph.nodes) { nodes.set(n.id, { ...n, _in: [] }); }
  for (const e of graph.edges) {
    if (nodes.has(e.to)) { nodes.get(e.to)._in.push(e); }
  }

  // 2. Layer assignment: longest-path rank from any source (DAG).
  //    A cycle short-circuits to rank 0 to keep things finite.
  const rank = new Map();
  const stack = new Set();
  const rankOf = (id) => {
    if (rank.has(id)) { return rank.get(id); }
    if (stack.has(id)) { return 0; }
    stack.add(id);
    // A user pin sets the minimum column so the chosen depth is honoured
    // even for an edge-less new stage; predecessors can only push later.
    const pin = pins && pins.get(id);
    let r = pin && pin.col > 0 ? pin.col : 0;
    for (const e of nodes.get(id)._in) {
      if (nodes.has(e.from)) { r = Math.max(r, rankOf(e.from) + 1); }
    }
    stack.delete(id);
    rank.set(id, r);
    return r;
  };
  for (const n of graph.nodes) { rankOf(n.id); }
  const maxRank = rank.size === 0 ? 0
      : Math.max(0, ...rank.values());

  // 3. Layers seeded with real nodes in their JSON order.
  const layers = [];
  for (let r = 0; r <= maxRank; r++) { layers.push([]); }
  for (const n of graph.nodes) {
    layers[rank.get(n.id)].push({ kind: 'real', id: n.id });
  }

  // 4. Insert dummy nodes for every multi-layer edge so it becomes a
  //    chain of unit-length edges. The dummies provide per-column
  //    routing slots that the crossing-min pass distributes
  //    vertically the same way real nodes are -- which keeps lines
  //    from hiding behind boxes or piling on top of each other.
  const edgeChain = new Map();
  for (const e of graph.edges) {
    if (!nodes.has(e.from) || !nodes.has(e.to)) { continue; }
    const rs = rank.get(e.from);
    const rd = rank.get(e.to);
    if (rd - rs <= 1) { continue; }
    const chain = [];
    for (let r = rs + 1; r < rd; r++) {
      // The dummy id must be unique across (edge, layer). Distinct
      // edges through the same column get distinct slots.
      const did = `__d:${e.from}#${e.from_port}>${e.to}#${e.to_port}@${r}`;
      chain.push({ id: did, rank: r });
      layers[r].push({ kind: 'dummy', id: did, edge: e });
    }
    edgeChain.set(e, chain);
  }

  // 5. Pred/succ adjacency for crossing-min (treating real + dummy
  //    nodes uniformly). For real nodes, port indices are collapsed
  //    -- "B is fed by A" rather than "B's iport k is fed by A's
  //    oport j". The final port-y on the rendered curve still uses
  //    the actual port index, so visual landing is correct.
  const preds = new Map();
  const succs = new Map();
  const push = (m, k, v) => {
    let xs = m.get(k);
    if (!xs) { xs = []; m.set(k, xs); }
    xs.push(v);
  };
  for (const e of graph.edges) {
    if (!nodes.has(e.from) || !nodes.has(e.to)) { continue; }
    const chain = edgeChain.get(e);
    if (!chain) {
      push(preds, e.to, e.from);
      push(succs, e.from, e.to);
    } else {
      push(succs, e.from, chain[0].id);
      push(preds, chain[0].id, e.from);
      for (let i = 1; i < chain.length; i++) {
        push(succs, chain[i - 1].id, chain[i].id);
        push(preds, chain[i].id, chain[i - 1].id);
      }
      push(succs, chain[chain.length - 1].id, e.to);
      push(preds, e.to, chain[chain.length - 1].id);
    }
  }

  // 6. Crossing minimisation: alternating barycentre sweeps. Each
  //    pass reorders one layer by the mean position of each item's
  //    anchor neighbours in the reference layer. Stable sort keeps
  //    no-neighbour items in place. SWEEPS iterations is plenty for
  //    typical graphs.
  function sweep(downward) {
    const start = downward ? 1 : maxRank - 1;
    const end   = downward ? maxRank + 1 : -1;
    const step  = downward ? 1 : -1;
    for (let r = start; r !== end; r += step) {
      const refIdx = r - step;
      if (refIdx < 0 || refIdx > maxRank) { continue; }
      const refLayer = layers[refIdx];
      const refPos = new Map();
      refLayer.forEach((n, i) => refPos.set(n.id, i));
      const neighbours = downward ? preds : succs;
      const layer = layers[r];
      const bary = layer.map((node) => {
        const ns = neighbours.get(node.id) || [];
        let sum = 0;
        let cnt = 0;
        for (const nid of ns) {
          if (refPos.has(nid)) { sum += refPos.get(nid); ++cnt; }
        }
        return cnt > 0 ? sum / cnt : null;
      });
      const order = layer.map((_, i) => i);
      order.sort((a, b) => {
        const ba = bary[a];
        const bb = bary[b];
        if (ba === null && bb === null) { return a - b; }
        if (ba === null) { return a - b; }
        if (bb === null) { return a - b; }
        if (ba !== bb) { return ba - bb; }
        return a - b;
      });
      layers[r] = order.map((i) => layer[i]);
    }
  }
  // A node's reference Y for placement-aware ordering: its previous
  // position if we have one, else its user-pin Y (a freshly dropped
  // stage), else undefined (new + unplaced -> estimated from neighbours).
  const seedY = (id) => {
    const s = seed && seed.get(id);
    if (s && typeof s.y === 'number') { return s.y; }
    const p = pins && pins.get(id);
    if (p && typeof p.y === 'number') { return p.y; }
    return undefined;
  };
  const hasSeed = (seed && seed.size > 0) || (pins && pins.size > 0);
  if (hasSeed) {
    // Placement-aware ordering: keep each column in the user's current
    // vertical order (minimal displacement) instead of reshuffling.
    // Seed real nodes with their previous Y; propagate that Y field to
    // new nodes and routing dummies (no seed) from their neighbours so
    // they land near where they connect, then order each layer by it.
    const y0 = new Map();
    for (const layer of layers) {
      for (const item of layer) {
        if (item.kind !== 'real') { continue; }
        const y = seedY(item.id);
        if (y !== undefined) { y0.set(item.id, y); }
      }
    }
    const propagate = (neighbours) => {
      for (const layer of layers) {
        for (const item of layer) {
          if (y0.has(item.id)) { continue; }
          const ns = neighbours.get(item.id) || [];
          let sum = 0, cnt = 0;
          for (const nid of ns) {
            if (y0.has(nid)) { sum += y0.get(nid); ++cnt; }
          }
          if (cnt > 0) { y0.set(item.id, sum / cnt); }
        }
      }
    };
    for (let it = 0; it < 4; it++) { propagate(preds); propagate(succs); }
    // Anything still unseeded (a wholly disconnected new node) sinks to
    // the bottom of its column, keeping the seeded nodes undisturbed.
    layers.forEach((layer) => {
      let base = 0;
      for (const item of layer) {
        if (y0.has(item.id)) { base = Math.max(base, y0.get(item.id)); }
      }
      for (const item of layer) {
        if (!y0.has(item.id)) { base += 1000; y0.set(item.id, base); }
      }
    });
    layers.forEach((layer, r) => {
      const idx = new Map(layer.map((n, i) => [n.id, i]));
      layers[r] = layer.slice().sort((a, b) => {
        const ya = y0.get(a.id), yb = y0.get(b.id);
        if (ya !== yb) { return ya - yb; }
        return idx.get(a.id) - idx.get(b.id);   // stable tie-break
      });
    });
  } else {
    for (let it = 0; it < SWEEPS; it++) {
      sweep(true);
      sweep(false);
    }
  }

  // 7. Y positioning. Each column is packed top-to-bottom (real nodes get
  //    their full nodeHeight + ROW_GAP; dummies get a small lane slot),
  //    then shifted vertically by `offset`. With a seed, `offset` is the
  //    median of the column's per-node Y deltas (packed vs. seed) so the
  //    bulk of the nodes stay exactly where they were; clamped so the
  //    column top stays on-canvas. Without a seed, offset == PAD (the
  //    original top-packing).
  const pos = new Map();
  let maxX = 0;
  let maxY = 0;
  layers.forEach((layer, r) => {
    const x = PAD + r * (NODE_W + COL_GAP);
    const packed = [];
    let y = 0;
    for (let i = 0; i < layer.length; i++) {
      const item = layer[i];
      const rn = item.kind === 'real' ? nodes.get(item.id) : null;
      const h = item.kind === 'real'
          ? nodeHeight(metrics.inCount(rn), rn.oports.length,
                       metrics.addStub(rn))
          : DUMMY_H;
      packed.push({ item, y, h });
      const nxt = layer[i + 1];
      const dummyAdj = item.kind === 'dummy'
                    || (nxt && nxt.kind === 'dummy');
      y += h + (dummyAdj ? DUMMY_GAP : ROW_GAP);
    }
    let offset = PAD;
    if (hasSeed) {
      const deltas = [];
      for (const p of packed) {
        if (p.item.kind !== 'real') { continue; }
        const sy = seedY(p.item.id);
        if (sy !== undefined) { deltas.push(sy - p.y); }
      }
      if (deltas.length) {
        deltas.sort((a, b) => a - b);
        offset = deltas[Math.floor((deltas.length - 1) / 2)];   // median
      }
      if (offset < PAD) { offset = PAD; }   // keep the column top on-canvas
    }
    for (const p of packed) {
      const yy = p.y + offset;
      pos.set(p.item.id, {
        x, y: yy, h: p.h, rank: r, kind: p.item.kind,
      });
      maxX = Math.max(maxX, x + NODE_W);
      maxY = Math.max(maxY, yy + p.h);
    }
  });

  return {
    nodes, pos, edgeChain,
    width: maxX + PAD,
    height: maxY + PAD,
  };
}

// Snap a world drop point to a grid pin {col, y} for a newly-dropped
// stage: nearest column, then clamped to within 5 columns / 5 rows of the
// existing graph's extent (read from the previous layout `seedPos`) so a
// stray far-away drop can't fling the node off into empty space. A
// null/empty `seedPos` (dropping onto an empty canvas) clamps around the
// origin instead.
export function worldToPin(pos, seedPos) {
  const cl = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const COL_PITCH = NODE_W + COL_GAP;
  let maxRank = 0, maxY = -Infinity, any = false;
  if (seedPos) {
    seedPos.forEach((p) => {
      if (p.kind && p.kind !== 'real') { return; }
      any = true;
      maxRank = Math.max(maxRank, p.rank || 0);
      maxY = Math.max(maxY, p.y + (p.h || 0));
    });
  }
  if (!any) { maxY = PAD; }
  const col = cl(Math.round((pos.x - PAD) / COL_PITCH), 0, maxRank + 5);
  // Vertical: keep the box on-canvas (>= PAD, the top margin) and not flung
  // far below the graph. The lower bound is an ABSOLUTE top margin, NOT
  // relative to the current topmost node -- so a box can always be dragged
  // back up to the top even after the rest of the graph was pushed down
  // (the old `minY - 5*ROW_PITCH` floor rose with the graph and trapped it).
  const y = cl(pos.y, PAD, maxY + 5 * ROW_PITCH);
  return { col, y };
}

// Nudge a dragged box's Y so it doesn't overlap the boxes already sharing
// its column -- moving ONLY the dragged box (the others stay put). Returns
// the Y nearest `desired` whose [y, y+h] span clears every occupied span
// (each padded by OVERLAP_GAP). `occupied` is [{y,h}] of the column's other
// real nodes. When `desired` already fits (a gap, or empty space above /
// below the column), it is returned unchanged -- INCLUDING space above the
// content top, so a box can be dragged up past the topmost stage (there is
// no PAD floor here; the caller places boxes wherever the cursor released).
function deOverlapY(desired, h, occupied) {
  const spans = occupied
    .map((o) => [o.y - OVERLAP_GAP, o.y + o.h + OVERLAP_GAP])
    .sort((a, b) => a[0] - b[0]);
  const hits = (y) => spans.some(([a, b]) => y < b && y + h > a);
  if (!hits(desired)) { return desired; }
  // Candidate slots: just above / just below each occupied span. Pick the
  // clear one nearest to `desired` (below the lowest span always clears,
  // and above the highest is allowed even into negative Y so an upward
  // reorder past the top box isn't blocked).
  let best = desired, bestD = Infinity, found = false;
  for (const [a, b] of spans) {
    for (const c of [a - h, b]) {
      if (hits(c)) { continue; }
      const d = Math.abs(c - desired);
      if (d < bestD) { bestD = d; best = c; found = true; }
    }
  }
  return found ? best : desired;
}

// Stable per-edge key: matches the buffer-status payload's
// {from,from_port,to,to_port} so the overlay can join stats to paths.
export function edgeKey(e) {
  return `${e.from}#${e.from_port}>${e.to}#${e.to_port}`;
}

// Build the SVG path d-attribute for one edge by chaining cubic Bezier
// segments through the edge's waypoints (source port, each dummy node
// centre, destination port). Horizontal control tangents at every
// waypoint produce a smooth, dagre-style spline. Returns { d, mid }
// where mid is a label anchor near the edge's visual middle.
function buildEdgePath(L, e) {
  const sp = L.pos.get(e.from);
  const dp = L.pos.get(e.to);
  if (!sp || !dp) { return null; }
  const pts = [];
  pts.push({ x: sp.x + NODE_W, y: portY(sp, e.from_port) });
  const chain = L.edgeChain.get(e) || [];
  for (const w of chain) {
    const wp = L.pos.get(w.id);
    if (!wp) { continue; }
    pts.push({ x: wp.x + NODE_W / 2, y: wp.y + wp.h / 2 });
  }
  pts.push({ x: dp.x, y: portY(dp, e.to_port) });
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const dx = Math.max(20, (p1.x - p0.x) * 0.5);
    d += ` C ${p0.x + dx} ${p0.y}, ${p1.x - dx} ${p1.y}, `
       + `${p1.x} ${p1.y}`;
  }
  // Label anchor: midpoint of the two central waypoints.
  const a = Math.floor((pts.length - 1) / 2);
  const b = Math.ceil((pts.length - 1) / 2);
  const mid = { x: (pts[a].x + pts[b].x) / 2,
                y: (pts[a].y + pts[b].y) / 2 };
  return { d, mid };
}

function portY(p, idx) { return p.y + PORT_TOP + idx * PORT_GAP; }

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// Build and return a container element (<div>) holding the graph SVG
// plus a floating zoom/center control cluster. opts:
//   { selected, onSelect, view }  (view: shared {k,cx,cy}, mutated)
// Composer (editor) extras, all optional and gated on `editable`:
//   editable        true => render port/edge edit affordances
//   pending         {from, from_port} armed output port (rubber-band)
//   selectedEdge    edgeKey() of the highlighted edge
//   onPortClick(kind, stageId, idx)   kind 'in' | 'out'
//   onAddInput(stageId)               the dashed "+ add input" stub
//   onPortDisconnect(stageId, iport)  the per-input × badge
//   onEdgeSelect(key | null)          edge click / empty-canvas click
export function renderGraph(graph, opts = {}) {
  const { selected, onSelect } = opts;
  const editable = !!opts.editable;
  const pending = opts.pending || null;
  // Beat type of the armed output port (for dimming incompatible
  // inputs while a connection is in progress); null/'any' = no filter.
  const pendingType = opts.pendingType || null;
  // Payload tags of the armed output (finer constraint on top of the
  // beat type; array or null). Empty/absent => no tag filter.
  const pendingTags = opts.pendingTags || null;
  const selectedEdge = opts.selectedEdge || null;
  const view = opts.view || {};

  // Input ports are drawn from the stage spec (named, indexed slots),
  // not just the wired ones: a 3-iport stage shows 3 slots up front.
  // `opts.inputPorts(node)` returns the slot descriptors [{name,type,
  // doc}]; default to the live wired iports when no resolver is given.
  // The "+ add input" stub is legacy (untyped stages with no spec):
  // `opts.showAddInput(node)` gates it; spec-driven stages suppress it.
  const inputPortsOf = opts.inputPorts || ((n) => n.iports);
  const addStubOf = (n) => editable && !!opts.onAddInput
      && (opts.showAddInput ? opts.showAddInput(n) : true);
  const metrics = {
    inCount: (n) => inputPortsOf(n).length,
    addStub: addStubOf,
  };
  // Previous layout positions (id -> {x,y,...}) drive placement-aware
  // re-layout + the reflow animation; onLayout hands the fresh positions
  // back to the caller to seed the next render.
  const seedPos = opts.seedPos || null;
  // User-placed drop pins (id -> {col, y}); honour the chosen grid cell.
  const pins = opts.pins || null;
  // Auto-arrange requests a FRESH tidy layout: ignore the seed ordering
  // and the pins for positioning (crossing-min from scratch). The reflow
  // animation below still glides from `seedPos`, so the tidy-up is visible.
  const fresh = !!opts.freshLayout;
  const L = computeLayout(graph, metrics, fresh ? null : seedPos,
                          fresh ? null : pins);
  if (opts.onLayout) { opts.onLayout(L.pos); }
  const contentW = Math.max(L.width, 1);
  const contentH = Math.max(L.height, 1);

  // Collected during node/edge creation to drive the reflow animation.
  const animNodes = [];   // {g, id, toX, toY}
  const animEdges = [];   // {path, e}

  // Which inputs are wired (for the × badge): "stageId#iport".
  const connectedIn = new Set();
  for (const e of graph.edges) { connectedIn.add(`${e.to}#${e.to_port}`); }

  const svg = svgEl('svg', { class: 'graph' + (editable ? ' editable' : '')
                              + (pending ? ' arming' : '') });

  // Edges underneath nodes. Each long edge has its own routing
  // dummies inserted by computeLayout(); buildEdgePath() splines the
  // curve through every dummy's centre so it can never hide behind a
  // real node and never coalesce with another edge unless the
  // crossing-min pass left them collinear.
  const eg = svgEl('g');
  svg.append(eg);
  // Buffer-utilization label layer (populated lazily by
  // applyBufferStats; sits above the edges, below the nodes so node
  // boxes still occlude labels that land under them).
  const utilG = svgEl('g', { class: 'gutil' });
  for (const e of graph.edges) {
    const built = buildEdgePath(L, e);
    if (!built) { continue; }
    const key = edgeKey(e);
    const gedge = svgEl('path', {
      class: 'gedge' + (e.type === 'any' ? ' untyped' : '')
             + (key === selectedEdge ? ' selected' : ''),
      d: built.d, 'data-ekey': key,
    });
    eg.append(gedge);
    animEdges.push({ path: gedge, e });
    // When editable, a fat invisible hit-path on top makes the thin
    // spline easy to click to select (then Delete removes it).
    if (editable && opts.onEdgeSelect) {
      const hit = svgEl('path', { class: 'gedge-hit', d: built.d });
      hit.addEventListener('click', (ev) => {
        ev.stopPropagation();
        opts.onEdgeSelect(key);
      });
      eg.append(hit);
    }
    // Empty label placed at the edge's midpoint; stays invisible
    // (no text) until applyBufferStats fills it in.
    const lbl = svgEl('text', {
      class: 'edge-util', 'data-ekey': key,
      x: built.mid.x, y: built.mid.y - 3, 'text-anchor': 'middle',
    });
    utilG.append(lbl);
  }
  svg.append(utilG);

  // Re-spline every edge touching `id` (and reposition its util label) in
  // place -- used during a node drag so the wires follow the box without
  // a whole-graph re-render.
  const resplineNode = (id) => {
    for (const ae of animEdges) {
      if (ae.e.from !== id && ae.e.to !== id) { continue; }
      const built = buildEdgePath(L, ae.e);
      if (!built) { continue; }
      ae.path.setAttribute('d', built.d);
      const lbl = utilG.querySelector(
        `.edge-util[data-ekey="${edgeKey(ae.e)}"]`);
      if (lbl) {
        lbl.setAttribute('x', built.mid.x);
        lbl.setAttribute('y', built.mid.y - 3);
      }
    }
  };

  let moved = false;   // set during a pan so the trailing click is ignored

  // Nodes.
  for (const n of graph.nodes) {
    const p = L.pos.get(n.id);
    const g = svgEl('g', {
      class: 'gnode' + (n.id === selected ? ' selected' : '')
             + (n.config_error ? ' needs-config' : ''),
      'data-id': n.id,
      transform: `translate(${p.x}, ${p.y})`,
    });
    animNodes.push({ g, id: n.id, toX: p.x, toY: p.y });
    if (n.config_error) {
      const t = svgEl('title');
      t.textContent = 'Needs configuration: ' + n.config_error;
      g.append(t);
    }
    // Drag-to-move sets this so the release's trailing click doesn't also
    // fire selection.
    let suppressClick = false;
    g.addEventListener('click', () => {
      if (moved || suppressClick) { suppressClick = false; return; }
      if (onSelect) { onSelect(n.id); }
    });
    // Right-click: a context menu (rename / delete) anchored at the cursor.
    if (opts.onNodeContext) {
      g.addEventListener('contextmenu', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        opts.onNodeContext(n.id, ev.clientX, ev.clientY);
      });
    }

    // Drag the box to reposition it. On release it snaps to the nearest
    // column (a minimal correction) and persists as a pin via
    // opts.onNodeMove so a later re-layout keeps the spot -- without
    // relaying-out (moving) the rest of the graph.
    if (opts.onNodeMove) {
      let nDrag = false, nCap = false, nMoved = false;
      let sX = 0, sY = 0, grabDX = 0, grabDY = 0;
      g.addEventListener('pointerdown', (e) => {
        if (e.button !== undefined && e.button !== 0) { return; }
        e.stopPropagation();      // don't let the SVG pan handler start
        moved = false;            // fresh interaction: re-enable click
        suppressClick = false;
        nDrag = true; nMoved = false;
        sX = e.clientX; sY = e.clientY;
        const w = clientToWorld(e);
        const cur = L.pos.get(n.id);
        grabDX = w.x - cur.x; grabDY = w.y - cur.y;
      });
      g.addEventListener('pointermove', (e) => {
        if (!nDrag) { return; }
        if (!nMoved) {
          if (Math.abs(e.clientX - sX) + Math.abs(e.clientY - sY) <= 3) {
            return;
          }
          nMoved = true;
          g.classList.add('dragging');
          svg.appendChild(g);     // raise above the others (paint = DOM order)
          // Capture AFTER re-parenting: appendChild moves the node in the
          // DOM, which RELEASES an already-set pointer capture -- then a
          // fast drag (whose cursor outruns the box and leaves it) stops
          // getting pointermove/up and never finalizes ("lags + sticks").
          try { g.setPointerCapture(e.pointerId); nCap = true; } catch (x) {}
        }
        const w = clientToWorld(e);
        const cur = L.pos.get(n.id);
        cur.x = w.x - grabDX; cur.y = w.y - grabDY;
        g.setAttribute('transform', `translate(${cur.x}, ${cur.y})`);
        resplineNode(n.id);
      });
      const endDrag = (e) => {
        if (!nDrag) { return; }
        nDrag = false;
        if (nCap) {
          try { g.releasePointerCapture(e.pointerId); } catch (x) {}
          nCap = false;
        }
        g.classList.remove('dragging');
        if (!nMoved) { return; }    // a press without a drag -> let it click
        suppressClick = true;
        // Snap to the nearest column (a small adjustment) and keep the drop
        // Y. Clamp against the current layout's extent, not the seed, so a
        // drag on a fresh graph isn't mis-clamped.
        const cur = L.pos.get(n.id);
        // worldToPin is used ONLY for the column snap; the vertical position
        // is the raw released Y (cur.y) so there is no top-margin floor --
        // the box can be placed anywhere the cursor let go, including above
        // the topmost stage. De-overlap then keeps it off its column-mates.
        const pin = worldToPin({ x: cur.x, y: cur.y }, L.pos);
        const gx = PAD + pin.col * (NODE_W + COL_GAP);
        const occ = [];
        L.pos.forEach((q, qid) => {
          if (qid === n.id || (q.kind && q.kind !== 'real')) { return; }
          if (Math.abs(q.x - gx) < 1) { occ.push(q); }
        });
        const y = deOverlapY(cur.y, cur.h, occ);
        cur.x = gx; cur.y = y;
        g.setAttribute('transform', `translate(${gx}, ${y})`);
        resplineNode(n.id);
        opts.onNodeMove(n.id, { col: pin.col, y }, { x: gx, y });
      };
      g.addEventListener('pointerup', endDrag);
      g.addEventListener('pointercancel', endDrag);
    }

    g.append(svgEl('rect', {
      class: 'box', x: 0, y: 0, width: NODE_W, height: p.h, rx: 6,
    }));
    const title = svgEl('text', { class: 'title', x: 10, y: 17 });
    title.textContent = truncate(n.id, 22);
    g.append(title);
    const sub = svgEl('text', { class: 'subtitle', x: 10, y: 30 });
    sub.textContent = truncate(n.type, 24);
    g.append(sub);

    // Input slots come from the stage spec (max'd with the wired
    // count): every declared input is shown with its semantic name, so
    // the index→meaning mapping is visible. Iports are positional and
    // may be gapped: a slot is "connected" iff an edge targets it
    // (`connectedIn`, built from the edge list), with no fill-in-order
    // constraint, so any slot is connectable in any order.
    const inSlots = inputPortsOf(n);
    inSlots.forEach((pt, i) => {
      const y = PORT_TOP + i * PORT_GAP;
      const connected = connectedIn.has(`${n.id}#${i}`);
      // While arming, an existing input that can't accept the armed
      // output is dimmed (CSS) -- a visible cue before the click (also
      // refused upstream). Two reasons: the beat types disagree, or the
      // payload tags (finer constraint) don't intersect.
      const typeBad = !!pendingType && pendingType !== 'any'
          && pt.type && pt.type !== 'any' && pt.type !== pendingType;
      const slotTags = (pt.tags || '').split(',')
          .map((x) => x.trim()).filter(Boolean);
      const tagBad = !!pendingTags && pendingTags.length && slotTags.length
          && !pendingTags.some((x) => slotTags.includes(x));
      const incompatible = typeBad || tagBad;
      const pg = svgEl('g', { class: 'port-grp in'
          + (pt.type === 'any' ? ' untyped' : '')
          + (connected ? ' connected' : ' empty')
          + (incompatible ? ' incompatible' : '') });
      pg.append(svgEl('circle', { class: 'port' + (connected ? '' : ' empty'),
        cx: 0, cy: y, r: PORT_R }));
      const lbl = svgEl('text', { class: 'portlabel', x: 8, y: y + 3 });
      // Show the port's semantic name (its index meaning); fall back to
      // the beat type for untyped/unnamed legacy ports.
      lbl.textContent = truncate(pt.name || shortType(pt.type), 12);
      pg.append(lbl);
      // Tooltip carries the full name + beat type + doc.
      const tip = svgEl('title');
      const ptDoc = pt.name ? tOr('port.' + n.type + '.' + pt.name, pt.doc)
                            : pt.doc;
      tip.textContent = (pt.name ? pt.name + '  ' : '')
          + '(' + shortType(pt.type) + ')'
          + (pt.tags ? '\ntags: ' + pt.tags : '')
          + (ptDoc ? '\n' + ptDoc : '');
      pg.append(tip);
      if (editable) {
        const hit = svgEl('circle', { class: 'porthit', cx: 0, cy: y, r: 9 });
        hit.addEventListener('pointerdown', (ev) => ev.stopPropagation());
        hit.addEventListener('click', (ev) => {
          ev.stopPropagation();
          if (opts.onPortClick) { opts.onPortClick('in', n.id, i); }
        });
        pg.append(hit);
        if (connected && opts.onPortDisconnect) {
          const bx = -13;
          const bg = svgEl('g', { class: 'port-x' });
          bg.append(svgEl('circle', { class: 'port-x-bg', cx: bx, cy: y,
            r: 6 }));
          const xg = svgEl('text', { class: 'port-x-glyph', x: bx,
            y: y + 3, 'text-anchor': 'middle' });
          xg.textContent = '×';
          bg.append(xg);
          bg.addEventListener('pointerdown', (ev) => ev.stopPropagation());
          bg.addEventListener('click', (ev) => {
            ev.stopPropagation();
            opts.onPortDisconnect(n.id, i);
          });
          pg.append(bg);
        }
      }
      g.append(pg);
    });
    n.oports.forEach((pt, i) => {
      const y = PORT_TOP + i * PORT_GAP;
      const armed = pending && pending.from === n.id
                 && pending.from_port === i;
      const pg = svgEl('g', { class: 'port-grp out'
          + (pt.type === 'any' ? ' untyped' : '')
          + (armed ? ' armed' : '') });
      pg.append(svgEl('circle', { class: 'port', cx: NODE_W, cy: y,
        r: PORT_R }));
      const lbl = svgEl('text', { class: 'portlabel', x: NODE_W - 8,
        y: y + 3, 'text-anchor': 'end' });
      lbl.textContent = truncate(shortType(pt.type), 12);
      pg.append(lbl);
      const otip = svgEl('title');
      otip.textContent = (pt.name ? pt.name + '  ' : '')
          + '(' + shortType(pt.type) + ')'
          + (pt.tags ? '\ntags: ' + pt.tags : '');
      pg.append(otip);
      if (editable) {
        const hit = svgEl('circle', { class: 'porthit', cx: NODE_W, cy: y,
          r: 9 });
        hit.addEventListener('pointerdown', (ev) => ev.stopPropagation());
        hit.addEventListener('click', (ev) => {
          ev.stopPropagation();
          if (opts.onPortClick) { opts.onPortClick('out', n.id, i); }
        });
        pg.append(hit);
      }
      g.append(pg);
    });
    // Dashed "+ add input" stub: appends a brand-new input port. Only
    // shown for legacy/untyped stages with no spec to enumerate ports
    // (addStubOf); spec-driven stages render their fixed named slots
    // above instead, so they never grow one-by-one.
    if (addStubOf(n)) {
      const y = PORT_TOP + inSlots.length * PORT_GAP;
      const pg = svgEl('g', { class: 'port-grp add-in' });
      pg.append(svgEl('circle', { class: 'port-add', cx: 0, cy: y,
        r: PORT_R }));
      const plus = svgEl('text', { class: 'port-add-glyph', x: 0,
        y: y + 3.5, 'text-anchor': 'middle' });
      plus.textContent = '+';
      pg.append(plus);
      const lbl = svgEl('text', { class: 'portlabel add', x: 8, y: y + 3 });
      lbl.textContent = 'add input';
      pg.append(lbl);
      const hit = svgEl('circle', { class: 'porthit', cx: 0, cy: y, r: 9 });
      hit.addEventListener('pointerdown', (ev) => ev.stopPropagation());
      hit.addEventListener('click', (ev) => {
        ev.stopPropagation();
        opts.onAddInput(n.id);
      });
      pg.append(hit);
      g.append(pg);
    }
    svg.append(g);
  }

  // --- view (viewBox) management ----------------------------------
  // Fit scale: show the whole graph with padding (never enlarge a tiny
  // graph past FIT_MAX_K, never shrink past MIN_K).
  const fitScale = (W, H) =>
    clamp(Math.min((W - 2 * PAD) / contentW,
                   (H - 2 * PAD) / contentH) || 1, MIN_K, FIT_MAX_K);

  function applyView() {
    const W = svg.clientWidth, H = svg.clientHeight;
    if (!W || !H) { return; }
    if (!view.k) {                       // unset -> fit + center
      view.k  = fitScale(W, H);
      view.cx = contentW / 2;
      view.cy = contentH / 2;
    }
    const vw = W / view.k, vh = H / view.k;
    svg.setAttribute('viewBox',
      `${view.cx - vw / 2} ${view.cy - vh / 2} ${vw} ${vh}`);
  }

  const setZoom = (k) => { view.k = clamp(k, MIN_K, MAX_K); applyView(); };
  const zoomBy  = (f) => setZoom((view.k || 1) * f);
  const center  = () => { view.cx = contentW / 2; view.cy = contentH / 2;
                          applyView(); };
  const fit     = () => { view.k = 0; applyView(); };   // 0 -> refit+center

  const ctl = (label, title, fn) =>
    el('button', { class: 'graph-ctl', title,
      onclick: (e) => { e.stopPropagation(); fn(); } }, label);
  const controls = el('div', { class: 'graph-controls' },
    ctl('−', 'Zoom out', () => zoomBy(1 / 1.2)),
    ctl('+', 'Zoom in', () => zoomBy(1.2)),
    ctl('1:1', 'Actual size', () => setZoom(1)),
    ctl('Fit', 'Fit whole pipeline', fit),
    ctl('⊙', 'Center', center));
  // Auto-arrange: re-layout the graph tidily (drops manual placements).
  // Sits with the view controls; only shown when the caller wires it.
  if (opts.onAutoArrange) {
    controls.append(ctl('⌗', tOr('pl.auto_arrange', 'Auto-arrange'),
      () => opts.onAutoArrange()));
  }

  const container = el('div', { class: 'graph-view' }, svg, controls);
  // Expose the screen->world mapper so the composer can turn a stage
  // drop point into a grid pin. Takes anything with {clientX, clientY}.
  container.clientToWorld = (evt) => clientToWorld(evt);

  // Drag to pan. The pointer is captured only once a real drag begins
  // (movement past a threshold). Capturing on pointerdown would
  // retarget the follow-up click to the <svg> and break node selection.
  let dragging = false, captured = false;
  let startX = 0, startY = 0, lastX = 0, lastY = 0;
  svg.addEventListener('pointerdown', (e) => {
    if (e.button !== undefined && e.button !== 0) { return; }
    dragging = true; captured = false; moved = false;
    startX = lastX = e.clientX; startY = lastY = e.clientY;
  });
  svg.addEventListener('pointermove', (e) => {
    if (!dragging) { return; }
    if (!moved) {
      if (Math.abs(e.clientX - startX)
          + Math.abs(e.clientY - startY) <= 3) { return; }
      moved = true;
      try { svg.setPointerCapture(e.pointerId); } catch (x) {}
      captured = true;
      container.classList.add('panning');
    }
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    const k = view.k || 1;
    view.cx -= dx / k; view.cy -= dy / k;
    applyView();
  });
  const endPan = (e) => {
    if (!dragging) { return; }
    dragging = false;
    if (captured) {
      try { svg.releasePointerCapture(e.pointerId); } catch (x) {}
      captured = false;
    }
    container.classList.remove('panning');
  };
  svg.addEventListener('pointerup', endPan);
  svg.addEventListener('pointercancel', endPan);

  // Wheel to zoom, anchored at the cursor.
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const W = svg.clientWidth, H = svg.clientHeight;
    if (!W || !H) { return; }
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const k0 = view.k || 1;
    const wx = (view.cx - (W / k0) / 2) + px / k0;
    const wy = (view.cy - (H / k0) / 2) + py / k0;
    const k = clamp(k0 * (e.deltaY < 0 ? 1.1 : 1 / 1.1), MIN_K, MAX_K);
    view.k = k;
    view.cx = wx - px / k + (W / k) / 2;
    view.cy = wy - py / k + (H / k) / 2;
    applyView();
  }, { passive: false });

  // Map a pointer event to "world" (content) coordinates -- same
  // inverse-viewBox math the wheel handler uses. Needed for the
  // connect rubber-band so the pending line tracks the cursor.
  function clientToWorld(evt) {
    const W = svg.clientWidth, H = svg.clientHeight;
    const k = view.k || 1;
    const rect = svg.getBoundingClientRect();
    const px = evt.clientX - rect.left, py = evt.clientY - rect.top;
    return {
      x: (view.cx - (W / k) / 2) + px / k,
      y: (view.cy - (H / k) / 2) + py / k,
    };
  }

  // Rubber-band: while an output port is armed, draw a live spline
  // from it to the cursor. Updated on pointermove without re-rendering
  // so it stays smooth; pointer-events:none (CSS) keeps it click-through.
  if (editable && pending) {
    const sp = L.pos.get(pending.from);
    if (sp) {
      const ox = sp.x + NODE_W;
      const oy = portY(sp, pending.from_port);
      const pendingPath = svgEl('path', { class: 'pending-edge' });
      svg.append(pendingPath);
      const draw = (wx, wy) => {
        const dx = Math.max(20, (wx - ox) * 0.5);
        pendingPath.setAttribute('d',
          `M ${ox} ${oy} C ${ox + dx} ${oy}, ${wx - dx} ${wy}, `
          + `${wx} ${wy}`);
      };
      draw(ox + 1, oy);
      svg.addEventListener('pointermove', (e) => {
        const w = clientToWorld(e);
        draw(w.x, w.y);
      });
    }
  }

  // Click on empty canvas: deselect any edge and cancel a pending
  // connect. A trailing click after a pan (moved) is ignored.
  if (editable) {
    svg.addEventListener('click', (e) => {
      if (e.target !== svg || moved) { return; }
      if (opts.onEdgeSelect) { opts.onEdgeSelect(null); }
      if (opts.onBackgroundClick) { opts.onBackgroundClick(); }
    });
  }

  // Re-fit/apply once the element is sized (and on resize).
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => applyView()).observe(svg);
  }
  applyView();
  requestAnimationFrame(applyView);

  // --- reflow animation -------------------------------------------
  // When re-laying-out an existing graph (a seed is present) glide every
  // node from its previous position to its new one over ~0.75s, re-
  // splining the edges each frame, so the user can follow how the stages
  // moved. Nodes that didn't move (and brand-new nodes, which have no
  // seed) simply sit at their final spot. The whole thing runs in "world"
  // coordinates inside the viewBox, so pan/zoom is untouched.
  if (seedPos && seedPos.size) {
    const from = new Map();
    L.pos.forEach((p, id) => from.set(id, { x: p.x, y: p.y }));  // default
    let moved = false;
    for (const a of animNodes) {
      const s = seedPos.get(a.id);
      if (s && (Math.abs(s.x - a.toX) > 0.5
                || Math.abs(s.y - a.toY) > 0.5)) {
        from.set(a.id, { x: s.x, y: s.y });
        moved = true;
      }
    }
    if (moved) {
      const DUR = 750;
      const ease = (t) =>
        (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
      const applyFrame = (e) => {
        for (const a of animNodes) {
          const f = from.get(a.id);
          a.g.setAttribute('transform', `translate(${f.x + (a.toX - f.x)
            * e}, ${f.y + (a.toY - f.y) * e})`);
        }
        const interp = new Map();
        L.pos.forEach((p, id) => {
          const f = from.get(id);
          interp.set(id, (f && (f.x !== p.x || f.y !== p.y))
            ? { ...p, x: f.x + (p.x - f.x) * e, y: f.y + (p.y - f.y) * e }
            : p);
        });
        const IL = { pos: interp, edgeChain: L.edgeChain };
        for (const ae of animEdges) {
          const built = buildEdgePath(IL, ae.e);
          if (built) { ae.path.setAttribute('d', built.d); }
        }
      };
      // Paint the FIRST frame at the old positions synchronously (before
      // the container is attached), so there's no flash of the final
      // layout before the glide starts.
      applyFrame(0);
      let t0 = null;
      const step = (ts) => {
        if (!svg.isConnected) { return; }   // superseded by a newer render
        if (t0 === null) { t0 = ts; }
        const t = Math.min(1, (ts - t0) / DUR);
        applyFrame(ease(t));
        if (t < 1) { requestAnimationFrame(step); }
      };
      requestAnimationFrame(step);
    }
  }

  return container;
}

const UTIL_CLASSES = ['util-idle', 'util-ok', 'util-high', 'util-full',
                      'util-drop'];

// Bucket a fill ratio (backlog / capacity) into a severity class. A
// DropOldest buffer that actually dropped is flagged regardless of fill
// (a drop is a lost beat, the thing the operator most needs to see).
function utilLevel(ratio, dropped) {
  if (dropped > 0) { return 'util-drop'; }
  if (!(ratio > 0)) { return 'util-idle'; }
  if (ratio >= 0.9) { return 'util-full'; }
  if (ratio >= 0.5) { return 'util-high'; }
  return 'util-ok';
}

// Patch the graph's buffer-utilization overlay in place (no re-render,
// so pan/zoom and selection are untouched). `root` is the element
// renderGraph returned (or any ancestor of the SVG); `edges` is the
// buffer-status payload's edges array. Pass [] / null to clear.
//   * each .gedge path is tinted by its consumer backlog / capacity,
//   * each .edge-util label shows "backlog/capacity" (+ "⚠N" drops).
// A label with no matching stat is blanked so a stale overlay can't
// linger after the topology changes.
export function applyBufferStats(root, edges) {
  if (!root) { return; }
  const byKey = new Map();
  for (const e of (edges || [])) {
    byKey.set(`${e.from}#${e.from_port}>${e.to}#${e.to_port}`, e);
  }
  root.querySelectorAll('.gedge[data-ekey]').forEach((path) => {
    const s = byKey.get(path.getAttribute('data-ekey'));
    path.classList.remove(...UTIL_CLASSES);
    if (s && s.capacity > 0) {
      path.classList.add(utilLevel(s.backlog / s.capacity, s.dropped || 0));
    }
  });
  root.querySelectorAll('.edge-util[data-ekey]').forEach((lbl) => {
    const s = byKey.get(lbl.getAttribute('data-ekey'));
    lbl.classList.remove(...UTIL_CLASSES);
    if (!s) { lbl.textContent = ''; return; }
    const cap = Number(s.capacity) || 0;
    const bl = Number(s.backlog) || 0;
    let txt = cap > 0 ? `${bl}/${cap}` : String(bl);
    if (s.dropped > 0) { txt += ` ⚠${s.dropped}`; }
    lbl.textContent = txt;
    if (cap > 0) { lbl.classList.add(utilLevel(bl / cap, s.dropped || 0)); }
  });
}
