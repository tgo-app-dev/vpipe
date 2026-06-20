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
const HEADER_H  = 38;
const PORT_TOP  = HEADER_H + 8;
const PORT_GAP  = 18;
const PORT_R    = 4.5;
const PAD       = 28;
const DUMMY_H   = 10;    // routing slot height for through-going edges
const DUMMY_GAP = 6;     // tighter spacing when an adjacent item is a dummy
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
function computeLayout(graph, metrics) {
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
    let r = 0;
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
  for (let it = 0; it < SWEEPS; it++) {
    sweep(true);
    sweep(false);
  }

  // 7. Y positioning. Real nodes get their full nodeHeight + ROW_GAP;
  //    dummies get a small lane slot with tighter spacing so the
  //    overall layout doesn't blow up vertically for long edges.
  const pos = new Map();
  let maxX = 0;
  let maxY = 0;
  layers.forEach((layer, r) => {
    const x = PAD + r * (NODE_W + COL_GAP);
    let y = PAD;
    for (let i = 0; i < layer.length; i++) {
      const item = layer[i];
      const rn = item.kind === 'real' ? nodes.get(item.id) : null;
      const h = item.kind === 'real'
          ? nodeHeight(metrics.inCount(rn), rn.oports.length,
                       metrics.addStub(rn))
          : DUMMY_H;
      pos.set(item.id, {
        x, y, h, rank: r, kind: item.kind,
      });
      const nxt = layer[i + 1];
      const dummyAdj = item.kind === 'dummy'
                    || (nxt && nxt.kind === 'dummy');
      y += h + (dummyAdj ? DUMMY_GAP : ROW_GAP);
      maxX = Math.max(maxX, x + NODE_W);
      maxY = Math.max(maxY, y);
    }
  });

  return {
    nodes, pos, edgeChain,
    width: maxX + PAD,
    height: maxY + PAD,
  };
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
  const L = computeLayout(graph, metrics);
  const contentW = Math.max(L.width, 1);
  const contentH = Math.max(L.height, 1);

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
    eg.append(svgEl('path', {
      class: 'gedge' + (e.type === 'any' ? ' untyped' : '')
             + (key === selectedEdge ? ' selected' : ''),
      d: built.d, 'data-ekey': key,
    }));
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

  let moved = false;   // set during a pan so the trailing click is ignored

  // Nodes.
  for (const n of graph.nodes) {
    const p = L.pos.get(n.id);
    const g = svgEl('g', {
      class: 'gnode' + (n.id === selected ? ' selected' : '')
             + (n.config_error ? ' needs-config' : ''),
      transform: `translate(${p.x}, ${p.y})`,
    });
    if (n.config_error) {
      const t = svgEl('title');
      t.textContent = 'Needs configuration: ' + n.config_error;
      g.append(t);
    }
    g.addEventListener('click', () => {
      if (moved) { return; }
      if (onSelect) { onSelect(n.id); }
    });

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
    // the index→meaning mapping is visible. In the dense model the
    // wired inputs are the contiguous prefix [0, liveCount); the slot
    // at liveCount is the next connectable one and slots beyond it are
    // "locked" (inputs fill in order).
    const inSlots = inputPortsOf(n);
    const liveCount = n.iports.length;
    inSlots.forEach((pt, i) => {
      const y = PORT_TOP + i * PORT_GAP;
      const connected = connectedIn.has(`${n.id}#${i}`) || i < liveCount;
      const locked = i > liveCount;
      // While arming, an existing input whose declared type can't
      // accept the armed output's type is dimmed (CSS) -- a visible
      // cue before the click (the click is also refused upstream).
      const incompatible = !!pendingType && pendingType !== 'any'
          && pt.type && pt.type !== 'any' && pt.type !== pendingType;
      const pg = svgEl('g', { class: 'port-grp in'
          + (pt.type === 'any' ? ' untyped' : '')
          + (connected ? ' connected' : ' empty')
          + (locked ? ' locked' : '')
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

  const container = el('div', { class: 'graph-view' }, svg, controls);

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
