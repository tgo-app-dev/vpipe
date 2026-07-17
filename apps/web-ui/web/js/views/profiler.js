// Profiler view: start/stop the session's per-worker event capture and
// show the captured timeline.
//
// Layout: a toolbar (Start / Stop / Refresh / Fit + a max-events input
// and a status readout), a canvas timeline with one horizontal lane per
// worker thread (+ the overflow lane), and an inspector panel below
// showing the selected event's details.
//
// The events are instantaneous markers ({ns, type, stage_gvid, value}),
// drawn as colored ticks per lane and positioned on a shared time axis.
// The timeline pans (drag / Shift+wheel) and zooms (wheel) horizontally;
// lanes shrink vertically to always fit. A canvas (not SVG/DOM) keeps it
// fast for tens of thousands of events.
//
// Data comes from /api/profiler/data, which is the session's
// dump_profiling document: live while capturing, else the snapshot taken
// at Stop (disabling frees the buffers, so the backend retains it).

import { el, clear, toast } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

// Categorical palette: stage gvid picks a hue, stable for a given
// pipeline build.
const PALETTE = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
  '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
  '#9c755f', '#bab0ac', '#1f77b4', '#2ca02c',
];

const AXIS_H   = 24;     // top time-ruler height (px)
const GUTTER_W = 100;    // left lane-label column width (px)

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Adaptive ns formatting: the largest unit where value >= 1.
function fmtNs(ns) {
  if (ns < 1e3) { return ns + ' ns'; }
  if (ns < 1e6) { return (ns / 1e3).toFixed(2) + ' µs'; }
  if (ns < 1e9) { return (ns / 1e6).toFixed(2) + ' ms'; }
  return (ns / 1e9).toFixed(3) + ' s';
}

// Per-block throughput for the LLM-lane activities, derived from the
// event payload (a count: tokens for prefill/decode, frames for vision,
// input samples for audio) and the block duration. Returns null when no
// sensible rate applies. text-prefill is the headline: tokens / seconds.
function rateLabel(stageId, count, durNs) {
  if (!count || durNs <= 0) { return null; }
  const perSec = count / (durNs / 1e9);
  if (stageId === 'text-prefill' || stageId === 'text-decode') {
    return perSec.toFixed(1) + ' tok/s';
  }
  if (stageId === 'vision-tower') {
    return perSec.toFixed(2) + ' frame/s';
  }
  if (stageId === 'audio-encoder') {
    // count is input samples @ 16 kHz -> audio-seconds per wall-second.
    return (perSec / 16000).toFixed(1) + '× RT';
  }
  return null;
}

// Absolute wall-clock from the realtime anchor + a ns delta.
function fmtAbs(anchorRealtimeNs, ns) {
  if (!anchorRealtimeNs) { return '—'; }
  const ms = (anchorRealtimeNs + ns) / 1e6;
  const d = new Date(ms);
  const frac = String(Math.floor(ms % 1000)).padStart(3, '0');
  return d.toLocaleString() + '.' + frac;
}

// "Nice" tick step (1/2/5 * 10^k) for ~target ticks over a ns span.
function pickTickStep(spanNs, target) {
  if (spanNs <= 0 || target <= 0) { return 1; }
  let rough = spanNs / target;
  let mag = 1;
  while (rough >= 10) { rough /= 10; mag *= 10; }
  const pick = rough <= 1 ? 1 : rough <= 2 ? 2 : rough <= 5 ? 5 : 10;
  return Math.max(1, pick * mag);
}

// First index in a t0-sorted item array with item.t0 >= target.
function lowerBoundT0(items, ns) {
  let lo = 0, hi = items.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (items[m].t0 < ns) { lo = m + 1; } else { hi = m; }
  }
  return lo;
}

// Fallback names for the runtime's reserved scheduling type ids, used
// when the dump's per-stage event_names table doesn't carry them (it
// only lists session-registered pipelines). Mirrors perf-event.h /
// Stage::perf_event_name. 0xFFFF0000 = schedule, +1 = unschedule.
const RUNTIME_NAMES = { 4294901760: 'schedule', 4294901761: 'unschedule' };

// Turn a ns-sorted event list into drawable items by pairing the
// even-begin / odd-end convention (perf-event.h): an even id is a
// duration BEGIN if a matching odd (id+1) end follows on this lane,
// else a transient POINT; an odd id closes the most recent open begin
// of id-1 (LIFO). Returns items sorted by t0.
//   point: { kind:'point', gvid, type, t0, value }
//   block: { kind:'block', gvid, type, t0, t1, value, endType, endValue }
// A begin that never gets a matching end within the capture becomes an
// 'ongoing' block (open:true) running to `captureEndNs` -- that's a
// worker still inside one resume (e.g. a blocking source stage), which
// would otherwise show as a lone begin-point on an apparently empty
// lane.
function pairItems(events, captureEndNs) {
  const items = [];
  // Key the open-begin stacks by (gvid, beginType), NOT type alone: a
  // multi-producer lane (the overflow + the aux LLM/ANE lanes) receives
  // interleaved begin/end from concurrent sources (e.g. the vision
  // tower and YOLO both predicting on the ANE), and pairing by type
  // alone would splice one source's begin to another's end -- the
  // "single long block overlapping everything" artifact.
  const open = new Map();   // "gvid:beginType" -> stack of open begins
  for (const ev of events) {
    if ((ev.type & 1) === 0) {
      const it = { kind: 'point', gvid: ev.gvid, type: ev.type,
                   t0: ev.ns, value: ev.value };
      items.push(it);
      const key = ev.gvid + ':' + ev.type;
      let st = open.get(key);
      if (!st) { st = []; open.set(key, st); }
      st.push(it);
    } else {
      const st = open.get(ev.gvid + ':' + (ev.type - 1));
      if (st && st.length) {
        const it = st.pop();
        it.kind = 'block';
        it.t1 = ev.ns;
        it.endType = ev.type;
        it.endValue = ev.value;
      } else {
        // Orphan end (begin not captured / dropped) -> a point.
        items.push({ kind: 'point', gvid: ev.gvid, type: ev.type,
                     t0: ev.ns, value: ev.value });
      }
    }
  }
  // Unclosed begins: promote the lone begin-point to an ongoing block
  // that runs to the capture edge (these share storage with the items
  // already pushed, so mutating in place updates the drawable list).
  for (const st of open.values()) {
    for (const it of st) {
      it.kind = 'block';
      it.t1   = Math.max(captureEndNs, it.t0);
      it.open = true;
    }
  }
  items.sort((a, b) => a.t0 - b.t0);
  packRows(items);
  return items;
}

// Assign each block a sub-row within its lane so overlapping blocks
// (concurrent work on a shared lane) stack instead of drawing on top of
// one another. Greedy first-fit over t0-sorted blocks gives the minimum
// number of rows (= peak concurrency). Returns that row count; each
// block gets an `.row`, points stay on row 0 (drawn full lane height).
function packRows(items) {
  const rowEnd = [];           // rowEnd[r] = t1 of the last block in row r
  let maxRow = 0;
  for (const it of items) {
    if (it.kind !== 'block') { it.row = 0; continue; }
    let r = 0;
    while (r < rowEnd.length && rowEnd[r] > it.t0) { r++; }
    if (r === rowEnd.length) { rowEnd.push(it.t1); }
    else { rowEnd[r] = it.t1; }
    it.row = r;
    if (r > maxRow) { maxRow = r; }
  }
  const rows = rowEnd.length || 1;
  items.rows = rows;
  return rows;
}

function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n, d) => (cs.getPropertyValue(n).trim() || d);
  return {
    bg:    g('--bg', '#14161a'),
    bg1:   g('--bg-1', '#1b1e24'),
    bg2:   g('--bg-2', '#21252c'),
    line:  g('--line', '#333a44'),
    fg:    g('--fg', '#d8dde5'),
    fgdim: g('--fg-dim', '#8b94a3'),
    accent:g('--accent', '#5cc8ff'),
  };
}

export function mountProfiler(container) {
  clear(container);

  const S = {
    lanes: [],            // visible lanes (allLanes filtered by hideWorkers)
    allLanes: [],         // every lane from the last ingest (unfiltered)
    hideWorkers: false,   // show only the aux LLM / ANE lanes (save space)
    stages: {},           // gvid -> {id, type, pipeline, names:{type->name}}
    gvidColor: {},        // gvid -> css color
    anchorRealtime: 0,
    t0: 0, t1: 1,         // ns domain (min..max event ns)
    totalEvents: 0,
    totalDropped: 0,
    enabled: false,
    startNs: 0, pxPerNs: 1, fitPx: 1,
    laneH: 28,
    W: 0, H: 0,
    selected: null,       // {laneIdx, itemIdx}
    autoFit: true,        // re-fit on live refresh until the user zooms
    poll: null,
    stopped: false,
  };

  // ---- DOM skeleton -----------------------------------------------
  const maxInput = el('input', { type: 'number', class: 'prof-max',
    value: '65536', min: '1', title: t('prof.max_events') });
  const startBtn = el('button', { class: 'btn primary' }, t('common.start'));
  const stopBtn  = el('button', { class: 'btn', disabled: true },
    t('common.stop'));
  const refreshBtn = el('button', { class: 'btn ghost' }, t('common.refresh'));
  const fitBtn   = el('button', { class: 'btn ghost' }, t('common.fit'));
  const resetBtn = el('button', { class: 'btn ghost',
    title: t('prof.reset_title') }, t('common.reset'));
  const statusEl = el('span', { class: 'prof-status' }, '—');

  // Hide the worker / overflow lanes, keeping only the aux LLM / ANE
  // activity lanes -- saves vertical space when those are all that matters.
  // Sticky across re-mounts (like the inspector height below).
  const HIDE_WK_KEY = 'vpipe_profiler_hide_workers';
  try { S.hideWorkers = localStorage.getItem(HIDE_WK_KEY) === '1'; }
  catch (e) { /* storage unavailable */ }
  const hideWorkersCb = el('input', { type: 'checkbox' });
  hideWorkersCb.checked = S.hideWorkers;
  const hideWorkersLabel = el('label', { class: 'prof-check',
    title: t('prof.hide_workers_title') },
    hideWorkersCb, el('span', {}, t('prof.hide_workers')));

  const head = el('div', { class: 'prof-head' },
    el('span', { class: 'title' }, t('nav.profiler')),
    el('span', { class: 'prof-sep' }),
    el('label', { class: 'prof-maxlabel' }, t('prof.events_per_worker'),
      maxInput),
    startBtn, stopBtn,
    el('span', { class: 'grow' }),
    hideWorkersLabel, refreshBtn, fitBtn, resetBtn, statusEl);

  const canvas = el('canvas', { class: 'prof-canvas' });
  const timeline = el('div', { class: 'prof-timeline' }, canvas);
  const resizer = el('div', { class: 'prof-resizer',
    title: t('prof.resize_title') });
  const inspector = el('div', { class: 'prof-inspector' });

  // Restore a previously-dragged details-panel height (sticky across
  // re-mounts / reloads, like the theme + db local-time prefs).
  const INSP_H_KEY = 'vpipe_profiler_inspector_h';
  let savedInspH = NaN;
  try { savedInspH = parseInt(localStorage.getItem(INSP_H_KEY), 10); }
  catch (e) { /* storage unavailable */ }
  if (Number.isFinite(savedInspH)) {
    inspector.style.height = savedInspH + 'px';
  }

  container.append(el('div', { class: 'profiler' }, head, timeline,
    resizer, inspector));

  const ctx = canvas.getContext('2d');

  // ---- geometry / layout ------------------------------------------
  function layout() {
    const dpr = window.devicePixelRatio || 1;
    const W = timeline.clientWidth;
    const H = timeline.clientHeight;
    if (W <= 0 || H <= 0) { return; }
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    S.W = W; S.H = H;
    const n = Math.max(1, S.lanes.length);
    S.laneH = clamp((H - AXIS_H) / n, 12, 34);
    const span = Math.max(1, S.t1 - S.t0);
    S.fitPx = (W - GUTTER_W) / (span * 1.04);
    if (S.autoFit) { fit(); }
  }

  function fit() {
    const span = Math.max(1, S.t1 - S.t0);
    S.pxPerNs = (S.W - GUTTER_W) / (span * 1.04);
    S.startNs = S.t0 - span * 0.02;
    S.autoFit = true;
    render();
  }

  function clampView() {
    const minPx = S.fitPx / 8;
    const maxPx = Math.max(S.fitPx * 4000, 4);
    S.pxPerNs = clamp(S.pxPerNs, minPx, maxPx);
    const vis = (S.W - GUTTER_W) / S.pxPerNs;
    const lo = S.t0 - vis * 0.5;
    const hi = Math.max(lo, S.t1 - vis * 0.5);
    S.startNs = clamp(S.startNs, lo, hi);
  }

  const colorFor = (gvid) =>
    S.gvidColor[gvid] || PALETTE[gvid % PALETTE.length];

  // ---- render ------------------------------------------------------
  function render() {
    if (S.W <= 0) { return; }
    const C = themeColors();
    const W = S.W, H = S.H, laneH = S.laneH;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Lane bands.
    for (let i = 0; i < S.lanes.length; i++) {
      const y = AXIS_H + i * laneH;
      ctx.fillStyle = (i % 2 === 0) ? C.bg : C.bg1;
      ctx.fillRect(GUTTER_W, y, W - GUTTER_W, laneH);
    }

    // Time gridlines + tick positions (labels drawn later, over ruler).
    const vis = (W - GUTTER_W) / S.pxPerNs;
    const step = pickTickStep(vis, 10);
    const first = Math.ceil(S.startNs / step) * step;
    const ticks = [];
    ctx.strokeStyle = C.line;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    for (let t = first; ; t += step) {
      const x = GUTTER_W + (t - S.startNs) * S.pxPerNs;
      if (x > W) { break; }
      if (x >= GUTTER_W) {
        ctx.moveTo(Math.round(x) + 0.5, AXIS_H);
        ctx.lineTo(Math.round(x) + 0.5, H);
        ticks.push({ x, t });
      }
      if (ticks.length > 2000) { break; }   // safety
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Events: duration blocks (begin/end pairs) + transient points,
    // culled to the visible window.
    ctx.font = '11px ui-monospace, Menlo, monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (let i = 0; i < S.lanes.length; i++) {
      const items = S.lanes[i].items;
      if (!items || !items.length) { continue; }
      const y = AXIS_H + i * laneH;
      // Sub-rows: overlapping blocks are stacked into rows (see
      // packRows) so concurrent ANE jobs don't draw over each other.
      const rows = items.rows || 1;
      const subH = laneH / rows;
      // Start before the first item at/after the left edge so a block
      // beginning off-screen-left but extending into view still draws.
      let j = lowerBoundT0(items, S.startNs);
      while (j > 0) {
        const p = items[j - 1];
        const endNs = (p.t1 !== undefined) ? p.t1 : p.t0;
        if (endNs < S.startNs) { break; }
        j--;
      }
      let lastPointX = -999;
      for (; j < items.length; j++) {
        const it = items[j];
        const x0 = GUTTER_W + (it.t0 - S.startNs) * S.pxPerNs;
        if (x0 > W) { break; }
        ctx.fillStyle = colorFor(it.gvid);
        if (it.kind === 'block') {
          const x1 = GUTTER_W + (it.t1 - S.startNs) * S.pxPerNs;
          if (x1 < GUTTER_W) { continue; }
          const rx = Math.max(GUTTER_W, x0);
          const rw = Math.max(2, x1 - rx);
          const top = y + (it.row || 0) * subH + (rows > 1 ? 1 : 3);
          const bh = subH - (rows > 1 ? 2 : 6);
          if (it.open) {
            // Ongoing (never-closed) interval: faded body so it reads
            // as "still running, not a measured span", with a solid
            // leading edge marking the real begin.
            ctx.globalAlpha = 0.32;
            ctx.fillRect(rx, top, rw, bh);
            ctx.globalAlpha = 1;
            if (x0 >= GUTTER_W) {
              ctx.fillRect(x0, top, Math.min(3, rw), bh);
            }
          } else {
            ctx.fillRect(rx, top, rw, bh);
          }
          if (rw > 38 && bh >= 8) {            // label wide/tall blocks
            const st = S.stages[it.gvid];
            const label = (st ? st.id : ('gvid ' + it.gvid))
                        + (it.open ? t('prof.ongoing') : '');
            ctx.save();
            ctx.beginPath();
            ctx.rect(rx, top, rw, bh);
            ctx.clip();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = 'rgba(0,0,0,.55)';
            ctx.strokeText(label, rx + 3, top + bh / 2);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, rx + 3, top + bh / 2);
            ctx.restore();
          }
        } else {
          const rx = Math.round(x0);
          if (rx < GUTTER_W) { continue; }
          if (rx === lastPointX) { continue; }   // pixel-dedup points
          lastPointX = rx;
          ctx.fillRect(rx - 1, y + 3, 3, laneH - 6);  // full-height tick
        }
      }
    }

    // Selected item highlight (outline; guide line for a point).
    if (S.selected) {
      const lane = S.lanes[S.selected.laneIdx];
      const it = lane && lane.items && lane.items[S.selected.itemIdx];
      if (it) {
        const y = AXIS_H + S.selected.laneIdx * laneH;
        const rows = (lane.items && lane.items.rows) || 1;
        const subH = laneH / rows;
        const x0 = GUTTER_W + (it.t0 - S.startNs) * S.pxPerNs;
        ctx.strokeStyle = C.accent;
        ctx.lineWidth = 2;
        if (it.kind === 'block') {
          const x1 = GUTTER_W + (it.t1 - S.startNs) * S.pxPerNs;
          const rx = Math.max(GUTTER_W, x0);
          const rw = Math.max(2, Math.min(W, x1) - rx);
          const top = y + (it.row || 0) * subH + 1;
          ctx.strokeRect(rx, top, rw, subH - 2);
        } else if (x0 >= GUTTER_W && x0 <= W) {
          ctx.globalAlpha = 0.6;
          ctx.beginPath();
          ctx.moveTo(Math.round(x0) + 0.5, AXIS_H);
          ctx.lineTo(Math.round(x0) + 0.5, H);
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.strokeRect(Math.round(x0) - 3, y + 2, 6, laneH - 4);
        }
        ctx.lineWidth = 1;
      }
    }

    // Left gutter mask + lane labels.
    ctx.fillStyle = C.bg1;
    ctx.fillRect(0, 0, GUTTER_W, H);
    ctx.strokeStyle = C.line;
    ctx.beginPath();
    ctx.moveTo(GUTTER_W + 0.5, 0);
    ctx.lineTo(GUTTER_W + 0.5, H);
    ctx.stroke();
    ctx.font = '11px ui-monospace, Menlo, monospace';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < S.lanes.length; i++) {
      const y = AXIS_H + i * laneH;
      ctx.fillStyle = C.fgdim;
      ctx.textAlign = 'left';
      ctx.fillText(S.lanes[i].label, 8, y + laneH / 2);
    }

    // Top ruler strip + tick labels.
    ctx.fillStyle = C.bg1;
    ctx.fillRect(0, 0, W, AXIS_H);
    ctx.strokeStyle = C.line;
    ctx.beginPath();
    ctx.moveTo(0, AXIS_H + 0.5);
    ctx.lineTo(W, AXIS_H + 0.5);
    ctx.stroke();
    ctx.fillStyle = C.fgdim;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (const tk of ticks) {
      ctx.fillText(fmtNs(Math.max(0, tk.t)), tk.x + 3, AXIS_H / 2);
    }
    // Gutter/ruler corner label.
    ctx.fillStyle = C.bg2;
    ctx.fillRect(0, 0, GUTTER_W, AXIS_H);
    ctx.fillStyle = C.fgdim;
    ctx.fillText(t('prof.lane_corner'), 8, AXIS_H / 2);

    // Empty-state hint.
    if (S.totalEvents === 0) {
      ctx.fillStyle = C.fgdim;
      ctx.textAlign = 'center';
      ctx.fillText(
        S.enabled ? t('prof.capturing_empty')
                  : t('prof.no_events'),
        GUTTER_W + (W - GUTTER_W) / 2, AXIS_H + (H - AXIS_H) / 2);
      ctx.textAlign = 'left';
    }
  }

  // ---- hit testing -------------------------------------------------
  function itemAt(mx, my) {
    if (mx < GUTTER_W || my < AXIS_H) { return null; }
    const li = Math.floor((my - AXIS_H) / S.laneH);
    if (li < 0 || li >= S.lanes.length) { return null; }
    const items = S.lanes[li].items;
    if (!items || !items.length) { return null; }
    // Within a sub-rowed lane, pick the block under the cursor's sub-row
    // (so overlapping/stacked blocks are individually selectable); else
    // the nearest point within 6px.
    const rows = items.rows || 1;
    const subH = S.laneH / rows;
    const myInLane = (my - AXIS_H) - li * S.laneH;
    const hitRow = Math.max(0, Math.min(rows - 1, Math.floor(myInLane / subH)));
    let bestPoint = -1, bestDx = 7;
    for (let j = 0; j < items.length; j++) {
      const it = items[j];
      const x0 = GUTTER_W + (it.t0 - S.startNs) * S.pxPerNs;
      if (it.kind === 'block') {
        if ((it.row || 0) !== hitRow) { continue; }
        const x1 = Math.max(x0 + 2,
            GUTTER_W + (it.t1 - S.startNs) * S.pxPerNs);
        if (mx >= x0 - 1 && mx <= x1 + 1) {
          return { laneIdx: li, itemIdx: j };
        }
      } else {
        const dx = Math.abs(x0 - mx);
        if (dx < bestDx) { bestDx = dx; bestPoint = j; }
      }
    }
    return bestPoint >= 0 ? { laneIdx: li, itemIdx: bestPoint } : null;
  }

  // ---- inspector ---------------------------------------------------
  function row(k, v) {
    return el('div', { class: 'prof-row' },
      el('span', { class: 'prof-k' }, k),
      el('span', { class: 'prof-v' }, v));
  }

  function renderInspector() {
    clear(inspector);
    if (!S.selected) {
      const dur = S.totalEvents ? fmtNs(S.t1 - S.t0) : '—';
      inspector.append(
        el('div', { class: 'prof-insp-title' }, t('prof.capture_summary')),
        el('div', { class: 'prof-grid' },
          row(t('prof.events'), String(S.totalEvents)),
          row(t('prof.lanes'), String(S.lanes.length)),
          row(t('prof.span'), dur),
          row(t('prof.dropped'), String(S.totalDropped))),
        el('div', { class: 'prof-hint' },
          S.totalEvents
            ? t('prof.hint_click')
            : t('prof.hint_start')));
      return;
    }
    const lane = S.lanes[S.selected.laneIdx];
    const it = lane.items[S.selected.itemIdx];
    const st = S.stages[it.gvid];
    const stageName = st ? (st.id + '  (' + st.type + ')'
      + (st.pipeline ? '  · ' + st.pipeline : '')) : ('gvid ' + it.gvid);
    const name = eventName(it.gvid, it.type);
    const sw = el('span', { class: 'prof-swatch' });
    sw.style.background = colorFor(it.gvid);
    const rows = [
      row(t('prof.stage'), stageName),
      row(t('prof.lane'), lane.label),
    ];
    if (it.kind === 'block' && it.open) {
      // Ongoing: the begin never closed within the capture -- the
      // worker is still inside this resume (e.g. a blocking source).
      rows.push(
        row(t('prof.event'), name + t('prof.open_suffix')),
        row(t('prof.begin'), fmtNs(it.t0) + '   (' + fmtAbs(S.anchorRealtime,
          it.t0) + ')'),
        row(t('prof.end'), t('prof.still_open')),
        row(t('prof.duration'),
          '≥ ' + fmtNs(it.t1 - it.t0) + t('prof.ongoing')),
        row(t('prof.value'), String(it.value)));
    } else if (it.kind === 'block') {
      const endName = eventName(it.gvid, it.endType);
      rows.push(
        row(t('prof.event'), name + ' → ' + endName),
        row(t('prof.begin'), fmtNs(it.t0) + '   (' + fmtAbs(S.anchorRealtime,
          it.t0) + ')'),
        row(t('prof.end'), fmtNs(it.t1)),
        row(t('prof.duration'), fmtNs(it.t1 - it.t0)),
        row(t('prof.value'), it.value + (it.endValue !== it.value
          ? ' → ' + it.endValue : '')));
      // Derived throughput for LLM-lane activities (tok/s for prefill /
      // decode, etc.), from the count payload + the block duration.
      const rl = rateLabel(st ? st.id : '',
                           it.endValue || it.value, it.t1 - it.t0);
      if (rl) { rows.push(row(t('prof.throughput'), rl)); }
    } else {
      rows.push(
        row(t('prof.event'), name + t('prof.transient')),
        row(t('prof.t_rel'), fmtNs(it.t0) + '   (' + it.t0 + ' ns)'),
        row(t('prof.t_abs'), fmtAbs(S.anchorRealtime, it.t0)),
        row(t('prof.value'), String(it.value)));
    }
    rows.push(row(t('prof.gvid'), String(it.gvid)));
    inspector.append(
      el('div', { class: 'prof-insp-title' }, sw,
        (st ? st.id : name)),
      el('div', { class: 'prof-grid' }, ...rows));
  }

  // Resolve a (gvid,type) to a label: prefer the dump's per-stage
  // event_names, fall back to the reserved runtime names, else "type N".
  function eventName(gvid, type) {
    const st = S.stages[gvid];
    if (st && st.names && st.names[type] !== undefined) {
      return st.names[type];
    }
    if (type in RUNTIME_NAMES) { return RUNTIME_NAMES[type]; }
    return 'type ' + type;
  }

  // ---- data load ---------------------------------------------------
  // Derive the visible lanes from the full set: hide-workers on keeps only
  // the aux LLM / ANE lanes (backend-labeled); off shows every lane.
  function applyLaneFilter() {
    S.lanes = S.hideWorkers
        ? S.allLanes.filter((l) => l.aux)
        : S.allLanes;
  }

  function ingest(doc) {
    S.anchorRealtime = doc.anchor_realtime_ns || 0;
    const stages = {};
    for (const s of (doc.stages || [])) {
      stages[s.gvid] = {
        id: s.id, type: s.type, pipeline: s.pipeline || '',
        names: s.event_names || {},
      };
    }
    S.stages = stages;
    // Stable color per gvid by first appearance for nicer spread.
    const colors = {};
    let ci = 0;
    const lanes = [];
    let total = 0, dropped = 0, minNs = Infinity, maxNs = 0;
    // Pass 1: decode every lane's events and find the global time span.
    // pairItems needs the capture edge (maxNs) to extend unclosed
    // begins, so it can't run until the whole document is scanned.
    const pending = [];
    for (const th of (doc.threads || [])) {
      const e = th.events || {};
      const ns = e.ns || [], ty = e.type || [], gv = e.stage_gvid || [],
            va = e.value || [];
      const events = [];
      for (let i = 0; i < ns.length; i++) {
        const g = gv[i] || 0;
        if (!(g in colors)) { colors[g] = PALETTE[ci++ % PALETTE.length]; }
        const n = ns[i] || 0;
        events.push({ ns: n, type: ty[i] || 0, gvid: g, value: va[i] || 0 });
        if (n < minNs) { minNs = n; }
        if (n > maxNs) { maxNs = n; }
      }
      events.sort((a, b) => a.ns - b.ns);
      total += events.length;
      dropped += th.dropped || 0;
      pending.push({ th, events });
    }
    // Pass 2: pair into drawable items now that maxNs (the capture
    // edge) is known. Auxiliary lanes (LLM, ANE) carry an explicit
    // label from the backend; worker / overflow lanes are positional.
    for (const { th, events } of pending) {
      const aux = !!th.label;
      lanes.push({
        label: th.label || (th.is_overflow ? t('prof.lane_overflow')
                                : t('prof.lane_worker', { id: th.worker_id })),
        aux: aux,
        dropped: th.dropped || 0, capacity: th.capacity || 0,
        items: pairItems(events, maxNs),
      });
    }
    // Float the auxiliary activity lanes (LLM / ANE) to the top so the
    // perf-critical timelines are seen first; workers/overflow follow.
    // Array.sort is stable, so order within each group is preserved.
    lanes.sort((a, b) => (a.aux === b.aux) ? 0 : (a.aux ? -1 : 1));
    S.gvidColor = colors;
    S.allLanes = lanes;
    applyLaneFilter();       // sets S.lanes (respects the hide-workers toggle)
    S.totalEvents = total;
    S.totalDropped = dropped;
    if (total === 0) { minNs = 0; maxNs = 1; }
    S.t0 = minNs;
    S.t1 = Math.max(maxNs, minNs + 1);
    // Drop a stale selection that no longer maps to an item.
    if (S.selected) {
      const ln = S.lanes[S.selected.laneIdx];
      if (!ln || !ln.items || !ln.items[S.selected.itemIdx]) {
        S.selected = null;
      }
    }
  }

  async function loadData() {
    let doc;
    try { doc = await api.profilerData(); }
    catch (e) {
      toast(t('prof.data_failed', { msg: e.message }), 'error');
      return;
    }
    if (S.stopped) { return; }
    ingest(doc);
    layout();           // recomputes fitPx + re-fits if autoFit
    if (!S.autoFit) { clampView(); }
    render();
    renderInspector();
    updateStatusText();
  }

  async function loadStatus() {
    try {
      const st = await api.profilerStatus();
      if (S.stopped) { return; }
      applyStatus(st);
    } catch (e) { /* transient */ }
  }

  function updateStatusText() {
    statusEl.textContent = S.enabled
      ? t('prof.capturing', { n: S.totalEvents })
      : (S.totalEvents > 0
          ? t('prof.events_count', { n: S.totalEvents }) : t('prof.idle'));
    statusEl.classList.toggle('live', S.enabled);
  }

  function applyStatus(st) {
    S.enabled = !!(st && st.enabled);
    startBtn.disabled = S.enabled;
    stopBtn.disabled = !S.enabled;
    maxInput.disabled = S.enabled;
    updateStatusText();
  }

  // ---- capture control --------------------------------------------
  function startPoll() {
    stopPoll();
    S.poll = setInterval(() => {
      if (S.stopped || !document.body.contains(canvas)) {
        stopPoll(); S.stopped = true; return;
      }
      if (S.enabled) { loadData().then(() => loadStatus()); }
    }, 1500);
  }
  function stopPoll() {
    if (S.poll) { clearInterval(S.poll); S.poll = null; }
  }

  startBtn.addEventListener('click', async () => {
    const maxEv = Math.max(1, parseInt(maxInput.value, 10) || 65536);
    try { await api.profilerStart(maxEv); }
    catch (e) {
      toast(t('prof.start_failed', { msg: e.message }), 'error');
      return;
    }
    S.selected = null;
    S.autoFit = true;
    await loadStatus();
    await loadData();
    startPoll();
  });

  stopBtn.addEventListener('click', async () => {
    try { await api.profilerStop(); }
    catch (e) {
      toast(t('prof.stop_failed', { msg: e.message }), 'error');
      return;
    }
    stopPoll();
    await loadStatus();
    await loadData();
  });

  refreshBtn.addEventListener('click', () => loadData().then(loadStatus));
  fitBtn.addEventListener('click', () => fit());

  hideWorkersCb.addEventListener('change', () => {
    S.hideWorkers = hideWorkersCb.checked;
    try { localStorage.setItem(HIDE_WK_KEY, S.hideWorkers ? '1' : '0'); }
    catch (e) { /* storage unavailable */ }
    S.selected = null;      // a laneIdx would be stale after re-filtering
    applyLaneFilter();
    layout();               // fewer lanes -> taller lanes (more room)
    render();
    renderInspector();
  });

  // Reset: drop the captured events (the retained snapshot, and the
  // live buffers if still capturing) and clear the view to a clean
  // slate for the next capture.
  resetBtn.addEventListener('click', async () => {
    try { await api.profilerReset(); }
    catch (e) {
      toast(t('prof.reset_failed', { msg: e.message }), 'error');
      return;
    }
    S.selected = null;
    S.autoFit = true;
    ingest({ threads: [], stages: [] });   // clear lanes/stages/totals
    layout();
    render();
    renderInspector();
    await loadStatus();
    await loadData();
  });

  // ---- interaction: zoom / pan / select ---------------------------
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.shiftKey) {
      S.startNs += (e.deltaY) / S.pxPerNs;   // horizontal pan
      S.autoFit = false;
      clampView();
      render();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const nsAt = (mx - GUTTER_W) / S.pxPerNs + S.startNs;
    const factor = Math.exp(-e.deltaY * 0.0015);
    S.pxPerNs *= factor;
    S.autoFit = false;
    clampView();
    S.startNs = nsAt - (mx - GUTTER_W) / S.pxPerNs;
    clampView();
    render();
  }, { passive: false });

  let drag = null;   // {mx0, startNs0, moved}
  canvas.addEventListener('pointerdown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (mx < GUTTER_W) { return; }
    drag = { mx0: mx, startNs0: S.startNs, moved: false };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add('grabbing');
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) { return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (Math.abs(mx - drag.mx0) > 3) { drag.moved = true; }
    S.startNs = drag.startNs0 - (mx - drag.mx0) / S.pxPerNs;
    S.autoFit = false;
    clampView();
    render();
  });
  canvas.addEventListener('pointerup', (e) => {
    const wasDrag = drag;
    drag = null;
    canvas.classList.remove('grabbing');
    if (!wasDrag || wasDrag.moved) { return; }   // a real drag, not a click
    const rect = canvas.getBoundingClientRect();
    const hit = itemAt(e.clientX - rect.left, e.clientY - rect.top);
    S.selected = hit;
    render();
    renderInspector();
  });

  const onResize = () => {
    if (S.stopped || !document.body.contains(canvas)) { return; }
    layout();
    if (!S.autoFit) { clampView(); }
    render();
  };
  window.addEventListener('resize', onResize);

  // ---- details-panel resize (drag the divider) --------------------
  // Dragging up grows the inspector (it takes height from the flex-1
  // timeline above), down shrinks it. Clamped to a sane band and
  // persisted; the timeline canvas is re-laid-out live as it changes.
  const INSP_MIN = 120;
  function setInspectorHeight(px) {
    const maxH = Math.max(INSP_MIN,
      (container.clientHeight || window.innerHeight || 600) * 0.8);
    const h = Math.round(clamp(px, INSP_MIN, maxH));
    inspector.style.height = h + 'px';
    try { localStorage.setItem(INSP_H_KEY, String(h)); } catch (e) {}
  }
  let rz = null;   // { y0, h0 }
  resizer.addEventListener('pointerdown', (e) => {
    rz = { y0: e.clientY, h0: inspector.getBoundingClientRect().height };
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add('dragging');
    e.preventDefault();
  });
  resizer.addEventListener('pointermove', (e) => {
    if (!rz) { return; }
    setInspectorHeight(rz.h0 + (rz.y0 - e.clientY));
    onResize();   // timeline shrank/grew -- re-fit the canvas
  });
  const endResize = (e) => {
    if (!rz) { return; }
    rz = null;
    resizer.classList.remove('dragging');
    try { resizer.releasePointerCapture(e.pointerId); } catch (er) {}
    onResize();
  };
  resizer.addEventListener('pointerup', endResize);
  resizer.addEventListener('pointercancel', endResize);

  // ---- boot --------------------------------------------------------
  renderInspector();
  // Defer first layout until the timeline has a measured size.
  requestAnimationFrame(() => {
    layout();
    render();
    loadStatus().then(() => loadData()).then(() => {
      if (S.enabled) { startPoll(); }
    });
  });
}
