// Bottom system status bar: the four live monitors on one 42px row,
// right-aligned, with the machine label pinned left.
//
// What each monitor MEANS -- which field it reads and why -- lives in
// system-metrics.js, shared with the phone's System view. This file is
// only the bar: a fixed row of cells built once so the layout is stable
// across polls and the sparklines accumulate continuously instead of
// being torn down on every render.
//
// Three of the four carry a 100-second sparkline. MEM does not: at 140px
// wide next to three other cells there is no room for a trend nobody
// reads at a glance, and the footprint number is the thing being
// watched. The phone view, which has a whole screen, draws all four.
//
// Failures (network, auth, missing endpoint) reduce to a dim "—"; the
// poll keeps retrying so a restarted backend recovers without a reload.

import { api } from './api.js';
import { el, clear } from './dom.js';
import { t, onLocaleChange } from './i18n.js';
import { METRICS, machineLabel, makeSparkline, makeHistory, POLL_MS,
         startSystemPoll } from './system-metrics.js';

// Which cells track history here. See the note above.
const SPARK = new Set(['ane', 'gpu', 'gpu_mem']);

// One cell: label + meter (sparkline + numeric value). relabel()
// re-applies the i18n key so the persistent bar re-localizes in place on
// a language change.
//
// `fill` swaps the sparkline for a left-to-right accent FILL -- the shape
// progress wants. A sparkline plots a value against time, which is right
// for GPU load but wrong for a task that only ever moves one way; the
// fill reads as "this far along" at a glance, and shares the accent the
// sparkline already tints with so the row stays one visual system.
function makeCell(labelKey, spark, fill) {
  const valEl = el('span', { class: 'sb-val' }, '—');
  const meter = el('div', { class: 'sb-meter' });
  let line = null;
  let hist = null;
  let fillEl = null;
  if (spark) {
    line = makeSparkline();
    hist = makeHistory();
    meter.append(line.svg);
  }
  if (fill) {
    fillEl = el('div', { class: 'sb-fill' });
    meter.append(fillEl);
  }
  meter.append(valEl);
  const labelEl = el('span', { class: 'sb-label' }, t(labelKey));
  const root = el('div', { class: 'sb-cell' }, labelEl, meter);

  return {
    root,
    relabel() { labelEl.textContent = t(labelKey); },
    // `ratio` null means INDETERMINATE when anything is live (the fill
    // goes striped and full-width) and simply empty when nothing is.
    set(display, ratio) {
      valEl.textContent = display;
      const idle = display === '—';
      valEl.classList.toggle('dim', idle);
      if (line) { line.update(hist.push(ratio)); }
      if (fillEl) {
        const indet = !idle && (ratio === null || !isFinite(ratio));
        const r = indet ? 1 : Math.max(0, Math.min(1, ratio || 0));
        fillEl.style.width = `${r * 100}%`;
        fillEl.classList.toggle('indet', indet);
      }
    },
  };
}

// ---- progress cell + panel -------------------------------------------
//
// The fifth monitor, and the only one that is NOT a system metric: it
// reads /api/io/progress (the UI delegate's live reports), not the
// /api/system/status sample the other four share. So it keeps its own
// poll and stays out of METRICS, which describes that sample.
//
// The CELL shows the most recently updated report -- highest `seq`,
// which is exactly what that field is bumped for -- because at a glance
// "what is happening now" beats an arbitrary pick. The PANEL, opened by
// clicking, shows all of them, since concurrent reports are the whole
// reason the registry holds a list.
function pctOf(item) {
  if (!item || !item.total) { return null; }   // 0 total => indeterminate
  const p = Math.round((item.done / item.total) * 100);
  return Math.max(0, Math.min(100, p));
}

function fracOf(item) {
  if (!item || !item.total) { return null; }
  return Math.max(0, Math.min(1, item.done / item.total));
}

// ---- remaining-time estimate -----------------------------------------
//
// Rate is measured over a sliding WINDOW rather than since the report
// opened. Progress here is not uniform -- a denoise loads and pins
// weights before the first step, a download slows when another starts --
// so an average taken from t=0 keeps dragging a stale early rate into a
// number the user is reading right now.
//
// It stays silent until it has ETA_MIN_SAMPLES over ETA_MIN_SPAN_MS. Two
// samples a second apart can "predict" an hour, and a figure that swings
// wildly for the first few seconds is worse than none.
const ETA_MIN_SAMPLES = 4;
const ETA_MIN_SPAN_MS = 3000;
const ETA_WINDOW_MS   = 30000;

function makeEtaTracker() {
  const hist = new Map();          // id -> [{t, frac}]
  return {
    // Record one poll's worth of samples and drop history for reports
    // that have finished (ids are never reused, so this cannot resurrect
    // a stale series).
    sample(items, now) {
      const live = new Set();
      for (const it of items) {
        live.add(it.id);
        const f = fracOf(it);
        if (f === null) { continue; }          // indeterminate: no rate
        let a = hist.get(it.id);
        if (!a) { a = []; hist.set(it.id, a); }
        a.push({ t: now, frac: f });
        while (a.length > 1 && now - a[0].t > ETA_WINDOW_MS) { a.shift(); }
      }
      for (const id of [...hist.keys()]) {
        if (!live.has(id)) { hist.delete(id); }
      }
    },
    // Seconds remaining, or null when there is not enough to say.
    seconds(item) {
      const f = fracOf(item);
      if (f === null || f >= 1) { return null; }
      const a = hist.get(item.id);
      if (!a || a.length < ETA_MIN_SAMPLES) { return null; }
      const first = a[0];
      const last = a[a.length - 1];
      const dt = last.t - first.t;
      const df = last.frac - first.frac;
      if (dt < ETA_MIN_SPAN_MS || df <= 0) { return null; }
      return ((1 - last.frac) * dt) / df / 1000;
    },
  };
}

function formatEta(sec) {
  if (sec === null || !isFinite(sec) || sec < 0) { return null; }
  const s = Math.round(sec);
  if (s < 60) { return `${s}s`; }
  const m = Math.floor(s / 60);
  if (m < 60) { return `${m}m ${String(s % 60).padStart(2, '0')}s`; }
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

function makeProgressPanel() {
  const list = el('div', { class: 'sb-prog-list' });
  const root = el('div', { class: 'sb-prog-panel', hidden: '' }, list);

  return {
    root,
    toggle() {
      if (root.hasAttribute('hidden')) { root.removeAttribute('hidden'); }
      else { root.setAttribute('hidden', ''); }
      return !root.hasAttribute('hidden');
    },
    close() { root.setAttribute('hidden', ''); },
    isOpen() { return !root.hasAttribute('hidden'); },
    render(items, eta) {
      clear(list);
      if (!items.length) {
        list.append(el('div', { class: 'sb-prog-empty' }, t('status.no_progress')));
        return;
      }
      for (const it of items) {
        const p = pctOf(it);
        // An indeterminate report gets a full-width striped bar rather
        // than a 0% one, so "no total yet" never reads as "stuck".
        const fill = el('div', {
          class: p === null ? 'sb-prog-fill indet' : 'sb-prog-fill',
          style: `width:${p === null ? 100 : p}%`,
        });
        const track = el('div', { class: 'sb-prog-track' }, fill);
        const name = el('span', { class: 'sb-prog-name' }, it.desc || '—');
        const val = el('span', { class: 'sb-prog-pct' },
                       p === null ? '—' : `${p}%`);
        const head = el('div', { class: 'sb-prog-head' }, name, val);
        const rows = [head, track];
        // Detail on the left, remaining-time on the right of one line --
        // both are secondary to the bar, so they share a row instead of
        // pushing every item taller.
        const left = it.detail || '';
        const right = eta ? (formatEta(eta.seconds(it)) || '') : '';
        if (left || right) {
          rows.push(el('div', { class: 'sb-prog-sub' },
                       el('span', { class: 'sb-prog-detail' }, left),
                       el('span', { class: 'sb-prog-eta' },
                          right ? t('status.eta_left').replace('{t}', right)
                                : '')));
        }
        list.append(el('div', { class: 'sb-prog-item' }, ...rows));
      }
    },
  };
}

// Poll /api/io/progress and hand the items to `onItems(items, now)`.
// Failures fall back to an empty list so a restarted backend recovers
// without a reload, matching how the system poll degrades.
//
// The version counter is used only to skip work while NOTHING is live --
// the idle case, which is most of a session. It cannot gate the callback
// generally: the estimator needs a sample per poll even when the report
// has not moved (a stall is exactly what it must notice), and the
// remaining-time text ticks down on its own clock.
function startProgressPoll(onItems) {
  let stopped = false;
  let timer = null;
  let wasIdle = false;
  const tick = async () => {
    try {
      const s = await api.ioProgress();
      const items = Array.isArray(s?.items) ? s.items : [];
      const idle = items.length === 0;
      if (!idle || !wasIdle) { onItems(items, Date.now()); }
      wasIdle = idle;
    } catch {
      if (!wasIdle) { onItems([], Date.now()); wasIdle = true; }
    }
    if (!stopped) { timer = setTimeout(tick, POLL_MS); }
  };
  tick();
  return () => { stopped = true; if (timer) { clearTimeout(timer); } };
}

export function mountStatusBar(container) {
  clear(container);
  const body = el('div', { class: 'sb-body' });
  container.append(body);

  const machineVal = el('span', { class: 'sb-mval' }, '—');
  const machine = el('div',
    { class: 'sb-machine', title: t('status.machine_title') }, machineVal);
  body.append(machine, el('div', { class: 'sb-grow' }));

  const cells = METRICS.map((m) => {
    const c = makeCell(m.labelKey, SPARK.has(m.key));
    body.append(c.root);
    return { metric: m, cell: c };
  });

  // The progress cell sits last, after the system monitors, and unlike
  // them it is interactive: clicking expands the panel of all live
  // reports. The panel is a child of the bar's container so it can be
  // positioned above the 42px row.
  const progCell = makeCell('status.progress', false, /*fill=*/true);
  progCell.root.classList.add('sb-clickable');
  body.append(progCell.root);
  const panel = makeProgressPanel();
  container.append(panel.root);

  let progItems = [];
  const eta = makeEtaTracker();
  progCell.root.addEventListener('click', () => {
    if (panel.toggle()) { panel.render(progItems, eta); }
  });
  // Clicking away closes it; the bar is persistent, so a panel left
  // open would hang over every view.
  const onDocClick = (ev) => {
    if (!panel.isOpen()) { return; }
    if (panel.root.contains(ev.target) || progCell.root.contains(ev.target)) {
      return;
    }
    panel.close();
  };
  document.addEventListener('click', onDocClick);

  const stopProg = startProgressPoll((items, now) => {
    progItems = items;
    // Sample on EVERY poll, including ones the version guard would let
    // through unchanged -- a stalled report is information the estimate
    // needs, and skipping those samples would make a stall look fast.
    eta.sample(items, now);
    if (!items.length) {
      progCell.set('—', 0);
      progCell.root.removeAttribute('title');
    } else {
      // Highest seq = most recently updated.
      const cur = items.reduce((a, b) => ((b.seq || 0) > (a.seq || 0) ? b : a));
      const p = pctOf(cur);
      const label = items.length > 1 ? `${cur.desc} +${items.length - 1}`
                                     : cur.desc;
      progCell.set(p === null ? `${label}` : `${label} ${p}%`,
                   p === null ? null : p / 100);
      // The meter is 140px, so the ETA rides in the tooltip rather than
      // crowding the value out of the cell; the panel shows it inline.
      const left = formatEta(eta.seconds(cur));
      if (left) {
        progCell.root.setAttribute(
            'title', t('status.eta_left').replace('{t}', left));
      } else {
        progCell.root.removeAttribute('title');
      }
    }
    if (panel.isOpen()) { panel.render(items, eta); }
  });

  // The bar is mounted once and persists across view switches, so unlike
  // the views it is never re-mounted on a language change. Re-label in
  // place instead (history and live values untouched).
  const offLocale = onLocaleChange(() => {
    for (const { cell } of cells) { cell.relabel(); }
    progCell.relabel();
    if (panel.isOpen()) { panel.render(progItems, eta); }
    machine.setAttribute('title', t('status.machine_title'));
  });

  let machineSet = false;      // the model is static -- set it once
  const stop = startSystemPoll(
    (s) => {
      if (!machineSet) {
        const label = machineLabel(s);
        if (label) { machineVal.textContent = label; machineSet = true; }
      }
      for (const { metric, cell } of cells) {
        const r = metric.read(s);
        cell.set(r.display, r.ratio);
      }
    },
    () => {
      // A null ratio still pushes a NaN sample, so the time axis keeps
      // advancing through an outage and the gap is visible.
      for (const { cell } of cells) { cell.set('—', null); }
    });

  return () => {
    stop();
    stopProg();
    offLocale();
    document.removeEventListener('click', onDocClick);
  };
}
