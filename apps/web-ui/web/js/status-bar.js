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

import { el, clear } from './dom.js';
import { t, onLocaleChange } from './i18n.js';
import { METRICS, machineLabel, makeSparkline, makeHistory,
         startSystemPoll } from './system-metrics.js';

// Which cells track history here. See the note above.
const SPARK = new Set(['ane', 'gpu', 'gpu_mem']);

// One cell: label + meter (sparkline + numeric value). relabel()
// re-applies the i18n key so the persistent bar re-localizes in place on
// a language change.
function makeCell(labelKey, spark) {
  const valEl = el('span', { class: 'sb-val' }, '—');
  const meter = el('div', { class: 'sb-meter' });
  let line = null;
  let hist = null;
  if (spark) {
    line = makeSparkline();
    hist = makeHistory();
    meter.append(line.svg);
  }
  meter.append(valEl);
  const labelEl = el('span', { class: 'sb-label' }, t(labelKey));
  const root = el('div', { class: 'sb-cell' }, labelEl, meter);

  return {
    root,
    relabel() { labelEl.textContent = t(labelKey); },
    set(display, ratio) {
      valEl.textContent = display;
      valEl.classList.toggle('dim', display === '—');
      if (line) { line.update(hist.push(ratio)); }
    },
  };
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

  // The bar is mounted once and persists across view switches, so unlike
  // the views it is never re-mounted on a language change. Re-label in
  // place instead (history and live values untouched).
  const offLocale = onLocaleChange(() => {
    for (const { cell } of cells) { cell.relabel(); }
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

  return () => { stop(); offLocale(); };
}
