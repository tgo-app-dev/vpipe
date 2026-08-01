// The live system monitors, independent of how they are drawn.
//
// /api/system/status carries the Apple-Silicon IOKit readings (the same
// IORegistry data nvtop's AS backend, asitop and mactop consume -- see
// apps/web-ui/system-status.cc). Turning those raw fields into the four
// monitors is not obvious: ANE utilisation is inferred from power, GPU
// memory has two counters that differ by ~10x and only one of them is
// comparable to what other tools report, and "MEM" is a footprint rather
// than an RSS. That reasoning lives HERE, once, because two very
// different renderers need it -- the desktop's 42px status bar and the
// phone's full-screen System view -- and a second copy of it would be a
// second set of decisions to keep in step.
//
// What each renderer keeps to itself is layout: sizes, which monitors
// get a sparkline, and where the numbers sit.

import { svgEl } from './dom.js';
import { api } from './api.js';

export const HIST_N  = 100;    // 100 samples * 1 s = a 100 s window
export const POLL_MS = 1000;

export function fmtBytes(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) { return '—'; }
  if (n < 1024) { return n + ' B'; }
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; ++i; }
  return v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2) + ' ' + units[i];
}

export function fmtPct(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) { return '—'; }
  return (Math.round(n * 10) / 10).toFixed(n >= 100 ? 0 : 1) + '%';
}

const num = (s, k) => (s && Number.isFinite(s[k]) ? s[k] : null);

// The four monitors, in display order. `read(status)` returns
//   { display, ratio }
// where `ratio` is 0..1 for the sparkline (null = no sample this tick,
// which draws a gap rather than a phantom zero).
export const METRICS = [
  {
    key: 'ane',
    labelKey: 'status.ane',
    // Estimated from ANE POWER, the same way macmon does it:
    // ane_power_w / ane_max_w. The backend already reduces that to a
    // percentage; the watts ride along because they are the reading
    // that is actually measured.
    read(s) {
      const pct = num(s, 'ane_util_pct');
      const w = num(s, 'ane_power_w');
      return {
        display: pct === null ? '—'
          : fmtPct(pct) + (w !== null ? ' (' + w.toFixed(2) + ' W)' : ''),
        ratio: pct === null ? null : pct / 100,
      };
    },
  },
  {
    key: 'gpu',
    labelKey: 'status.gpu',
    read(s) {
      const pct = num(s, 'gpu_util_pct');
      return { display: fmtPct(pct), ratio: pct === null ? null : pct / 100 };
    },
  },
  {
    key: 'gpu_mem',
    labelKey: 'status.gpu_mem',
    // The IOAccelerator "Alloc system memory" counter -- the total
    // wired-down GPU pool, which is what nvtop's Apple-Silicon backend,
    // asitop and mactop all report. The "In use system memory" counter
    // is ~10x smaller because it only counts pages actively touched by
    // current work, so using it would badly underreport the footprint;
    // it is the fallback only when the first is missing. Apple Silicon
    // is UMA, so the bar scales against host physical memory.
    read(s) {
      const used = num(s, 'gpu_alloc_bytes') !== null
        ? num(s, 'gpu_alloc_bytes') : num(s, 'gpu_in_use_bytes');
      const total = num(s, 'phys_total_bytes');
      return {
        display: used === null ? '—'
          : fmtBytes(used) + (total ? ' / ' + fmtBytes(total) : ''),
        ratio: (used !== null && total && total > 0) ? used / total : null,
      };
    },
  },
  {
    key: 'mem',
    labelKey: 'status.mem',
    // task_vm_info.phys_footprint -- the number Activity Monitor's
    // Memory column shows for this process. The older RSS reading
    // counted shared library text and tracked system memory size rather
    // than process footprint.
    read(s) {
      const used = num(s, 'phys_footprint_bytes');
      const total = num(s, 'phys_total_bytes');
      return {
        display: used === null ? '—' : fmtBytes(used),
        ratio: (used !== null && total && total > 0) ? used / total : null,
      };
    },
  },
];

// The static machine label (chip / GPU model + core count), or null when
// this sample doesn't carry it. Not a live metric: a caller sets it once
// and leaves it, so a transient fetch error never blanks it.
export function machineLabel(s) {
  if (!s || !s.gpu_model) { return null; }
  const cores = Number.isFinite(s.gpu_cores) ? ' (' + s.gpu_cores + ')' : '';
  return s.gpu_model + cores;
}

// ----- Sparkline -----------------------------------------------------
//
// History is an Array of values in 0..1, padded at the front with NaN
// until HIST_N samples have arrived. The path SKIPS NaN runs, so a fresh
// page or a temporary fetch error leaves a gap rather than drawing a
// phantom line along the bottom.
//
// The viewBox is fixed and `preserveAspectRatio: none` lets a caller
// stretch the svg to any box with CSS -- which the phone view does, to
// several times the desktop's 140x28. `non-scaling-stroke` is what makes
// that safe: without it the line would thicken with the stretch, and by
// a different amount on each axis.
export function makeSparkline() {
  const w = 140;
  const h = 28;
  const svg = svgEl('svg', { class: 'sb-spark', viewBox: `0 0 ${w} ${h}`,
                             preserveAspectRatio: 'none' });
  const fillPath = svgEl('path', { class: 'fill', d: '' });
  const linePath = svgEl('path', { class: 'line', d: '',
                                   'vector-effect': 'non-scaling-stroke' });
  svg.append(fillPath, linePath);

  const update = (history /* length-HIST_N array of 0..1 or NaN */) => {
    const stride = w / (HIST_N - 1);
    let line = '';
    let fill = '';
    let inSeg = false;
    let segStartX = 0;
    for (let i = 0; i < HIST_N; i++) {
      const v = history[i];
      const x = i * stride;
      if (!Number.isFinite(v)) {
        if (inSeg) {
          fill += ` L ${(i - 1) * stride} ${h} Z`;   // close to baseline
          inSeg = false;
        }
        continue;
      }
      const y = h - 1 - Math.min(h - 2, Math.max(0, v) * (h - 2));
      if (!inSeg) {
        line += `M ${x} ${y}`;
        fill += `M ${x} ${h} L ${x} ${y}`;
        inSeg = true;
        segStartX = x;
      } else {
        line += ` L ${x} ${y}`;
        fill += ` L ${x} ${y}`;
      }
    }
    if (inSeg) {
      fill += ` L ${(HIST_N - 1) * stride} ${h} L ${segStartX} ${h} Z`;
    }
    linePath.setAttribute('d', line);
    fillPath.setAttribute('d', fill);
  };

  return { svg, update };
}

// A rolling history buffer for one monitor.
export function makeHistory() {
  const hist = new Array(HIST_N).fill(NaN);
  return {
    push(ratio) {
      hist.shift();
      hist.push(Number.isFinite(ratio) ? ratio : NaN);
      return hist;
    },
    values: hist,
  };
}

// Poll /api/system/status until stopped. `onSample(status)` per success,
// `onError()` per failure -- the loop keeps going either way, so a
// restarted backend recovers without a page reload. Returns stop().
export function startSystemPoll(onSample, onError) {
  let stopped = false;
  let timer = null;
  const tick = async () => {
    try {
      const s = await api.systemStatus();
      if (!stopped) { onSample(s); }
    } catch (e) {
      if (!stopped) { onError(); }
    } finally {
      if (!stopped) { timer = setTimeout(tick, POLL_MS); }
    }
  };
  tick();
  return () => {
    stopped = true;
    if (timer) { clearTimeout(timer); timer = null; }
  };
}
