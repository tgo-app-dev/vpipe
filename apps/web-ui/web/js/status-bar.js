// Bottom system status bar. Polls /api/system/status every 1 s and
// renders GPU utilisation + memory side-by-side with MLX memory and
// host RSS. Backed by the same IOKit IORegistry data nvtop's
// Apple-Silicon backend uses (see apps/web-ui/system-status.cc).
//
// GPU util and GPU memory utilisation each carry a 100-second
// sparkline -- a ring buffer of the last N=100 samples is kept on the
// closure and re-rendered into the cell's <svg> on every poll. The
// numeric value sits on top of the sparkline at the right edge.
//
// Failures (network, auth, missing endpoint) reduce to a dim "—" --
// the poll keeps retrying so a restarted backend recovers without a
// page refresh.

import { el, clear, svgEl } from './dom.js';
import { api } from './api.js';
import { t, onLocaleChange } from './i18n.js';

const POLL_MS  = 1000;
const HIST_N   = 100;     // 100 samples * 1 s = 100 s window

function fmtBytes(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) { return '—'; }
  if (n < 1024) { return n + ' B'; }
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; ++i; }
  return v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2) + ' ' + units[i];
}

function fmtPct(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) { return '—'; }
  return (Math.round(n * 10) / 10).toFixed(n >= 100 ? 0 : 1) + '%';
}

// ----- Sparkline -----------------------------------------------------
//
// The history is stored as an Array of values in 0..1, padded at the
// front with NaN until we have HIST_N samples. The path skips NaN runs
// so a fresh page or a temporary fetch error doesn't draw a phantom
// line at the bottom.
function makeSparkline() {
  const w = 140;
  const h = 28;
  const svg = svgEl('svg', { class: 'sb-spark', viewBox: `0 0 ${w} ${h}`,
                             preserveAspectRatio: 'none' });
  const fillPath = svgEl('path', { class: 'fill', d: '' });
  const linePath = svgEl('path', { class: 'line', d: '' });
  svg.append(fillPath, linePath);

  const update = (history /* length-HIST_N array of 0..1 or NaN */) => {
    const stride = w / (HIST_N - 1);
    // Build segments separated by NaN gaps; the fill closes each
    // segment down to the baseline.
    let line = '';
    let fill = '';
    let inSeg = false;
    let segStartX = 0;
    for (let i = 0; i < HIST_N; i++) {
      const v = history[i];
      const x = i * stride;
      if (!Number.isFinite(v)) {
        if (inSeg) {
          // Close the current fill segment down to baseline.
          fill += ` L ${(i - 1) * stride} ${h} Z`;
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

// One "cell": label + meter (sparkline + numeric value). `kind` is
// either 'spark' (track 100s history; pass `ratio` per poll) or
// 'text' (just show the value, no history). `labelKey` is an i18n
// catalogue key resolved through t(); relabel() re-applies it so the
// persistent status bar re-localizes in place on a language change.
function makeCell(labelKey, kind) {
  const valEl = el('span', { class: 'sb-val' }, '—');
  const meter = el('div', { class: 'sb-meter' });
  let spark = null;
  let history = null;
  if (kind === 'spark') {
    spark = makeSparkline();
    history = new Array(HIST_N).fill(NaN);
    meter.append(spark.svg);
  }
  meter.append(valEl);
  const labelEl = el('span', { class: 'sb-label' }, t(labelKey));
  const root = el('div', { class: 'sb-cell' }, labelEl, meter);

  return {
    root,
    relabel() { labelEl.textContent = t(labelKey); },
    // For spark cells: push a fresh value (0..1) and a display string.
    // For text cells: just set the display.
    set(displayText, ratio) {
      valEl.textContent = displayText;
      valEl.classList.toggle('dim', displayText === '—');
      if (spark) {
        history.shift();
        history.push(Number.isFinite(ratio) ? ratio : NaN);
        spark.update(history);
      }
    },
    clearSpark() {
      if (history) {
        history.shift();
        history.push(NaN);
        spark.update(history);
      }
    },
  };
}

export function mountStatusBar(container) {
  clear(container);
  const body = el('div', { class: 'sb-body' });
  container.append(body);

  // Build cells up front so the layout is stable across polls and
  // the sparklines accumulate continuously rather than being torn
  // down on every render. Display order: ANE -> GPU -> GPU mem ->
  // MEM, right-aligned on the bar. "MEM" is task_vm_info.
  // phys_footprint -- the number Activity Monitor's Memory column
  // shows; the older "RSS" reading included shared library text and
  // tracked system memory size rather than process footprint.
  // Static machine label on the LEFT (chip / GPU model + core count,
  // e.g. "Apple M4 (10)"); a flex spacer keeps the live monitors
  // clustered at the right.
  const machineVal = el('span', { class: 'sb-mval' }, '—');
  const machine = el('div',
    { class: 'sb-machine', title: t('status.machine_title') }, machineVal);
  const grow = el('div', { class: 'sb-grow' });

  const cellAne     = makeCell('status.ane',     'spark');
  const cellGpu     = makeCell('status.gpu',     'spark');
  const cellGpuMem  = makeCell('status.gpu_mem', 'spark');
  const cellMem     = makeCell('status.mem',     'text');
  body.append(machine, grow,
              cellAne.root, cellGpu.root, cellGpuMem.root, cellMem.root);

  // The status bar is mounted once and persists across view switches, so
  // it doesn't get re-mounted on a language change like the views do.
  // Re-label its static cells in place when the locale changes (the
  // sparkline history + live values are untouched).
  const offLocale = onLocaleChange(() => {
    cellAne.relabel();
    cellGpu.relabel();
    cellGpuMem.relabel();
    cellMem.relabel();
    machine.setAttribute('title', t('status.machine_title'));
  });

  let stopped = false;
  let timer = null;
  let machineSet = false;   // the model is static -- set it once

  const update = (s) => {
    // Machine label: GPU/chip model + core count. Static, so set it on
    // the first poll that carries it and leave it thereafter (never
    // blanked on a transient fetch error -- it's not a live metric).
    if (!machineSet && s && s.gpu_model) {
      const cores = Number.isFinite(s.gpu_cores) ? ' (' + s.gpu_cores + ')'
                                                 : '';
      machineVal.textContent = s.gpu_model + cores;
      machineSet = true;
    }

    // ANE utilisation estimated from ANE power, the same approach
    // macmon uses: ane_power_w / ane_max_w. The backend already
    // computes the percentage; here we only need to scale it to the
    // sparkline's 0..1 range and pick a sensible label.
    const ane = s && Number.isFinite(s.ane_util_pct)
        ? s.ane_util_pct : null;
    const aneW = s && Number.isFinite(s.ane_power_w)
        ? s.ane_power_w : null;
    cellAne.set(
        ane === null ? '—'
            : fmtPct(ane)
              + (aneW !== null
                  ? ' (' + aneW.toFixed(2) + ' W)' : ''),
        ane !== null ? ane / 100 : null);

    // GPU utilisation [0,100] -> sparkline ratio in [0,1].
    const util = s && Number.isFinite(s.gpu_util_pct)
        ? s.gpu_util_pct : null;
    cellGpu.set(fmtPct(util), util !== null ? util / 100 : null);

    // GPU memory: report the IOAccelerator "Alloc system memory"
    // counter -- the total wired-down GPU pool. nvtop's Apple-
    // Silicon backend, asitop and mactop all use the same field
    // (the "In use system memory" counter is much smaller because
    // it only counts pages actively touched by current work; using
    // it underreports GPU footprint by ~10x relative to nvtop).
    // Apple Silicon is UMA so we still scale against host physical
    // memory for the bar.
    const gpuMem = s && Number.isFinite(s.gpu_alloc_bytes)
        ? s.gpu_alloc_bytes
        : (s && Number.isFinite(s.gpu_in_use_bytes)
            ? s.gpu_in_use_bytes : null);
    const total = s && Number.isFinite(s.phys_total_bytes)
        ? s.phys_total_bytes : null;
    const memRatio = (gpuMem !== null && total && total > 0)
        ? gpuMem / total : null;
    cellGpuMem.set(
        gpuMem === null ? '—'
            : fmtBytes(gpuMem) + (total ? ' / ' + fmtBytes(total) : ''),
        memRatio);

    // Process physical footprint (matches Activity Monitor's
    // Memory > Memory column for this row).
    cellMem.set(s && Number.isFinite(s.phys_footprint_bytes)
        ? fmtBytes(s.phys_footprint_bytes) : '—',
        null);
  };

  const onError = () => {
    // set(..., null) already pushes a NaN sample on spark cells, so
    // the time axis still advances during outages. No need to also
    // call clearSpark (which would double-advance).
    cellAne.set('—', null);
    cellGpu.set('—', null);
    cellGpuMem.set('—', null);
    cellMem.set('—', null);
  };

  const poll = async () => {
    try {
      const s = await api.systemStatus();
      if (stopped) { return; }
      update(s);
    } catch (e) {
      if (stopped) { return; }
      onError();
    } finally {
      if (!stopped) { timer = setTimeout(poll, POLL_MS); }
    }
  };

  poll();

  return () => {
    stopped = true;
    if (timer) { clearTimeout(timer); timer = null; }
    offLocale();
  };
}
