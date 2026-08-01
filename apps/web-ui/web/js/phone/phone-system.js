// Phone System view: the four live monitors from the desktop's status
// bar, given a screen instead of a 42px strip.
//
// The desktop bar is squeezed -- 140px per meter, and MEM drops its
// sparkline for want of room. Here each monitor is a full-width card
// with a tall trace, so all FOUR are plotted, including the process
// footprint. What each one reads and why is in system-metrics.js, shared
// with the bar; this file is layout and lifecycle only.
//
// The phone shell hides the status bar (there is no room for it above a
// soft keyboard), which is what makes this view worth having: on a phone
// it is the only place these numbers appear.
//
// Not cached by the shell, so leaving the view stops the 1 Hz poll
// rather than leaving a phone waking its radio for a screen nobody is
// looking at. The trace restarts empty on return -- the sparkline pads
// with gaps, so a fresh history reads as "no samples yet" and not as a
// run of zeroes.

import { el, clear } from '../dom.js';
import { t } from '../i18n.js';
import { METRICS, machineLabel, makeSparkline, makeHistory,
         startSystemPoll, HIST_N, POLL_MS } from '../system-metrics.js';

export function mountPhoneSystem({ body, actions, setTitle }) {
  clear(body);
  clear(actions);
  setTitle(t('phone.system'));

  const machineVal = el('span', { class: 'ph-sys-machine' }, '—');
  const scroll = el('div', { class: 'ph-sys' },
    el('div', { class: 'ph-sys-head' },
      el('span', { class: 'ph-sys-head-label' }, t('status.machine_title')),
      machineVal));
  body.append(scroll);

  const cards = METRICS.map((m) => {
    const valEl = el('span', { class: 'ph-sys-val' }, '—');
    const line = makeSparkline();
    const hist = makeHistory();
    const card = el('div', { class: 'ph-sys-card' },
      el('div', { class: 'ph-sys-card-head' },
        el('span', { class: 'ph-sys-label' }, t(m.labelKey)), valEl),
      el('div', { class: 'ph-sys-plot' }, line.svg));
    scroll.append(card);
    return { metric: m, valEl, line, hist };
  });

  scroll.append(el('div', { class: 'ph-sys-foot' },
    t('phone.sys_window', { s: Math.round(HIST_N * POLL_MS / 1000) })));

  const paint = (c, display, ratio) => {
    c.valEl.textContent = display;
    c.valEl.classList.toggle('dim', display === '—');
    c.line.update(c.hist.push(ratio));
  };

  let machineSet = false;
  const stop = startSystemPoll(
    (s) => {
      if (!machineSet) {
        const label = machineLabel(s);
        if (label) { machineVal.textContent = label; machineSet = true; }
      }
      for (const c of cards) {
        const r = c.metric.read(s);
        paint(c, r.display, r.ratio);
      }
    },
    () => { for (const c of cards) { paint(c, '—', null); } });

  return () => stop();
}
