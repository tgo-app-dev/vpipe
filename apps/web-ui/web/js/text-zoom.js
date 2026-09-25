// A text-size (− / +) control for a console pane. The User I/O console
// and the Session Log both carry one: they are the two panes someone
// reads rather than operates, and the choice between "fit more of the
// run on screen" and "make this legible" is theirs, not the layout's.
//
// THE SIZE IS A PREFERENCE, NOT PANE STATE. It persists under `key`,
// and every live console sharing that key follows a change at once --
// the log can be split into several panes, and two of them showing
// different sizes would be a disagreement the next re-mount resolved
// silently in favour of whichever wrote last.
//
// Applied as an inline font-size on the console element and nothing
// else: the stylesheet gives its lines `line-height: 1.55` (unitless)
// and sizes every Markdown element in `em`, so one number scales the
// whole console proportionally, headings and code blocks included.
//
// NO PREFERENCE MEANS NO INLINE STYLE, which is why 0 is a value here
// rather than a missing one. The phone shell sets its own console size
// in CSS (12.5px, not the desktop's 12), and stamping a hardcoded
// default over it would quietly undo that on a device whose operator
// never touched this control.

import { el } from './dom.js';
import { t } from './i18n.js';

const MIN_PX = 8;
const MAX_PX = 28;
const STEP_PX = 1;

// key -> Set of apply(px) closures, one per live console on that key.
const followers = new Map();

const clampPx = (v) => (v < MIN_PX ? MIN_PX : (v > MAX_PX ? MAX_PX : v));

// The stored size, or 0 for "no opinion". Out-of-range values (a stale
// key, a hand-edited one) read as no opinion rather than being clamped
// into something the operator never chose.
function readPref(key) {
  try {
    const v = parseFloat(localStorage.getItem(key));
    return (v >= MIN_PX && v <= MAX_PX) ? v : 0;
  } catch (e) { return 0; }   // storage blocked in a private window
}

function writePref(key, px) {
  try {
    if (px) { localStorage.setItem(key, String(px)); }
    else { localStorage.removeItem(key); }
  } catch (e) { /* the preference just will not persist */ }
}

// What the stylesheet gives this console, MEASURED rather than assumed,
// so the phone's own rule is the phone's default. Returns 12 for a node
// that is not in the document yet (getComputedStyle answers nothing for
// one), which is why the caller refreshes the readout once on the next
// frame.
function baseSize(node) {
  const v = parseFloat(getComputedStyle(node).fontSize);
  return (Number.isFinite(v) && v > 0) ? v : 12;
}

function broadcast(key, px) {
  writePref(key, px);
  for (const fn of followers.get(key) || []) { fn(px); }
}

// Build the control for `node` (the scrolling console element) and
// register it as a follower of `key`. Returns {el, dispose}; the host's
// cleanup MUST call dispose, or a closed pane keeps taking broadcasts.
export function mountTextZoom(node, key) {
  const readout = el('button', {
    class: 'tz-size', type: 'button', title: t('common.text_size_reset'),
  });
  const btn = (label, titleKey, onclick) => el('button', {
    class: 'tz-btn', type: 'button', title: t(titleKey), onclick,
  }, label);

  // A step is taken from what is ON SCREEN, not from a remembered
  // number: with nothing stored that is the stylesheet's size, and
  // rounding it first means the ladder stays on whole pixels from a
  // fractional default (the phone's 12.5 steps to 13, not 13.5).
  const step = (d) => {
    const cur = readPref(key) || Math.round(baseSize(node));
    broadcast(key, clampPx(cur + d));
  };

  const ctl = el('span', { class: 'tz', title: t('common.text_size') },
    btn('A−', 'common.text_smaller', () => step(-STEP_PX)),
    readout,
    btn('A+', 'common.text_larger', () => step(STEP_PX)));

  // The readout is the number the BUTTONS operate on, which is why it
  // rounds: the phone's stylesheet default is 12.5 and it reads 13,
  // because that is what a press of + or - will step from.
  const apply = (px) => {
    if (px) { node.style.fontSize = px + 'px'; }
    else { node.style.removeProperty('font-size'); }
    readout.textContent = String(Math.round(px || baseSize(node)));
  };
  readout.addEventListener('click', () => broadcast(key, 0));

  let set = followers.get(key);
  if (!set) { set = new Set(); followers.set(key, set); }
  set.add(apply);
  apply(readPref(key));
  // The readout above may have been filled from a detached node. One
  // frame later the pane is in the document and the stylesheet's size
  // is readable; re-read it only while nothing is stored, since a
  // stored size needs no measurement.
  requestAnimationFrame(() => {
    if (!readPref(key) && node.isConnected) { apply(0); }
  });

  return { el: ctl, dispose: () => { set.delete(apply); } };
}
