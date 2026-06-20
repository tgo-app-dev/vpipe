// Session Log view: a read-only console that streams the session's
// diagnostic log (the log_* channels: normal / verbose / debug, plus
// always), separate from the User I/O view (which carries the
// user-facing error / warn / info + interactive input). A debug-level
// dropdown in the pane header sets the live capture threshold; changing
// it affects only FUTURE messages -- already-shown lines stay.
//
// Polling model mirrors the User I/O console: one timer ticks ~600ms,
// pulling new lines incrementally by sequence. While the pane is
// detached (the user switched nav views), the tick PAUSES but the timer
// stays alive, so polling resumes when the persistent workspace
// re-attaches the same DOM. The backend caps the ring (wraparound
// buffer); the cap is configurable in Settings.
//
// Embeddable pane: renders the console into `body` and appends its
// level dropdown + Clear control into the caller-provided `actions`
// (the host pane header). Returns a cleanup that stops the poll.

import { el, clear, toast } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

const POLL_MS = 600;

// Fallback level list if the server doesn't return one (older backend).
const FALLBACK_LEVELS =
  ['error', 'warn', 'info', 'normal', 'verbose', 'debug'];

export function mountLog(body, actions) {
  clear(body);

  const consoleEl = el('div', { class: 'console allow-context-menu' });
  const root = el('div', { class: 'logview' }, consoleEl);
  body.append(root);

  // ---- header controls (level dropdown + Clear) -------------------
  const levelSel = el('select', { class: 'log-level',
    title: t('log.threshold') });
  // Populated once we learn the level list from the server; seed with
  // the fallback so the control is usable immediately.
  function fillLevels(levels, current) {
    clear(levelSel);
    for (const lv of levels) {
      levelSel.append(el('option', { value: lv }, lv));
    }
    if (current) { levelSel.value = current; }
  }
  fillLevels(FALLBACK_LEVELS, null);

  levelSel.addEventListener('change', async () => {
    const lv = levelSel.value;
    try {
      const r = await api.logSetLevel(lv);
      if (r && r.level) { levelSel.value = r.level; }
    } catch (e) {
      toast(t('log.set_level_failed', { msg: e.message }), 'error');
    }
  });

  const clearBtn = el('button', { class: 'btn ghost', onclick: () => {
    clear(consoleEl);
    lineEls.clear();
    // Keep lastSeq: future polls fetch only newer lines, so the
    // just-cleared history isn't re-pulled.
    api.logClear().catch(() => {});
  } }, t('common.clear'));

  if (actions) {
    actions.append(el('span', { class: 'log-level-label' }, t('log.level')),
                   levelSel, clearBtn);
  }

  // Load the current threshold + the server's level list.
  (async () => {
    try {
      const r = await api.logGetLevel();
      const levels = (r && r.levels && r.levels.length) ? r.levels
                                                        : FALLBACK_LEVELS;
      fillLevels(levels, r && r.level);
    } catch (e) {
      // Older backend / not available: keep the fallback list.
    }
  })();

  // ---- console state + follow-the-tail ----------------------------
  let lastSeq = 0;
  let stopped = false;
  const lineEls = new Map();   // seq -> element

  function atBottom() {
    return consoleEl.scrollHeight - consoleEl.scrollTop
           - consoleEl.clientHeight < 40;
  }

  // `pinned` keeps the newest line in view across detach/re-attach and
  // pane moves (both reset scrollTop to 0). Only a genuine user scroll
  // gesture flips it (an implicit scrollTop reset also fires 'scroll').
  let pinned = true;
  let lastGesture = 0;
  const markGesture = () => { lastGesture = Date.now(); };
  for (const ev of ['wheel', 'touchstart', 'touchmove', 'keydown',
                    'mousedown']) {
    consoleEl.addEventListener(ev, markGesture, { passive: true });
  }
  consoleEl.addEventListener('scroll', () => {
    if (Date.now() - lastGesture < 500) { pinned = atBottom(); }
  });
  function stickToBottom() {
    if (pinned) { consoleEl.scrollTop = consoleEl.scrollHeight; }
  }

  // Render one log line, keyed by seq (a re-fetch rebuilds without
  // duplicating). `frag`, when supplied, receives new nodes so a batch
  // is a single DOM mutation.
  function renderLine(l, frag) {
    let node = lineEls.get(l.seq);
    if (!node) {
      node = el('div', { class: 'line ' + (l.level || 'normal') });
      (frag || consoleEl).append(node);
      lineEls.set(l.seq, node);
    }
    node.textContent = l.text;
  }

  // Apply a poll batch: build a fragment, append once, scroll once. For
  // very large batches splice across rAF so the browser keeps painting.
  const CHUNK_LINES = 400;
  function applyBatch(lines) {
    if (!lines || !lines.length) { return; }
    let i = 0;
    const step = () => {
      const frag = document.createDocumentFragment();
      const end = Math.min(lines.length, i + CHUNK_LINES);
      for (; i < end; i++) { renderLine(lines[i], frag); }
      if (frag.firstChild) { consoleEl.append(frag); }
      if (i < lines.length) {
        requestAnimationFrame(step);
      } else {
        stickToBottom();
      }
    };
    step();
  }

  async function tick() {
    if (stopped) { return; }
    // Pause (don't stop) while detached: the workspace persists across
    // nav switches and re-attaches this same DOM.
    if (!document.body.contains(consoleEl)) { return; }
    stickToBottom();
    try {
      const c = await api.logConsole(lastSeq);
      if (c && c.lines) {
        applyBatch(c.lines);
        if (typeof c.latest === 'number') { lastSeq = c.latest; }
      }
    } catch (e) {
      // Transient (e.g. auth prompt) -- stay quiet, retry next tick.
    }
  }

  // Defer the first poll to a microtask: the workspace appends this
  // pane AFTER mountLog returns, so polling synchronously here would
  // trip tick()'s "detached" check and stop forever.
  queueMicrotask(() => { if (!stopped) { tick(); } });
  const timer = setInterval(() => {
    if (stopped) { clearInterval(timer); return; }
    tick();
  }, POLL_MS);

  return () => { stopped = true; clearInterval(timer); };
}
