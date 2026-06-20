// User I/O view: a console that streams the session's error / warn /
// info messages, plus an input row that answers interactive getline()
// requests (e.g. text-input / onvif-discovery stages).
//
// Polling model: one timer ticks ~600ms. Each tick pulls new console
// lines (incrementally, by sequence) and the current pending-input
// request. While this pane is detached from the document (the user
// switched to another nav view), the tick PAUSES -- it skips the fetch
// but keeps the timer alive -- so polling resumes when the persistent
// workspace re-attaches the same DOM. A permanent stop happens only via
// the returned cleanup() (when the pane is closed).
//
// Embeddable pane: renders the console/input into `body`, and appends
// its "Clear" control into the caller-provided `actions` element (the
// host pane header). Returns a cleanup function that stops the poll.

import { el, clear, toast, kbd } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';
import { renderMarkdown } from '../markdown.js';

const POLL_MS = 600;

// Persisted preference for the "Markdown" render toggle (off by default).
const MD_KEY = 'vpipe_io_markdown';
function readMdPref() {
  try { return localStorage.getItem(MD_KEY) === '1'; }
  catch (e) { return false; }
}

export function mountUserIo(body, actions) {
  clear(body);

  // .allow-context-menu opts the console + input out of the app-wide
  // right-click suppression so copy / paste / select-all work here.
  const consoleEl = el('div', { class: 'console allow-context-menu' });
  const promptEl  = el('span', { class: 'io-prompt' }, t('userio.waiting'));
  // Multi-line, wrapping input. Enter sends; Ctrl+J inserts a newline.
  const input     = el('textarea', { class: 'io-text allow-context-menu',
                                     rows: '1',
                                     placeholder: t('userio.response_ph'),
                                     disabled: true });
  const sendBtn   = el('button', { class: 'btn primary', disabled: true },
                       t('common.send'), kbd('⏎'));

  const clearBtn  = el('button', { class: 'btn ghost', onclick: () => {
    clear(consoleEl);
    lineEls.clear();
    // Keep lastSeq as-is: future polls fetch only lines newer than the
    // current tip, so the just-cleared history isn't re-pulled (and we
    // avoid a race with the in-flight clear).
    api.ioClear().catch(() => {});   // authoritative: drop backend history
  } }, t('common.clear'));

  // Markdown render toggle: when on, each console line's text is rendered
  // as simple Markdown (bold/italic/underline, headings, lists, code,
  // tables) instead of plain pre-wrapped text. The choice persists and
  // flipping it re-renders the lines already on screen.
  let mdEnabled = readMdPref();
  const mdCheck = el('input', { type: 'checkbox' });
  mdCheck.checked = mdEnabled;
  const mdToggle = el('label', { class: 'io-md-toggle',
                                 title: t('userio.markdown_title') },
                      mdCheck, t('userio.markdown'));
  mdCheck.addEventListener('change', () => {
    mdEnabled = mdCheck.checked;
    try { localStorage.setItem(MD_KEY, mdEnabled ? '1' : '0'); }
    catch (e) { /* storage blocked -- preference just won't persist */ }
    rerenderAll();
  });

  const hint = el('div', { class: 'io-hint' },
    el('kbd', { class: 'kbd' }, 'Ctrl+J'), t('userio.newline') + '  ·  ',
    el('kbd', { class: 'kbd' }, '⏎'), t('userio.send_word'));

  const root = el('div', { class: 'userio' },
    consoleEl,
    el('div', { class: 'io-input' }, promptEl, input, sendBtn),
    hint);
  body.append(root);
  // The pane header (owned by the workspace) carries the Markdown toggle
  // + Clear button.
  if (actions) { actions.append(mdToggle, clearBtn); }

  // Grow the textarea with its content up to a cap, then scroll.
  function autosize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  }

  let lastSeq = 0;
  let pendingId = 0;        // id currently shown in the input box (0 = none)
  let lastAnsweredId = 0;   // highest request id we've already answered
  let stopped = false;
  const lineEls = new Map();   // seq -> element

  function atBottom() {
    return consoleEl.scrollHeight - consoleEl.scrollTop
           - consoleEl.clientHeight < 40;
  }

  // Follow-the-tail state. `pinned` means "keep the newest line in
  // view". It must survive the pane being detached/re-attached (a nav
  // switch away and back) or moved in the DOM (a workspace split):
  // both reset the scroll container's scrollTop to 0, which would
  // otherwise look like the user scrolled to the top and would strand
  // the console there. We re-pin to the bottom on every tick while
  // pinned (see tick()), and only let a genuine user scroll change
  // `pinned` -- an implicit scrollTop reset also fires a 'scroll'
  // event, so we ignore scroll events that aren't right after a real
  // scroll gesture.
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

  // Render or update one console line, keyed by seq. Every kind of
  // output -- info/warn/error, live text streams ("stream", grows in
  // place while `open`), and answered input echoes ("input") -- comes
  // through here. Keying by seq means a re-fetch (e.g. after the view
  // is re-mounted) rebuilds the full transcript without duplicating,
  // and a growing stream updates its own line.
  //
  // `frag`, when supplied, receives newly-created nodes instead of
  // appending them to consoleEl directly. The caller is responsible
  // for appending the fragment exactly once -- that turns N appends
  // into a single DOM mutation and a single layout, which is the
  // difference between "view switch is instant" and "browser prompts
  // to kill the page" on a long backlog.
  function renderLine(l, frag) {
    let node = lineEls.get(l.seq);
    if (!node) {
      node = el('div', { class: 'line ' + (l.level || 'info') });
      (frag || consoleEl).append(node);
      lineEls.set(l.seq, node);
    }
    // Keep the source line on the node so toggling Markdown can re-render
    // the existing transcript without re-fetching it.
    node._src = l;
    renderLineBody(node, l);
    if (l.level === 'stream') {
      node.classList.toggle('streaming', !!l.open);
    }
  }

  // Fill a line node from its source text, in the current render mode.
  function renderLineBody(node, l) {
    if (mdEnabled) {
      node.classList.add('md');
      node.replaceChildren(renderMarkdown(l.text));
    } else {
      node.classList.remove('md');
      node.textContent = l.text;
    }
  }

  // Re-render every line already on screen (after the Markdown toggle
  // flips). Toggling is a deliberate, infrequent action, so a straight
  // pass over the existing nodes is fine.
  function rerenderAll() {
    for (const node of lineEls.values()) {
      if (node._src) { renderLineBody(node, node._src); }
    }
    stickToBottom();
  }

  // Apply a batch of lines from one poll. Reads scroll position once
  // up front, builds a DocumentFragment for new nodes, appends it in
  // one DOM mutation, and scrolls once at the end. For very large
  // batches we splice the work across requestAnimationFrame so the
  // event loop can keep painting + responding to input -- this is
  // what makes "switch to User I/O after the log has been running
  // for a while" not freeze the browser.
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
        // Yield to the browser before the next chunk so it can paint
        // and process user input between chunks.
        requestAnimationFrame(step);
      } else {
        stickToBottom();
      }
    };
    step();
  }

  function setPending(on, prompt, id, masked) {
    if (on) {
      pendingId = id;
      promptEl.textContent = prompt || t('userio.input_requested');
      promptEl.classList.add('active');
      input.disabled = false;
      sendBtn.disabled = false;
      // A masked (password) request: hide the typed characters. The
      // input is a <textarea>, so we mask via CSS text-security rather
      // than swapping element types.
      input.classList.toggle('masked', !!masked);
      input.setAttribute('placeholder',
          masked ? t('userio.password_ph') : t('userio.response_ph'));
      if (document.activeElement !== input) { input.focus(); }
    } else {
      pendingId = 0;
      promptEl.textContent = t('userio.waiting');
      promptEl.classList.remove('active');
      input.disabled = true;
      sendBtn.disabled = true;
      input.classList.remove('masked');
      input.setAttribute('placeholder', t('userio.response_ph'));
    }
  }

  async function send() {
    if (!pendingId) { return; }
    const text = input.value;
    const id = pendingId;
    input.value = '';
    autosize();
    // Mark this id answered so the next poll doesn't re-show it before
    // the backend has released the slot (which would steal the next
    // request's turn -- the cause of sequential prompts breaking).
    lastAnsweredId = Math.max(lastAnsweredId, id);
    setPending(false);
    // The backend echoes the answer as an "input" line (so it persists
    // across a re-mount); the tick() below fetches and renders it.
    try {
      await api.ioInput(id, text);
    } catch (e) {
      toast(e.message, 'error');
    }
    tick();   // reflect new state promptly
  }

  function insertNewline() {
    const s = input.selectionStart, e = input.selectionEnd;
    const v = input.value;
    input.value = v.slice(0, s) + '\n' + v.slice(e);
    input.selectionStart = input.selectionEnd = s + 1;
    autosize();
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (input.disabled) { return; }
    // Ctrl+J -> newline (matches the console-style "line feed" key).
    if (e.ctrlKey && (e.key === 'j' || e.key === 'J')) {
      e.preventDefault();
      insertNewline();
      return;
    }
    // Plain Enter sends; a bare textarea would otherwise insert a
    // newline, so suppress that. (Modified Enter is left alone.)
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey
        && !e.metaKey && !e.altKey) {
      e.preventDefault();
      send();
    }
  });

  async function tick() {
    if (stopped) { return; }
    // Pause (don't stop) while detached: the workspace persists across
    // nav switches and re-attaches this same DOM, so resume polling
    // when it comes back. Permanent stop is via cleanup() (pane close).
    if (!document.body.contains(consoleEl)) { return; }
    // Re-attaching this pane (nav switch back) or moving it (a split)
    // zeroes scrollTop; snap back to the tail so we keep showing the
    // latest line. No-op when already at the bottom or unpinned.
    stickToBottom();
    try {
      const c = await api.ioConsole(lastSeq);
      if (c && c.lines) {
        applyBatch(c.lines);
        if (typeof c.latest === 'number') { lastSeq = c.latest; }
      }
      const p = await api.ioPending();
      // Only act on a request we haven't already answered: after
      // submitting id N the backend may still report N as pending for
      // a few ms before its worker releases the slot.
      const fresh = p && p.pending && p.id > lastAnsweredId;
      if (fresh) {
        if (p.id !== pendingId) {
          setPending(true, p.prompt, p.id, p.masked);
        }
      } else if (pendingId) {
        setPending(false);
      }
    } catch (e) {
      // Transient (e.g. auth prompt) -- stay quiet, retry next tick.
    }
  }

  // Defer the first poll to a microtask. When the workspace mounts this
  // pane, its DOM isn't attached to the document yet (the workspace
  // appends the pane AFTER mountUserIo returns). Polling synchronously
  // here would trip tick()'s "navigated away" self-terminate check
  // (consoleEl not yet in the document) and stop the poll forever -- so
  // the console would never receive the session's UI text.
  queueMicrotask(() => { if (!stopped) { tick(); } });
  const timer = setInterval(() => {
    if (stopped) { clearInterval(timer); return; }
    tick();
  }, POLL_MS);

  // Cleanup for the workspace: stop polling when this pane is closed.
  return () => { stopped = true; clearInterval(timer); };
}
