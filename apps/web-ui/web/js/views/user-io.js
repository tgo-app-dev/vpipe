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

// Persisted preference for the "Thinking" toggle (off by default: the
// model's reasoning segments collapse to a 💭 glyph).
const THINK_KEY = 'vpipe_io_thinking';
function readThinkPref() {
  try { return localStorage.getItem(THINK_KEY) === '1'; }
  catch (e) { return false; }
}

// Unified vpipe thinking markers. The backend detokenizers rewrite
// every model family's reasoning begin/end tokens (Qwen <think>,
// Gemma channel tokens) to this single pair, so the client only ever
// sees these.
const THINK_START = '<|__vpipe_think_start__|>';
const THINK_END = '<|__vpipe_think_end__|>';

// Split a line's text into ordered {think, text} segments. A close
// marker with no preceding open means the message BEGAN inside a
// thinking block (shouldn't normally happen — the backend emits the
// start marker — but render it sanely); an open with no close means
// thinking runs to end-of-message (mid-stream, or a truncated turn).
function splitThinking(text) {
  const segs = [];
  let pos = 0;
  while (pos < text.length) {
    const s = text.indexOf(THINK_START, pos);
    const e = text.indexOf(THINK_END, pos);
    if (s < 0 && e < 0) {
      segs.push({ think: false, text: text.slice(pos) });
      break;
    }
    if (e >= 0 && (s < 0 || e < s)) {
      segs.push({ think: true, text: text.slice(pos, e) });
      pos = e + THINK_END.length;
      continue;
    }
    if (s > pos) { segs.push({ think: false, text: text.slice(pos, s) }); }
    const e2 = text.indexOf(THINK_END, s + THINK_START.length);
    if (e2 < 0) {
      segs.push({ think: true, text: text.slice(s + THINK_START.length) });
      break;
    }
    segs.push({ think: true,
                text: text.slice(s + THINK_START.length, e2) });
    pos = e2 + THINK_END.length;
  }
  return segs;
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

  // Media attach controls, shown only while a getmedialine() request is
  // pending (pending.media). Attached/dropped files are read as base64
  // and held aside in `attachments`; a short visible placeholder
  // (⟦img1⟧ / ⟦aud1⟧) marks the insertion point in the textarea and is
  // expanded to the full media-line marker on send. Deleting the
  // placeholder text drops the attachment.
  const attachImgBtn = el('button',
      { class: 'btn ghost', title: t('userio.attach_image') }, '🖼');
  const attachAudBtn = el('button',
      { class: 'btn ghost', title: t('userio.attach_audio') }, '🔊');
  const fileImg = el('input', { type: 'file', accept: 'image/*' });
  const fileAud = el('input', { type: 'file', accept: 'audio/*' });
  fileImg.multiple = true;
  fileAud.multiple = true;
  for (const n of [attachImgBtn, attachAudBtn]) { n.style.display = 'none'; }
  for (const n of [fileImg, fileAud]) { n.style.display = 'none'; }

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

  // Thinking toggle: reveal or collapse the model's reasoning
  // segments (unified thinking markers in the streamed text).
  let thinkEnabled = readThinkPref();
  const thinkCheck = el('input', { type: 'checkbox' });
  thinkCheck.checked = thinkEnabled;
  const thinkToggle = el('label', { class: 'io-md-toggle',
                                    title: t('userio.thinking_title') },
                         thinkCheck, t('userio.thinking'));
  thinkCheck.addEventListener('change', () => {
    thinkEnabled = thinkCheck.checked;
    try { localStorage.setItem(THINK_KEY, thinkEnabled ? '1' : '0'); }
    catch (e) { /* storage blocked -- preference just won't persist */ }
    rerenderAll();
  });

  const hint = el('div', { class: 'io-hint' },
    el('kbd', { class: 'kbd' }, 'Ctrl+J'), t('userio.newline') + '  ·  ',
    el('kbd', { class: 'kbd' }, '⏎'), t('userio.send_word'));

  const root = el('div', { class: 'userio' },
    consoleEl,
    el('div', { class: 'io-input' }, promptEl, input,
       attachImgBtn, attachAudBtn, fileImg, fileAud, sendBtn),
    hint);
  body.append(root);
  // The pane header (owned by the workspace) carries the Thinking +
  // Markdown toggles + Clear button.
  if (actions) { actions.append(thinkToggle, mdToggle, clearBtn); }

  // Grow the textarea with its content up to a cap, then scroll.
  function autosize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  }

  let lastSeq = 0;
  let pendingId = 0;        // id currently shown in the input box (0 = none)
  let lastAnsweredId = 0;   // highest request id we've already answered
  let mediaPending = false; // current request accepts media attachments
  let stopped = false;
  const lineEls = new Map();   // seq -> element

  // Pending attachments: placeholder text -> {kind:'im'|'au', size, b64}.
  // `size` is the file's decoded byte count (the marker LENGTH field).
  let attachSeq = 0;
  const attachments = new Map();

  function insertAtCursor(text) {
    const s = input.selectionStart, e = input.selectionEnd;
    const v = input.value;
    input.value = v.slice(0, s) + text + v.slice(e);
    input.selectionStart = input.selectionEnd = s + text.length;
    autosize();
    input.focus();
  }

  function addAttachment(file, kind) {
    const reader = new FileReader();
    reader.onerror = () => {
      toast(t('userio.attach_failed', { name: file.name }), 'error');
    };
    reader.onload = () => {
      // readAsDataURL yields "data:<mime>;base64,<data>"; keep the tail.
      const url = String(reader.result);
      const b64 = url.slice(url.indexOf(',') + 1);
      attachSeq += 1;
      const ph = (kind === 'im' ? '⟦img' : '⟦aud') + attachSeq + '⟧';
      attachments.set(ph, { kind, size: file.size, b64 });
      insertAtCursor(ph);
    };
    reader.readAsDataURL(file);
  }

  // Route a file by MIME type; non-media files are rejected with a toast.
  function addAttachmentAuto(file) {
    const ty = file.type || '';
    if (ty.startsWith('image/')) { addAttachment(file, 'im'); }
    else if (ty.startsWith('audio/')) { addAttachment(file, 'au'); }
    else { toast(t('userio.attach_unsupported', { name: file.name }),
                 'error'); }
  }

  // Expand each still-present placeholder to its wire marker
  // (<|__vpipe_base64_im_start__|>LENGTH,DATA<|__vpipe_base64_im_end__|>).
  // Placeholders the user deleted just drop their attachment.
  function expandAttachments(text) {
    let out = text;
    for (const [ph, a] of attachments) {
      const tag = a.kind === 'im' ? 'im' : 'au';
      const marker = '<|__vpipe_base64_' + tag + '_start__|>'
          + a.size + ',' + a.b64
          + '<|__vpipe_base64_' + tag + '_end__|>';
      out = out.split(ph).join(marker);
    }
    return out;
  }

  function clearAttachments() {
    attachments.clear();
  }

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
  // Thinking segments (unified markers) render dimmed when the
  // Thinking toggle is on, and collapse to a bare 💭 when off; the
  // non-thinking remainder honours the Markdown toggle as before.
  function renderLineBody(node, l) {
    node.classList.toggle('md', mdEnabled);
    const hasThink = l.text.indexOf(THINK_START) >= 0
        || l.text.indexOf(THINK_END) >= 0;
    if (!hasThink) {
      if (mdEnabled) {
        node.replaceChildren(renderMarkdown(l.text));
      } else {
        node.textContent = l.text;
      }
      return;
    }
    const parts = [];
    for (const seg of splitThinking(l.text)) {
      if (seg.think) {
        if (thinkEnabled) {
          parts.push(el('span', { class: 'thinking' },
                        '💭 ' + seg.text));
        } else {
          parts.push(el('span', { class: 'thinking-hidden',
                                  title: t('userio.thinking_hidden') },
                        '💭'));
        }
      } else if (seg.text) {
        parts.push(mdEnabled ? renderMarkdown(seg.text)
                             : document.createTextNode(seg.text));
      }
    }
    node.replaceChildren(...parts);
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

  function setPending(on, prompt, id, masked, media) {
    if (on) {
      pendingId = id;
      mediaPending = !!media;
      promptEl.textContent = prompt || t('userio.input_requested');
      promptEl.classList.add('active');
      input.disabled = false;
      sendBtn.disabled = false;
      // A masked (password) request: hide the typed characters. The
      // input is a <textarea>, so we mask via CSS text-security rather
      // than swapping element types.
      input.classList.toggle('masked', !!masked);
      input.setAttribute('placeholder',
          masked ? t('userio.password_ph')
                 : (media ? t('userio.media_ph')
                          : t('userio.response_ph')));
      // Media request: surface the attach buttons (drag-and-drop onto
      // the textarea is armed via mediaPending).
      attachImgBtn.style.display = media ? '' : 'none';
      attachAudBtn.style.display = media ? '' : 'none';
      if (document.activeElement !== input) { input.focus(); }
    } else {
      pendingId = 0;
      mediaPending = false;
      promptEl.textContent = t('userio.waiting');
      promptEl.classList.remove('active');
      input.disabled = true;
      sendBtn.disabled = true;
      input.classList.remove('masked');
      input.setAttribute('placeholder', t('userio.response_ph'));
      attachImgBtn.style.display = 'none';
      attachAudBtn.style.display = 'none';
      clearAttachments();
    }
  }

  async function send() {
    if (!pendingId) { return; }
    // Expand attachment placeholders to their wire markers just before
    // shipping; the console echo comes back from the backend with the
    // markers already compressed to glyphs.
    const text = expandAttachments(input.value);
    const id = pendingId;
    input.value = '';
    clearAttachments();
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
  attachImgBtn.addEventListener('click', () => fileImg.click());
  attachAudBtn.addEventListener('click', () => fileAud.click());
  fileImg.addEventListener('change', () => {
    for (const f of fileImg.files) { addAttachment(f, 'im'); }
    fileImg.value = '';
  });
  fileAud.addEventListener('change', () => {
    for (const f of fileAud.files) { addAttachment(f, 'au'); }
    fileAud.value = '';
  });
  // Drag-and-drop onto the input row while a media request is pending.
  // dragover must be cancelled for the drop to be allowed at all.
  for (const zone of [input, promptEl]) {
    zone.addEventListener('dragover', (e) => {
      if (mediaPending) { e.preventDefault(); }
    });
    zone.addEventListener('drop', (e) => {
      if (!mediaPending) { return; }
      e.preventDefault();
      const files = (e.dataTransfer && e.dataTransfer.files) || [];
      for (const f of files) { addAttachmentAuto(f); }
    });
  }
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (input.disabled) { return; }
    // Backspace/Delete touching an attachment placeholder (⟦img1⟧ /
    // ⟦aud1⟧) removes the placeholder WHOLE -- it stands for one
    // attachment, so it behaves as a single character. Applies when
    // the character being deleted lies anywhere inside a placeholder
    // (boundary or interior); range selections keep the default
    // behavior. Unmatched (user-mangled) text falls through to normal
    // per-character editing.
    if ((e.key === 'Backspace' || e.key === 'Delete')
        && !e.ctrlKey && !e.metaKey && !e.altKey
        && input.selectionStart === input.selectionEnd) {
      const v = input.value;
      const pos = input.selectionStart;
      // Index of the character the key would delete.
      const target = e.key === 'Backspace' ? pos - 1 : pos;
      if (target >= 0 && target < v.length) {
        outer:
        for (const ph of attachments.keys()) {
          for (let i = v.indexOf(ph); i >= 0; i = v.indexOf(ph, i + 1)) {
            if (target >= i && target < i + ph.length) {
              e.preventDefault();
              input.value = v.slice(0, i) + v.slice(i + ph.length);
              input.selectionStart = input.selectionEnd = i;
              autosize();
              break outer;
            }
            if (i > target) { break; }   // occurrences are in order
          }
        }
      }
      if (e.defaultPrevented) { return; }
    }
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
          setPending(true, p.prompt, p.id, p.masked, p.media);
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
