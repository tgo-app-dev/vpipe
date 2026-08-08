// One file preview, shared by the desktop file-browser view and the
// open/save dialog's side panel.
//
// It lives on its own for the same reason fs-kinds.js does: a preview
// implemented twice is a file that renders in the browser and offers a
// bare download in the dialog, with nothing to say why. The dialog grew
// a preview panel after this view already had one, so the choice was to
// share this or to copy it.
//
// The `fb-` class names are kept rather than renamed: those rules are
// already unscoped in app.css (they never depended on a `.fb` ancestor),
// so both hosts pick them up as-is and there is no styling fork.
//
// Every category gets a Download button, not just the ones that cannot
// be previewed -- being able to see a file is a poor reason to be unable
// to fetch it.

import { el, clear } from './dom.js';
import { makeIcon } from './icons.js';
import { api } from './api.js';
import { t } from './i18n.js';
import { humanSize, joinPath } from './fs-list.js';
import { categorize } from './fs-kinds.js';

// One screen of text: bytes fetched for a text preview (Range-capped so
// a huge file is never pulled whole).
const TEXT_PREVIEW_BYTES = 64 * 1024;
// A pipeline summary reads only the head of the file: the stage list of
// even a large graph is far inside this, and a truncated read still
// parses far enough to be useful (see summarizePipeline).
const PIPELINE_BYTES = 256 * 1024;

// A pipeline spec is JSON, so it WOULD render as text -- but the useful
// thing about one at a glance is its stage list, not its punctuation.
export function isPipeline(name) {
  return /\.vpipeline$/i.test(String(name || ''));
}

// [{id, type}] from a pipeline spec, or null when it will not parse.
// Tolerates the two shapes a spec has been written in (`stages` array,
// or a bare object map) so an older file still summarizes.
export function summarizePipeline(text) {
  let doc;
  try { doc = JSON.parse(text); }
  catch (e) { return null; }
  if (!doc || typeof doc !== 'object') { return null; }
  const out = [];
  if (Array.isArray(doc.stages)) {
    for (const s of doc.stages) {
      if (s && typeof s === 'object') {
        out.push({ id: String(s.id || ''), type: String(s.type || '') });
      }
    }
  } else if (doc.stages && typeof doc.stages === 'object') {
    for (const [k, v] of Object.entries(doc.stages)) {
      out.push({ id: k, type: String((v && v.type) || '') });
    }
  } else {
    return null;
  }
  return out;
}

// `onError` (optional) is called when the host wants to know a preview
// failed; the panel itself already degrades to the download offer.
export function createFsPreview() {
  const root = el('div', { class: 'fb-preview-host' });
  // Guards a late async fill (text / pipeline body) against a selection
  // that has moved on since the request went out.
  let gen = 0;

  function empty(msg) {
    ++gen;
    clear(root).append(el('div', { class: 'fb-preview-empty' },
      msg || t('fb.pick_to_preview')));
  }

  // name · size, with Download always on the right.
  function metaRow(entry, url) {
    return el('div', { class: 'fb-file-meta' },
      el('span', { class: 'fb-file-name' },
        entry.name + '  ·  ' + humanSize(entry.size || 0)),
      el('a', { class: 'btn ghost mini fb-dl', href: url,
                download: entry.name, title: t('fb.download') },
        makeIcon('load', 'sm'), el('span', {}, t('fb.download'))));
  }

  // The pixel dimensions, under the media. Written once the element
  // knows them -- an <img> only has naturalWidth after it decodes, and a
  // <video> only has videoWidth after metadata arrives, so neither can
  // be read at construction.
  function dimsLine() {
    return el('div', { class: 'fb-dims' });
  }
  const setDims = (node, w, h) => {
    node.textContent = (w && h) ? (w + ' x ' + h) : '';
  };

  function unpreviewable(entry, url, why) {
    clear(root).append(el('div', { class: 'fb-preview-inner' },
      metaRow(entry, url),
      el('div', { class: 'fb-preview-empty' },
        el('div', { class: 'fb-no-preview' }, why || t('fb.no_preview')))));
  }

  // `dir` + `entry` name the file; entry is {name, size}.
  function show(dir, entry) {
    const mine = ++gen;
    if (!entry || entry.dir) { empty(); return; }
    const vpath = joinPath(dir, entry.name);
    const url = api.fsFileUrl(vpath);
    const cat = categorize(entry.name);
    clear(root);

    if (isPipeline(entry.name)) {
      const list = el('div', { class: 'fb-pl-list' }, t('common.loading'));
      root.append(el('div', { class: 'fb-preview-inner' },
        metaRow(entry, url), list));
      api.fsText(vpath, PIPELINE_BYTES).then((res) => {
        if (mine !== gen) { return; }
        const stages = summarizePipeline(res.text);
        clear(list);
        if (!stages) {
          // Unparseable (or truncated mid-document): say so and show the
          // bytes, which is more use than an empty panel.
          list.append(el('div', { class: 'fb-note' },
            t('fb.pl_unparsed')));
          list.append(el('pre', { class: 'fb-text allow-context-menu' },
            res.text));
          return;
        }
        list.append(el('div', { class: 'fb-pl-count' },
          t('fb.pl_stages', { n: stages.length })));
        for (const s of stages) {
          list.append(el('div', { class: 'fb-pl-row' },
            el('span', { class: 'fb-pl-id' }, s.id || '—'),
            el('span', { class: 'fb-pl-type' },
               s.type ? '[' + s.type + ']' : '')));
        }
      }).catch((err) => {
        if (mine !== gen) { return; }
        clear(list).append(el('div', { class: 'fb-note' },
          t('fb.preview_failed', { msg: err.message })));
      });
      return;
    }

    if (cat === 'image') {
      const dims = dimsLine();
      const img = el('img', { class: 'fb-img', alt: entry.name, src: url });
      img.addEventListener('load',
        () => setDims(dims, img.naturalWidth, img.naturalHeight));
      img.addEventListener('error', () => unpreviewable(entry, url));
      root.append(el('div', { class: 'fb-preview-inner' },
        metaRow(entry, url),
        el('div', { class: 'fb-media-wrap' }, img), dims));

    } else if (cat === 'video') {
      const dims = dimsLine();
      const v = el('video', { class: 'fb-video', controls: true,
        preload: 'metadata', src: url });
      v.addEventListener('loadedmetadata',
        () => setDims(dims, v.videoWidth, v.videoHeight));
      v.addEventListener('error', () => unpreviewable(entry, url));
      root.append(el('div', { class: 'fb-preview-inner' },
        metaRow(entry, url),
        el('div', { class: 'fb-media-wrap' }, v), dims));

    } else if (cat === 'audio') {
      const a = el('audio', { class: 'fb-audio', controls: true,
        preload: 'metadata', src: url });
      a.addEventListener('error', () => unpreviewable(entry, url));
      root.append(el('div', { class: 'fb-preview-inner audio' },
        metaRow(entry, url),
        el('div', { class: 'fb-media-wrap audio' }, a)));

    } else if (cat === 'text') {
      const pre = el('pre', { class: 'fb-text allow-context-menu' },
        t('common.loading'));
      const note = el('div', { class: 'fb-note', hidden: true });
      root.append(el('div', { class: 'fb-preview-inner' },
        metaRow(entry, url), note, pre));
      api.fsText(vpath, TEXT_PREVIEW_BYTES).then((res) => {
        if (mine !== gen) { return; }
        clear(pre).append(document.createTextNode(res.text));
        if (res.truncated) {
          note.hidden = false;
          note.textContent = t('fb.truncated',
            { n: humanSize(TEXT_PREVIEW_BYTES) });
        }
      }).catch((err) => {
        if (mine !== gen) { return; }
        clear(pre).append(document.createTextNode(
          t('fb.preview_failed', { msg: err.message })));
      });

    } else {
      unpreviewable(entry, url);
    }
  }

  return { el: root, show, empty };
}
