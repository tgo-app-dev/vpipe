// Phone Files view: browse the session's filesystem namespace and
// preview what's in it.
//
// The desktop file browser is three columns (folder tree | listing |
// preview) and none of that survives a phone, so this is the same
// capability as two SCREENS: the shared one-column walker
// (phone-fs.js -- the same one the pipeline view points at for load and
// save), and a preview that replaces it when a file is tapped and hands
// the walker back on Back. Preview categories come from fs-kinds.js, the
// table the desktop browser reads too, so a format previews the same way
// in both.
//
// Read-only: browsing and preview. Renaming and creating folders are
// still desktop-only -- they are not what a phone is reached for, and
// each needs a text entry over a listing that a phone has no room for.

import { el, clear } from '../dom.js';
import { makeIcon } from '../icons.js';
import { api } from '../api.js';
import { t } from '../i18n.js';
import { categorize } from '../fs-kinds.js';
import { mountFsBrowser, baseOf } from './phone-fs.js';

// One screen of text, Range-capped so a huge file is never pulled whole.
const TEXT_BYTES = 64 * 1024;

const LAST_DIR = 'vpipe_phone_files_dir';

function humanSize(n) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = Number(n) || 0, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(v < 10 ? 1 : 0)) + ' ' + u[i];
}

export function mountPhoneFiles({ body, actions, setTitle }) {
  clear(body);
  clear(actions);
  setTitle(t('nav.files'));

  let start = '';
  try { start = localStorage.getItem(LAST_DIR) || ''; } catch (e) {}

  const browseEl = el('div', { class: 'ph-files-browse' });
  const previewEl = el('div', { class: 'ph-files-preview' });
  previewEl.hidden = true;
  body.append(browseEl, previewEl);

  const backBtn = el('button', { class: 'ph-act', title: t('phone.back'),
    'aria-label': t('phone.back'), onclick: () => showBrowse() }, '‹');
  const refreshBtn = el('button', { class: 'ph-act',
    title: t('common.refresh'), 'aria-label': t('common.refresh'),
    onclick: () => browser.go(browser.path()) }, makeIcon('refresh', 'sm'));
  backBtn.hidden = true;
  actions.append(backBtn, refreshBtn);

  const browser = mountFsBrowser(browseEl, {
    start,
    onDir: (p) => {
      // Come back to where you left off next time the view is opened.
      try { localStorage.setItem(LAST_DIR, p); } catch (e) {}
      setTitle(baseOf(p) || t('nav.files'));
    },
    onFile: (p, e) => showPreview(p, e),
  });

  function showBrowse() {
    // Drop the media element rather than merely hiding it: a hidden
    // <video>/<audio> keeps its buffer (and can keep playing).
    clear(previewEl);
    previewEl.hidden = true;
    browseEl.hidden = false;
    backBtn.hidden = true;
    refreshBtn.hidden = false;
    setTitle(baseOf(browser.path()) || t('nav.files'));
  }

  function showPreview(path, entry) {
    browseEl.hidden = true;
    previewEl.hidden = false;
    backBtn.hidden = false;
    refreshBtn.hidden = true;
    setTitle(baseOf(path));
    clear(previewEl);

    const meta = el('div', { class: 'ph-file-meta' },
      el('span', { class: 'ph-file-name' }, baseOf(path)),
      el('span', { class: 'ph-file-size' },
         entry && entry.size != null ? humanSize(entry.size) : ''));
    // A media element cannot set headers, so the access key rides in the
    // URL -- the same way the file browser and the Preview socket do it.
    const url = api.fsFileUrl(path);
    const raw = el('a', { class: 'btn ghost ph-file-raw', href: url,
      target: '_blank', rel: 'noopener' }, t('phone.open_raw'));
    previewEl.append(meta);

    const kind = categorize(baseOf(path));
    const stage = el('div', { class: 'ph-file-stage' });
    previewEl.append(stage, raw);

    if (kind === 'image') {
      stage.append(el('img', { class: 'ph-file-img', src: url, alt: '' }));
    } else if (kind === 'audio') {
      stage.append(el('audio', { class: 'ph-file-av', src: url,
        controls: true, preload: 'metadata' }));
    } else if (kind === 'video') {
      stage.append(el('video', { class: 'ph-file-av', src: url,
        controls: true, playsinline: '', preload: 'metadata' }));
    } else if (kind === 'text') {
      const pre = el('pre', { class: 'ph-file-text allow-context-menu' },
        t('common.loading'));
      stage.append(pre);
      api.fsText(path, TEXT_BYTES).then((r) => {
        if (!pre.isConnected) { return; }
        clear(pre);
        pre.append(document.createTextNode(r.text || ''));
        if (r.truncated) {
          previewEl.insertBefore(
            el('div', { class: 'ph-sheet-hint' },
               t('phone.text_truncated',
                 { shown: humanSize(TEXT_BYTES), total: humanSize(r.total) })),
            raw);
        }
      }).catch((e) => {
        if (pre.isConnected) { pre.textContent = e.message; }
      });
    } else {
      stage.append(el('div', { class: 'ph-empty' },
        t('phone.no_preview_kind')));
    }
  }

  return () => { clear(previewEl); };
}
