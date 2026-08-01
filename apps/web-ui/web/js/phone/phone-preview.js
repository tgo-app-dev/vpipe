// Phone preview view.
//
// The panel is the one the PREVIEW STAGE ships with itself (embedded in
// libvpipe, discovered at /api/ui/views) -- the same module the desktop
// workspace mounts. It already does what the phone needs: pick a stage
// from the list of every declared preview stage, then play whatever that
// stage produces (fMP4 video through MSE, PNG stills, PCM through
// WebAudio), with pan/zoom over the frame. So this file only locates the
// view and mounts it; nothing about the media path is phone-specific.
//
// The panel's config (which stage it follows) is persisted HERE rather
// than left to the shell, because this view is deliberately not cached:
// its channel streams video, and a phone should not be pulling frames
// for a view nobody is looking at. Tearing it down on the way out and
// restoring the designation on the way back gives the same continuity
// for none of the traffic -- the panel re-states its `watch` on open and
// the backend replies with waiting/playing, so it resumes by itself.

import { el, clear } from '../dom.js';
import { t } from '../i18n.js';
import { listStageViews, stageViewMount } from '../stage-views.js';

const CFG_KEY = 'vpipe_phone_preview';

function readCfg() {
  try {
    const s = localStorage.getItem(CFG_KEY);
    const o = s ? JSON.parse(s) : null;
    return (o && typeof o === 'object') ? o : {};
  } catch (e) { return {}; }
}

function writeCfg(cfg) {
  try { localStorage.setItem(CFG_KEY, JSON.stringify(cfg || {})); }
  catch (e) { /* storage blocked -- the choice just won't persist */ }
}

export function mountPhonePreview({ body, actions, setTitle }) {
  clear(body);
  clear(actions);
  setTitle(t('phone.preview'));

  let inner = null;                 // the stage view's cleanup, once up
  let dead = false;

  const hint = el('div', { class: 'ph-empty' }, t('common.loading'));
  body.append(hint);

  listStageViews().then((views) => {
    if (dead) { return; }
    const v = (views || []).find(
      (x) => x.id === 'preview' || x.stage_type === 'preview');
    if (!v) {
      hint.textContent = t('phone.no_preview');
      return;
    }
    hint.remove();
    const cfg = readCfg();
    inner = stageViewMount(v)(body, actions, cfg, {
      onTitle: (title) => setTitle(title),
      onConfigChange: () => writeCfg(cfg),
    });
  }).catch(() => {
    if (!dead) { hint.textContent = t('phone.no_preview'); }
  });

  return () => {
    dead = true;
    if (inner) {
      try { inner(); } catch (e) { /* a bad view can't block the switch */ }
      inner = null;
    }
  };
}
