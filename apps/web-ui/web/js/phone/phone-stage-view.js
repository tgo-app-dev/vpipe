// Host for a panel that a STAGE ships with itself.
//
// Preview and Compare Images are both embedded in libvpipe next to their
// stage's C++ and discovered at /api/ui/views; the desktop workspace
// mounts them as pane types. Neither needs anything phone-specific --
// they already pick their own stage from a list and size themselves to
// the box they are given -- so the phone side of each is only: find the
// view the backend registered under this id, mount it, and remember what
// it was pointed at. That is this file, once, rather than once per panel.
//
// The panel's config is persisted HERE rather than by the shell, because
// these views are deliberately NOT cached across view switches: they
// hold a live channel (Preview streams video), and a phone should not be
// pulling frames for a view nobody is looking at. Tearing the panel down
// on the way out and restoring its designation on the way back gives the
// same continuity for none of the traffic -- the panel re-states its
// `watch` when the channel opens and the backend replies with
// waiting/playing, so it resumes by itself.

import { el, clear } from '../dom.js';
import { t } from '../i18n.js';
import { listStageViews, stageViewMount } from '../stage-views.js';
import { lastPipeline } from './phone-recent.js';

function readCfg(key) {
  try {
    const s = localStorage.getItem(key);
    const o = s ? JSON.parse(s) : null;
    return (o && typeof o === 'object') ? o : {};
  } catch (e) { return {}; }
}

function writeCfg(key, cfg) {
  try { localStorage.setItem(key, JSON.stringify(cfg || {})); }
  catch (e) { /* storage blocked -- the choice just won't persist */ }
}

// A phone-shell mount function for the stage view registered as
// `viewId`. `labelKey` titles the pane until the panel names itself.
export function stageViewMounter(viewId, labelKey) {
  const store = 'vpipe_phone_view_' + viewId;
  return function mountStageView({ body, actions, setTitle }) {
    clear(body);
    clear(actions);
    setTitle(t(labelKey));

    let inner = null;             // the panel's cleanup, once it is up
    let dead = false;

    const hint = el('div', { class: 'ph-empty' }, t('common.loading'));
    body.append(hint);

    listStageViews().then((views) => {
      if (dead) { return; }
      const v = (views || []).find((x) => x.id === viewId);
      if (!v) {
        // An older backend, or a build without this stage: say so rather
        // than leave an empty panel.
        hint.textContent = t('phone.no_stage_view', { id: viewId });
        return;
      }
      hint.remove();
      const cfg = readCfg(store);
      // A hint, not a setting: which pipeline the operator was last
      // looking at. A panel that has never been pointed at anything can
      // use it to open on a stage from the right pipeline rather than an
      // empty picker; one that already has a designation ignores it. Set
      // fresh on every mount so it never goes stale in the stored
      // config, and cleared when there is nothing to say.
      const last = lastPipeline();
      if (last) { cfg.prefer_pipeline = last; }
      else { delete cfg.prefer_pipeline; }
      inner = stageViewMount(v)(body, actions, cfg, {
        onTitle: (title) => setTitle(title),
        onConfigChange: () => writeCfg(store, cfg),
      });
    }).catch(() => {
      if (!dead) {
        hint.textContent = t('phone.no_stage_view', { id: viewId });
      }
    });

    return () => {
      dead = true;
      if (inner) {
        try { inner(); } catch (e) { /* a bad view can't block a switch */ }
        inner = null;
      }
    };
  };
}
