// HLS video pane: picks one of the session's active "hls-broadcast"
// stages and plays its live stream.
//
// Embeddable like the User I/O pane: renders into `body`, puts its
// controls into the caller-provided `actions` (pane header), and
// returns a cleanup that tears the player down. `ctx.stream`, when
// given, skips the picker and plays that stream directly; ctx.onTitle
// lets the host pane reflect the current stream in its title.
//
// Playback: native HLS where the browser supports it (Safari / iOS --
// zero dependencies), otherwise the vendored hls.js
// (web/js/vendor/hls.min.js, served from the doc-root). Both paths work
// fully offline with no external requests. If hls.js fails to load the
// pane shows a direct link to the stream.

import { el, clear } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

// Browser-reachable URL for a stream descriptor. The stage may bind
// 0.0.0.0; the browser should reach the same host it loaded the UI
// from, on the stage's configured port.
function streamUrl(s) {
  let host = (s.bind_address || '').trim();
  if (!host || host === '0.0.0.0' || host === '::'
      || host === 'localhost' || host === '127.0.0.1' || host === '::1') {
    host = window.location.hostname || 'localhost';
  }
  const port = s.port || 8080;
  const name = s.playlist_name || 'stream.m3u8';
  return window.location.protocol + '//' + host + ':' + port + '/' + name;
}

// Lazy hls.js loader for non-native browsers. hls.js 1.5.18 is vendored
// at web/js/vendor/hls.min.js (served from the doc-root), so playback
// works fully offline with no external requests. Cached so concurrent
// panes share one load.
let hlsLoad = null;
function loadHlsJs() {
  if (window.Hls) { return Promise.resolve(); }
  if (hlsLoad) { return hlsLoad; }
  hlsLoad = new Promise((res, rej) => {
    const sc = document.createElement('script');
    sc.src = '/js/vendor/hls.min.js';
    sc.onload = res;
    sc.onerror = () => rej(new Error('hls.js failed to load'));
    document.head.append(sc);
  }).then(() => { if (!window.Hls) { throw new Error('hls.js absent'); } });
  return hlsLoad;
}

export function mountHlsVideo(body, actions, ctx) {
  let destroyed = false;
  let player = null;                       // { destroy() }
  const onTitle = (ctx && ctx.onTitle) || (() => {});

  function teardownPlayer() {
    if (player) {
      try { player.destroy(); } catch (e) {}
      player = null;
    }
  }

  // ---- stream picker ----------------------------------------------
  function showPicker() {
    teardownPlayer();
    onTitle(t('io.hls'));
    clear(body);
    if (actions) { clear(actions); }
    const list = el('div', { class: 'hls-list' });
    if (actions) {
      actions.append(el('button', { class: 'btn ghost', onclick: load },
        t('common.refresh')));
    }
    body.append(el('div', { class: 'hls-picker' },
      el('div', { class: 'hls-picker-title' }, t('hls.select')),
      list));

    function load() {
      clear(list);
      list.append(el('div', { class: 'hls-hint' }, t('common.loading')));
      api.hlsStreams().then((r) => {
        if (destroyed) { return; }
        clear(list);
        const streams = (r && r.streams) ? r.streams : [];
        if (streams.length === 0) {
          list.append(el('div', { class: 'hls-hint' },
            t('hls.no_streams')));
          return;
        }
        for (const s of streams) {
          list.append(el('button', { class: 'hls-item',
            onclick: () => play(s) },
            el('span', { class: 'hls-item-name' },
              s.pipeline + ' / ' + s.stage),
            el('span', { class: 'hls-item-url' }, streamUrl(s))));
        }
      }).catch((e) => {
        if (destroyed) { return; }
        clear(list);
        list.append(el('div', { class: 'hls-hint' },
          t('hls.list_failed', { msg: e.message })));
      });
    }
    load();
  }

  // ---- player ------------------------------------------------------
  function play(s) {
    teardownPlayer();
    clear(body);
    if (actions) { clear(actions); }
    const url = streamUrl(s);
    onTitle(t('hls.playing', { stage: s.stage }));

    // controlsList hides the fullscreen + playback-rate + download
    // entries and disablePictureInPicture drops PiP, which between them
    // empties Chromium's "more" (overflow) menu so it disappears too.
    const video = el('video', { class: 'hls-video',
      controls: '', autoplay: '', playsinline: '',
      controlslist: 'nofullscreen noplaybackrate nodownload',
      disablepictureinpicture: '' });
    video.muted = true;        // required for autoplay in most browsers
    // Auto-unmute when the stream actually carries audio (the stage's audio
    // iport is wired). Start muted so autoplay always begins, then drop the
    // mute once playback starts -- the click that opened this view is the user
    // gesture that lets the browser honor sound. If the browser still refuses
    // (rejects the re-play), fall back to muted so the picture keeps running
    // and the viewer can unmute from the controls. Video-only streams stay
    // muted (nothing to hear).
    if (s.audio) {
      video.addEventListener('playing', function unmuteOnce() {
        video.removeEventListener('playing', unmuteOnce);
        video.muted = false;
        const p = video.play();
        if (p && typeof p.catch === 'function') {
          p.catch(() => { video.muted = true; });
        }
      });
    }
    video.disablePictureInPicture = true;
    // Disable the built-in click/double-click-to-toggle on the video
    // surface: this is a live realtime view, so pausing or scrubbing by
    // clicking the frame makes no sense (and a stray click-pause is one
    // way playback ends up stranded behind live). The native control bar
    // (play / volume / scrubber) still works.
    video.addEventListener('click', (e) => e.preventDefault());
    video.addEventListener('dblclick', (e) => e.preventDefault());
    const status = el('div', { class: 'hls-status' });
    body.append(video, status);

    if (actions) {
      actions.append(
        el('button', { class: 'btn ghost', onclick: showPicker },
          t('hls.change_stream')),
        el('a', { class: 'btn ghost', href: url, target: '_blank' },
          t('common.open')));
    }

    const fail = (msg) => {
      clear(status);
      status.append(msg + ' — ',
        el('a', { href: url, target: '_blank' }, url));
    };
    player = attachPlayer(video, url, fail);
  }

  function attachPlayer(video, url, onError) {
    let hls = null;
    let gone = false;
    let usingNative = false;
    const onNativeError = () => onError(t('hls.playback_error'));

    // The current live-edge target: hls.js's computed live-sync
    // position when available, else the end of the seekable range.
    const liveTarget = () => {
      if (hls && typeof hls.liveSyncPosition === 'number'
          && isFinite(hls.liveSyncPosition)) {
        return hls.liveSyncPosition;
      }
      if (video.seekable && video.seekable.length) {
        return video.seekable.end(video.seekable.length - 1);
      }
      return NaN;
    };

    // Aim a second shy of the reported live edge. Seeking to the exact
    // seekable end can clamp or fail (the edge segment may not be fully
    // appended yet); landing on an unusable position is what leaves
    // playback stuck back at the start. A small margin keeps the seek
    // safely inside the buffered live window.
    const LIVE_SEEK_MARGIN_S = 1;
    const seekNearLive = (t) => {
      const target = Math.max(0, t - LIVE_SEEK_MARGIN_S);
      if (target > video.currentTime) {
        try { video.currentTime = target; } catch (e) {}
      }
    };

    // Jump to the live edge -- used both on initial play and after the
    // pane comes back into view.
    //
    // Two ways a single seek fails: (1) on a fresh stream (or a pane
    // just re-attached, where the browser PAUSED the element and its
    // seekable range went stale), the live position is not known yet --
    // liveTarget() reads NaN -- so there is nothing to seek TO, and
    // playback just runs on from the start; (2) the browser CLAMPS a
    // forward seek to the (stale) seekable end. So: kick the loader,
    // then re-assert the forward seek a few times -- retrying while the
    // live edge is still UNKNOWN (realtime not yet available to the
    // player) as well as while we are still behind it -- until the
    // seekable range catches up and the seek lands. We only ever seek
    // FORWARD (never rewind a viewer already at live) and stop once
    // we're within a couple seconds of the edge.
    //
    // Retries are spaced at 600ms over a generous budget (~30s): a stream
    // that takes well past 5s to start/recover and surface its live edge
    // still gets re-synced once it does, instead of the retries lapsing
    // and leaving playback stranded at the start.
    const RESYNC_RETRY_MS = 600;
    const RESYNC_MAX_TRIES = 50;
    const jumpToLive = () => {
      if (hls) { try { hls.startLoad(); } catch (e) {} }
      let tries = 0;
      const tryOnce = () => {
        if (gone || destroyed) { return; }
        const t = liveTarget();
        const known = isFinite(t) && t > 0;
        if (known && t - video.currentTime > 2) {
          seekNearLive(t);
        }
        const needMore = !known || (t - video.currentTime > 4);
        if (needMore && ++tries < RESYNC_MAX_TRIES) {
          setTimeout(tryOnce, RESYNC_RETRY_MS);
        }
      };
      tryOnce();
      const p = video.play();
      if (p && p.catch) { p.catch(() => {}); }
    };

    // The whole workspace (including this pane) is detached while the
    // user is on another nav view and re-attached on return. Watch for
    // that detached -> attached edge and jump back to live.
    let wasAttached = document.body.contains(video);
    const resync = setInterval(() => {
      if (gone || destroyed) { return; }
      const nowAttached = document.body.contains(video);
      if (nowAttached && !wasAttached) { jumpToLive(); }
      wasAttached = nowAttached;
    }, 1000);

    // Live-latency catch-up. Even while in view the player drifts behind
    // the live edge: each rebuffer/stall leaves currentTime a little
    // further behind the (still-advancing) live-sync position, and
    // hls.js does not reseek on its own. Every few seconds, if we are
    // playing, attached, and have drifted past the live target, seek
    // forward to catch up -- keeping latency tight to realtime. Only
    // ever forward, and only past a threshold so the normal live latency
    // and brief deliberate scrub-backs are left alone (and a hard seek
    // re-syncs to ~0 drift, so steady playback doesn't seek every tick).
    const CATCHUP_PERIOD_MS = 3 * 1000;
    const CATCHUP_DRIFT_S   = 6;
    const catchup = setInterval(() => {
      if (gone || destroyed) { return; }
      if (video.paused) { return; }                    // respect a pause
      if (!document.body.contains(video)) { return; }  // detached -> paused
      const t = liveTarget();
      if (isFinite(t) && t - video.currentTime > CATCHUP_DRIFT_S) {
        seekNearLive(t);
      }
    }, CATCHUP_PERIOD_MS);

    // Native HLS: set src directly and start. Safari/iOS play HLS this
    // way reliably (and with no dependency). play() is explicit because
    // a muted autoplay can still need a nudge under MSE/autoplay rules.
    const useNative = () => {
      usingNative = true;
      video.src = url;
      video.addEventListener('error', onNativeError);
      // Catch up to the live edge once metadata (and the seekable
      // range) is available, retrying if realtime is not surfaced yet.
      video.addEventListener('loadedmetadata', jumpToLive, { once: true });
      const p = video.play();
      if (p && p.catch) { p.catch(() => {}); }
    };

    // hls.js (Chrome / Firefox / Edge): attach media, load on attach,
    // and explicitly play() once the manifest is parsed -- without the
    // play() the element just sits black. Recover transient fatal
    // network / media errors the way the hls.js guide recommends.
    const startHls = () => {
      const H = window.Hls;
      if (!H || !H.isSupported()) {
        if (video.canPlayType('application/vnd.apple.mpegurl')) {
          useNative();
        } else {
          onError(t('hls.no_hls'));
        }
        return;
      }
      // maxLiveSyncPlaybackRate > 1 lets hls.js gently speed playback
      // (up to 1.5x) to ease back toward the live-sync position whenever
      // it falls behind, which keeps latency from accumulating between
      // the periodic hard catch-up seeks below.
      hls = new H({ maxLiveSyncPlaybackRate: 1.5 });
      hls.attachMedia(video);
      hls.on(H.Events.MEDIA_ATTACHED, () => hls.loadSource(url));
      hls.on(H.Events.MANIFEST_PARSED, () => {
        const p = video.play();
        if (p && p.catch) { p.catch(() => {}); }
        // Start at the live edge, retrying until hls.js surfaces the
        // live-sync position (otherwise playback runs on from the
        // start of the first loaded segment).
        jumpToLive();
      });
      let netRetries = 0;
      hls.on(H.Events.ERROR, (_evt, d) => {
        if (!d || !d.fatal) { return; }
        if (d.type === H.ErrorTypes.NETWORK_ERROR && netRetries++ < 3) {
          hls.startLoad();
        } else if (d.type === H.ErrorTypes.MEDIA_ERROR) {
          hls.recoverMediaError();
        } else {
          onError(t('hls.stream_error', { detail: d.details || d.type }));
        }
      });
    };

    // Safari/iOS get native HLS directly. Everywhere else prefer hls.js
    // (its MSE playback is more dependable than Chromium's partial /
    // misreported native HLS, which is the "Open works but the embedded
    // player is black" symptom).
    const nativeOk = !!video.canPlayType('application/vnd.apple.mpegurl');
    const isSafari =
      /^((?!chrome|crios|android|edg).)*safari/i.test(navigator.userAgent);

    if (nativeOk && isSafari) {
      useNative();
    } else {
      loadHlsJs()
        .then(() => { if (!gone && !destroyed) { startHls(); } })
        .catch(() => {
          if (gone || destroyed) { return; }
          if (nativeOk) { useNative(); }
          else { onError(t('hls.load_failed')); }
        });
    }

    return { destroy() {
      gone = true;
      clearInterval(resync);
      clearInterval(catchup);
      if (hls) { try { hls.destroy(); } catch (e) {} }
      if (usingNative) {
        video.removeEventListener('error', onNativeError);
        video.removeAttribute('src');
        video.load();
      }
    } };
  }

  // Entry point.
  if (ctx && ctx.stream) { play(ctx.stream); } else { showPicker(); }

  return () => { destroyed = true; teardownPlayer(); };
}
