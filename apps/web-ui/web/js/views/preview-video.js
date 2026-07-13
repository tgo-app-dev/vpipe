// Live Preview pane: picks one of the session's active "preview" stages
// and plays its low-latency stream.
//
// Transport: a WebSocket carrying the backend's custom message protocol
// (see common/preview-channel.h) -- a config header, an fMP4 init segment,
// then fMP4 media fragments (video) and planar-float PCM (audio). Video is
// played through Media Source Extensions; audio through WebAudio. MSE is
// NOT gated to secure contexts, so this works over plain-HTTP LAN origins
// with no HTTPS (unlike WebCodecs). No external dependencies.
//
// Embeddable like the other panes: renders into `body`, puts controls into
// `actions`, returns a cleanup that tears the player down. `ctx.stream`
// skips the picker; `ctx.onTitle` reflects the current stream in the title.

import { el, clear } from '../dom.js';
import { api } from '../api.js';
import { t } from '../i18n.js';

export function mountPreview(body, actions, ctx) {
  let destroyed = false;
  let player = null;                        // { destroy() }
  const onTitle = (ctx && ctx.onTitle) || (() => {});

  function teardownPlayer() {
    if (player) {
      try { player.destroy(); } catch (e) {}
      player = null;
    }
  }

  // ---- stream picker ------------------------------------------------
  function showPicker() {
    teardownPlayer();
    onTitle(t('io.preview'));
    clear(body);
    if (actions) { clear(actions); }
    const list = el('div', { class: 'preview-list' });
    if (actions) {
      actions.append(el('button', { class: 'btn ghost', onclick: load },
        t('common.refresh')));
    }
    body.append(el('div', { class: 'preview-picker' },
      el('div', { class: 'preview-picker-title' }, t('preview.select')),
      list));

    function load() {
      clear(list);
      list.append(el('div', { class: 'preview-hint' }, t('common.loading')));
      api.previewStreams().then((r) => {
        if (destroyed) { return; }
        clear(list);
        const streams = (r && r.streams) ? r.streams : [];
        if (streams.length === 0) {
          list.append(el('div', { class: 'preview-hint' },
            t('preview.no_streams')));
          return;
        }
        for (const s of streams) {
          const label = s.title || s.stage;
          const dims = (s.width && s.height)
            ? (s.width + '×' + s.height) : '';
          const kind = [s.video ? 'video' : '', s.audio ? 'audio' : '']
            .filter(Boolean).join(' + ');
          const sub = [s.pipeline + ' / ' + s.stage, dims, kind]
            .filter(Boolean).join('  ·  ');
          list.append(el('button', { class: 'preview-item',
            onclick: () => play(s) },
            el('span', { class: 'preview-item-name' }, label),
            el('span', { class: 'preview-item-sub' }, sub)));
        }
      }).catch((e) => {
        if (destroyed) { return; }
        clear(list);
        list.append(el('div', { class: 'preview-hint' },
          t('preview.list_failed', { msg: e.message })));
      });
    }
    load();
  }

  // ---- player -------------------------------------------------------
  function play(s) {
    teardownPlayer();
    clear(body);
    if (actions) { clear(actions); }
    onTitle(t('preview.playing', { stage: s.title || s.stage }));

    const video = el('video', { class: 'preview-video',
      autoplay: '', playsinline: '', muted: '' });
    video.muted = true;                 // audio is played via WebAudio, not
    const status = el('div', { class: 'preview-status' });
    body.append(video, status);

    if (actions) {
      actions.append(
        el('button', { class: 'btn ghost', onclick: showPicker },
          t('preview.change_stream')));
    }

    player = attachPlayer(video, status, api.previewWsUrl(s.pipeline, s.stage));
  }

  function attachPlayer(video, status, wsUrl) {
    if (typeof window.MediaSource === 'undefined') {
      status.textContent = t('preview.unsupported');
      return { destroy() {} };
    }

    let gone = false;
    let ws = null;
    let ms = null;            // MediaSource
    let sb = null;            // SourceBuffer
    let mime = null;          // 'video/mp4; codecs="avc1.xxxx"'
    let objUrl = null;
    let queue = [];           // pending fMP4 fragment buffers
    let reconnectTimer = null;

    let audioCtx = null;
    let audioCh = 0;
    let audioRate = 0;
    let audioCursor = 0;

    const setStatus = (m) => { if (!gone) { status.textContent = m; } };
    setStatus(t('preview.connecting'));

    // A user gesture is needed to start a suspended AudioContext; resume on
    // a click if autoplay blocked it.
    const resumeAudio = () => {
      if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume().then(() => { if (!gone) { setStatus(''); } })
          .catch(() => {});
      }
    };
    video.addEventListener('click', resumeAudio);

    // ---- Media Source Extensions (video) ---------------------------
    function teardownMse() {
      if (sb) {
        try { sb.removeEventListener('updateend', pump); } catch (e) {}
        try { if (ms && ms.readyState === 'open') { ms.removeSourceBuffer(sb); } }
        catch (e) {}
        sb = null;
      }
      if (objUrl) { try { URL.revokeObjectURL(objUrl); } catch (e) {} objUrl = null; }
      ms = null;
      queue = [];
    }

    function buildMse(initBytes) {
      teardownMse();
      if (!mime) { return; }              // config must arrive first
      ms = new MediaSource();
      objUrl = URL.createObjectURL(ms);
      video.src = objUrl;
      ms.addEventListener('sourceopen', () => {
        if (gone || !ms || ms.readyState !== 'open') { return; }
        try {
          sb = ms.addSourceBuffer(mime);
        } catch (e) { setStatus(t('preview.mse_error')); return; }
        sb.addEventListener('updateend', pump);
        try { sb.appendBuffer(initBytes); }
        catch (e) { queue.unshift(initBytes); }
      }, { once: true });
      const p = video.play();
      if (p && p.catch) { p.catch(() => {}); }
    }

    function pump() {
      if (gone || !sb || sb.updating || queue.length === 0) { return; }
      const buf = queue.shift();
      try {
        sb.appendBuffer(buf);
      } catch (e) {
        if (e && e.name === 'QuotaExceededError') {
          trimBuffer(true);
          queue.unshift(buf);          // retry after the pending remove()
        }
      }
    }

    function trimBuffer(aggressive) {
      if (!sb || sb.updating) { return; }
      try {
        if (sb.buffered.length) {
          const start = sb.buffered.start(0);
          const keepFrom = Math.max(
            start, video.currentTime - (aggressive ? 2 : 12));
          if (keepFrom > start + 0.5) { sb.remove(start, keepFrom); }
        }
      } catch (e) {}
    }

    function seekLive() {
      try {
        if (video.buffered.length) {
          const end = video.buffered.end(video.buffered.length - 1);
          if (end - video.currentTime > 1.5) {
            video.currentTime = Math.max(video.currentTime, end - 0.4);
          }
        }
      } catch (e) {}
      if (video.paused) {
        const p = video.play();
        if (p && p.catch) { p.catch(() => {}); }
      }
    }

    function onFragment(buf) {
      queue.push(buf);
      pump();
      seekLive();
    }

    // Keep latency + memory bounded even while playing steadily.
    const maint = setInterval(() => {
      if (gone) { return; }
      trimBuffer(false);
      seekLive();
    }, 3000);

    // ---- audio (WebAudio) ------------------------------------------
    function onAudio(buf) {
      if (!audioCtx || !audioCh || !audioRate) { return; }
      const frames = Math.floor(buf.byteLength / (audioCh * 4));
      if (frames <= 0) { return; }
      const f32 = new Float32Array(buf);
      let ab;
      try { ab = audioCtx.createBuffer(audioCh, frames, audioRate); }
      catch (e) { return; }
      for (let c = 0; c < audioCh; c++) {
        ab.getChannelData(c).set(f32.subarray(c * frames, (c + 1) * frames));
      }
      const src = audioCtx.createBufferSource();
      src.buffer = ab;
      src.connect(audioCtx.destination);
      const now = audioCtx.currentTime;
      if (audioCursor < now + 0.05) { audioCursor = now + 0.05; }
      try { src.start(audioCursor); } catch (e) {}
      audioCursor += frames / audioRate;
    }

    // ---- config ----------------------------------------------------
    function onConfig(j) {
      if (j.video && j.video.codec) {
        mime = 'video/mp4; codecs="' + j.video.codec + '"';
      }
      if (j.audio) {
        audioCh = j.audio.channels || 1;
        audioRate = j.audio.sampleRate || 48000;
        if (!audioCtx) {
          const AC = window.AudioContext || window.webkitAudioContext;
          if (AC) {
            audioCtx = new AC();
            audioCursor = 0;
            if (audioCtx.state === 'suspended') {
              audioCtx.resume().catch(() => {});
            }
          }
        }
      }
    }

    // ---- message dispatch ------------------------------------------
    function onMessage(ev) {
      const data = ev.data;
      if (!(data instanceof ArrayBuffer) || data.byteLength < 1) { return; }
      const type = new Uint8Array(data, 0, 1)[0];
      const payload = data.slice(1);       // ArrayBuffer of the payload
      if (type === 1) {                    // config (JSON)
        let j;
        try { j = JSON.parse(new TextDecoder().decode(payload)); }
        catch (e) { return; }
        onConfig(j);
      } else if (type === 2) {             // fMP4 init segment
        buildMse(payload);                 // (re)builds MediaSource
        setStatus('');
      } else if (type === 3) {             // fMP4 media fragment
        onFragment(payload);
      } else if (type === 4) {             // PCM audio
        onAudio(payload);
      }
    }

    // ---- WebSocket lifecycle ---------------------------------------
    function connect() {
      if (gone) { return; }
      try {
        ws = new WebSocket(wsUrl);
      } catch (e) { scheduleReconnect(); return; }
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { if (!gone) { setStatus(''); } };
      ws.onmessage = onMessage;
      ws.onerror = () => {};
      ws.onclose = () => {
        if (gone) { return; }
        teardownMse();                     // next connect re-seeds init
        scheduleReconnect();
      };
    }

    function scheduleReconnect() {
      if (gone || reconnectTimer) { return; }
      setStatus(t('preview.reconnecting'));
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, 1500);
    }

    connect();

    return { destroy() {
      gone = true;
      clearInterval(maint);
      if (reconnectTimer) { clearTimeout(reconnectTimer); }
      video.removeEventListener('click', resumeAudio);
      if (ws) { try { ws.onclose = null; ws.close(); } catch (e) {} }
      teardownMse();
      try { video.removeAttribute('src'); video.load(); } catch (e) {}
      if (audioCtx) { try { audioCtx.close(); } catch (e) {} }
    } };
  }

  // Entry point.
  if (ctx && ctx.stream) { play(ctx.stream); } else { showPicker(); }

  return () => { destroyed = true; teardownPlayer(); };
}
