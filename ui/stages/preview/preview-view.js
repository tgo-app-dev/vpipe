// Live Preview panel -- the GUI view the "preview" stage ships with
// itself.
//
// This module is embedded in libvpipe next to the stage's C++ and served
// to the browser by whatever front end is attached; it imports nothing
// from that front end. Everything it needs arrives through mount()'s
// `ctx`: the host's widget helpers (ctx.host), the panel-title and
// config hooks, and ctx.openChannel() -- a message channel to
// preview-view-backend.cc.
//
// The BACKEND owns discovery and lifecycle: this panel says which stage
// it wants to follow and is then told when that stage is waiting, live,
// or gone. There is no polling here, and no REST endpoint behind it.
//
// Media rides the same channel as binary payloads: an fMP4 init segment
// and media fragments played through Media Source Extensions, PNG stills
// for a slow image source, and planar-float PCM played through WebAudio.
// MSE is not gated to secure contexts, so this plays over plain-HTTP LAN
// origins with no HTTPS (unlike WebCodecs).

export { strings } from './preview-strings.js';

export function mount(body, actions, ctx) {
  const { el, clear, makeIcon, t } = ctx.host;
  const cfg = ctx.config || {};
  const onTitle = ctx.onTitle || (() => {});
  const saveConfig = ctx.onConfigChange || (() => {});

  let destroyed = false;
  let player = null;          // { destroy() } -- the MSE/WebAudio player
  let screen = null;          // 'picker' | 'waiting' | 'playing'
  // The (pipeline, stage) this panel follows. It need NOT be running:
  // the backend watches it and pushes "playing" when it goes live and
  // "waiting" when it stops. Persisted in the panel config so a saved
  // layout resumes the same stream.
  let designation = cfg.designation || null;
  // An optional HINT from the host: the pipeline the operator was last
  // working in. When this panel has never been pointed at anything, the
  // first stage list is filtered through it so the panel opens on a
  // stage from that pipeline instead of an empty picker -- which is the
  // difference between a phone showing the stream you were just looking
  // at and asking you to find it again.
  //
  // Consumed EXACTLY ONCE, and only when the panel started with no
  // designation. Re-applying it would make "Change stream" impossible:
  // clearing the designation shows the picker, and a hint that fired
  // again would immediately pick something and take the picker away.
  let hint = designation ? null : (cfg.prefer_pipeline || null);

  // ---- channel ------------------------------------------------------
  const channel = ctx.openChannel({
    onOpen() {
      // A (re)connection is a fresh backend with no memory of us:
      // restate what we want rather than assume it knows.
      if (destroyed) { return; }
      if (designation) {
        channel.send({ m: 'watch', pipeline: designation.pipeline,
                       stage: designation.stage });
      } else {
        channel.send({ m: 'list' });
      }
    },
    onMessage(msg) { if (!destroyed) { handleMessage(msg); } },
    onBinary(header, payload) {
      if (!destroyed && player) { player.onBinary(header, payload); }
    },
    onClose() {
      if (!destroyed && player) { player.setStatus(t('preview.reconnecting')); }
    },
  });

  function handleMessage(msg) {
    switch (msg && msg.m) {
      case 'stages':
        if (screen === 'picker') { renderStageList(msg.list || []); }
        break;
      case 'waiting':
        showWaiting(msg.pipeline, msg.stage);
        break;
      case 'playing':
        showPlaying(msg.title || msg.stage);
        break;
      case 'gone':
        // The followed pipeline was unloaded, so it can never go live:
        // forget it (a saved layout must not resume a dead target) and
        // fall back to the picker.
        setDesignation(null);
        break;
      case 'config':
        if (player) { player.onConfig(msg); }
        break;
      default:
        break;
    }
  }

  // Follow `d` ({pipeline, stage}) or nothing, persist the choice, and
  // tell the backend.
  function setDesignation(d) {
    designation = d;
    cfg.designation = d;
    saveConfig();
    if (d) {
      channel.send({ m: 'watch', pipeline: d.pipeline, stage: d.stage });
    } else {
      channel.send({ m: 'unwatch' });
      showPicker();
    }
  }

  function teardownPlayer() {
    if (player) {
      try { player.destroy(); } catch (e) {}
      player = null;
    }
  }

  function changeBtn() {
    return el('button', { class: 'btn ghost',
      onclick: () => setDesignation(null) }, t('preview.change_stream'));
  }

  // ---- picker (every declared preview stage, live or not) -----------
  let listEl = null;

  function showPicker() {
    teardownPlayer();
    screen = 'picker';
    onTitle(t('preview.panel'));
    clear(body);
    if (actions) { clear(actions); }
    listEl = el('div', { class: 'preview-list' },
      el('div', { class: 'preview-hint' }, t('preview.loading')));
    body.append(el('div', { class: 'preview-picker' },
      el('div', { class: 'preview-picker-title' }, t('preview.select')),
      listEl));
    if (actions) {
      actions.append(el('button', { class: 'btn ghost',
        onclick: () => channel.send({ m: 'list' }) },
        t('preview.refresh')));
    }
    channel.send({ m: 'list' });
  }

  function renderStageList(stages) {
    if (!listEl) { return; }
    // Spend the host's hint on the first list that arrives, then drop it
    // whether or not it matched -- a hint that survives its first list
    // would keep re-selecting and the picker could never be reached.
    if (hint) {
      const want = hint;
      hint = null;
      // A LIVE stage first: with several in the pipeline, the one
      // already producing is the one worth opening on.
      const pick = stages.find((s) => s.pipeline === want && s.live)
                || stages.find((s) => s.pipeline === want);
      if (pick) {
        setDesignation({ pipeline: pick.pipeline, stage: pick.stage });
        return;
      }
    }
    clear(listEl);
    if (stages.length === 0) {
      listEl.append(el('div', { class: 'preview-hint' },
        t('preview.no_stages')));
      return;
    }
    for (const st of stages) {
      const sub = [st.pipeline + ' / ' + st.stage,
        st.live ? t('preview.state_live') : (st.state || '')]
        .filter(Boolean).join('  ·  ');
      listEl.append(el('button',
        { class: 'preview-item' + (st.live ? ' live' : ''),
          onclick: () => setDesignation(
            { pipeline: st.pipeline, stage: st.stage }) },
        el('span', { class: 'preview-item-name' }, st.title || st.stage),
        el('span', { class: 'preview-item-sub' }, sub)));
    }
  }

  // ---- waiting (followed, not running yet) --------------------------
  function showWaiting(pipeline, stage) {
    teardownPlayer();
    listEl = null;
    screen = 'waiting';
    clear(body);
    if (actions) { clear(actions); }
    onTitle(t('preview.waiting_title', { stage }));
    body.append(el('div', { class: 'preview-picker' },
      el('div', { class: 'preview-waiting' }, makeIcon('hourglass', 'sm')),
      el('div', { class: 'preview-picker-title' },
        t('preview.waiting_title', { stage })),
      el('div', { class: 'preview-hint' },
        t('preview.waiting', { pipeline, stage }))));
    if (actions) { actions.append(changeBtn()); }
  }

  // ---- playing ------------------------------------------------------
  function showPlaying(title) {
    teardownPlayer();
    listEl = null;
    screen = 'playing';
    onTitle(t('preview.playing', { stage: title }));
    clear(body);
    if (actions) { clear(actions); }

    // The media wrapper stacks the MSE <video> and the still-image <img>
    // (only one visible at a time): the backend switches to stills for a
    // slow image source and back to fMP4 for video. Pan/zoom transforms
    // this wrapper, so it survives mode flips.
    const video = el('video', { class: 'preview-video',
      autoplay: '', playsinline: '', muted: '' });
    video.muted = true;              // audio plays via WebAudio, not here
    const img = el('img', { class: 'preview-still', alt: '' });
    const media = el('div', { class: 'preview-media' }, video, img);
    const status = el('div', { class: 'preview-status' });

    const view = makeView(media);
    const vbtn = (label, tip, fn) =>
      el('button', { class: 'preview-ctl', title: tip,
        onclick: (e) => { e.stopPropagation(); fn(); } }, label);
    const controls = el('div', { class: 'preview-controls' },
      vbtn('−', t('preview.zoom_out'), () => view.zoomBy(1 / 1.2)),
      vbtn('+', t('preview.zoom_in'), () => view.zoomBy(1.2)),
      vbtn('1:1', t('preview.actual_size'), view.actual),
      vbtn('Fit', t('preview.fit'), view.fit),
      vbtn('⊙', t('preview.center'), view.center));

    // A box that fills the panel body: the host body is position:
    // relative but not always a flex container, so without this the
    // video sizes to its intrinsic aspect and the panel's overflow clips
    // it. With a real box to fill, object-fit:contain letterboxes it.
    const viewport = el('div', { class: 'preview-viewport no-view' },
      media, controls);
    view.attach(viewport);
    body.append(el('div', { class: 'preview-play' }, viewport, status));
    if (actions) { actions.append(changeBtn()); }

    player = makePlayer({ video, img, media, status, view }, t);
  }

  // Transform-based pan/zoom controller for the media wrapper. `−`/`+`
  // zoom around center, `1:1` shows the frame at native pixels, `Fit`
  // letterboxes the whole frame, `⊙` recenters; drag pans a zoomed
  // frame, wheel zooms at the cursor. Disabled (audio-only) is a no-op.
  function makeView(media) {
    let vp = null, iw = 0, ih = 0, k = 1, tx = 0, ty = 0, enabled = false;
    const MINK = 0.1, MAXK = 12;
    const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
    // px per media-pixel at k=1 (object-fit: contain letterbox scale).
    const baseScale = () => {
      if (!vp || !iw || !ih) { return 1; }
      const w = vp.clientWidth, h = vp.clientHeight;
      return (w && h) ? Math.min(w / iw, h / ih) : 1;
    };
    // Size the media box to the displayed (zoomed) dimensions so the
    // browser rasterizes the image/video from the source AT that
    // resolution -- a CSS `scale()` would instead magnify the fit-size
    // raster (its own compositor layer), which looks blurry at 1:1 /
    // zoom-in. Pan with a translate; that only moves the layer.
    const apply = () => {
      if (!vp) { return; }
      const vw = vp.clientWidth, vh = vp.clientHeight;
      if (iw && ih && vw && vh) {
        const s = baseScale() * k;      // on-screen px per media-pixel
        const dw = iw * s, dh = ih * s;
        const mx = Math.max(0, (dw - vw) / 2);
        const my = Math.max(0, (dh - vh) / 2);
        tx = clamp(tx, -mx, mx);
        ty = clamp(ty, -my, my);
        media.style.width  = dw.toFixed(1) + 'px';
        media.style.height = dh.toFixed(1) + 'px';
        media.style.left   = ((vw - dw) / 2).toFixed(1) + 'px';
        media.style.top    = ((vh - dh) / 2).toFixed(1) + 'px';
      } else {                          // intrinsic size unknown yet
        media.style.width = '100%'; media.style.height = '100%';
        media.style.left = '0'; media.style.top = '0';
      }
      media.style.transform =
        'translate(' + tx.toFixed(1) + 'px,' + ty.toFixed(1) + 'px)';
      vp.classList.toggle('zoomable', enabled && k > 1.0001);
    };
    const zoomAt = (cx, cy, factor) => {  // keep (cx,cy) [rel. center] put
      const k2 = clamp(k * factor, MINK, MAXK);
      const r = k2 / k;
      tx = cx - (cx - tx) * r;
      ty = cy - (cy - ty) * r;
      k = k2;
      apply();
    };
    const attach = (viewport) => {
      vp = viewport;
      let drag = false, cap = false, moved = false;
      let sx = 0, sy = 0, lx = 0, ly = 0;
      viewport.addEventListener('pointerdown', (e) => {
        if (!enabled || (e.button !== undefined && e.button !== 0)) {
          return;
        }
        drag = true; cap = false; moved = false;
        sx = lx = e.clientX; sy = ly = e.clientY;
      });
      viewport.addEventListener('pointermove', (e) => {
        if (!drag) { return; }
        if (!moved) {
          if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) <= 3) {
            return;
          }
          moved = true;
          try { viewport.setPointerCapture(e.pointerId); } catch (x) {}
          cap = true; viewport.classList.add('panning');
        }
        tx += e.clientX - lx; ty += e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        apply();
      });
      const end = (e) => {
        if (!drag) { return; }
        drag = false;
        if (cap) {
          try { viewport.releasePointerCapture(e.pointerId); } catch (x) {}
          cap = false;
        }
        viewport.classList.remove('panning');
      };
      viewport.addEventListener('pointerup', end);
      viewport.addEventListener('pointercancel', end);
      viewport.addEventListener('wheel', (e) => {
        if (!enabled) { return; }
        e.preventDefault();
        const r = viewport.getBoundingClientRect();
        zoomAt(e.clientX - r.left - viewport.clientWidth / 2,
               e.clientY - r.top - viewport.clientHeight / 2,
               e.deltaY < 0 ? 1.1 : 1 / 1.1);
      }, { passive: false });
      if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => apply()).observe(viewport);
      }
    };
    return {
      attach,
      setIntrinsic(w, h) { if (w > 0 && h > 0) { iw = w; ih = h; } apply(); },
      setEnabled(on) {
        enabled = !!on;
        if (vp) { vp.classList.toggle('no-view', !enabled); }
        apply();
      },
      zoomBy: (f) => zoomAt(0, 0, f),
      actual() { const bs = baseScale();
        k = clamp(bs > 0 ? 1 / bs : 1, MINK, MAXK); tx = 0; ty = 0; apply(); },
      fit() { k = 1; tx = 0; ty = 0; apply(); },
      center() { tx = 0; ty = 0; apply(); },
    };
  }

  // The media sink: MSE for video, an <img> for stills, WebAudio for
  // PCM. It is fed by the channel's binary payloads and its `config`
  // message; it owns no transport of its own.
  function makePlayer(ui, t) {
    const { video, img, media, status, view } = ui;
    const setStatus = (m) => { status.textContent = m; };

    // iPhone Safari has never exposed MediaSource -- Apple's answer for
    // it was always HLS. iOS 17.1 added MANAGED Media Source instead:
    // the same API surface, with the browser owning buffer eviction and
    // telling the page when it actually wants data.
    //
    // The classic one is preferred WHERE IT EXISTS, so this is purely
    // additive: every platform that plays the preview today keeps the
    // exact path it has now, and the managed source is reached only
    // where there was previously nothing -- which is precisely iPhone.
    // (iPadOS and macOS Safari 17.1 expose both. Switching those over
    // would mean disabling AirPlay on the element and attaching by
    // srcObject, i.e. changing something that already works to gain
    // nothing.)
    const MSE = window.MediaSource || window.ManagedMediaSource;
    const managed = !!window.ManagedMediaSource
                 && MSE === window.ManagedMediaSource;
    // No source-buffer API at all (iOS 16 and older): video cannot play,
    // but that is NOT the end of the panel. The stage sends full-quality
    // stills whenever its source runs below 1 fps, and PCM goes through
    // WebAudio -- neither needs MSE. Degrading to those beats the dead
    // end this used to be, where a perfectly playable still-image stream
    // was refused because video happened to be unavailable.
    setStatus(MSE ? t('preview.connecting') : t('preview.no_video'));

    let gone = false;
    let ms = null;            // MediaSource / ManagedMediaSource
    let sb = null;            // SourceBuffer
    let mime = null;          // 'video/mp4; codecs="avc1.xxxx"'
    let objUrl = null;
    let queue = [];           // pending fMP4 fragment buffers
    let mode = 'video';       // 'video' | 'image' (still-picture mode)
    let lastInit = null;      // retained init -> MSE rebuild on resume
    let imgUrl = null;        // object URL of the current still

    let audioCtx = null;
    let audioCh = 0;
    let audioRate = 0;
    let audioCursor = 0;

    // A user gesture is needed to start a suspended AudioContext; resume
    // on a click if autoplay blocked it.
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
        try {
          if (ms && ms.readyState === 'open') { ms.removeSourceBuffer(sb); }
        } catch (e) {}
        sb = null;
      }
      if (objUrl) {
        try { URL.revokeObjectURL(objUrl); } catch (e) {}
        objUrl = null;
      }
      try { if (video.srcObject) { video.srcObject = null; } } catch (e) {}
      ms = null;
      queue = [];
    }

    // Returns whether a source was actually built. The caller uses that
    // to decide about the status line: clearing it on every init would
    // wipe the "this browser cannot decode live video" explanation in
    // exactly the case that needs it.
    function buildMse(initBytes) {
      teardownMse();
      if (!MSE || !mime) { return false; }   // no API, or config not yet
      // Ask before building. A codec the browser will not take otherwise
      // shows up as an addSourceBuffer throw with nothing to act on;
      // naming it turns a "the preview is broken" report into one that
      // says which codec, from a device nobody here can attach to.
      if (typeof MSE.isTypeSupported === 'function'
          && !MSE.isTypeSupported(mime)) {
        setStatus(t('preview.codec_unsupported', { codec: mime }));
        return false;
      }
      ms = new MSE();
      if (managed) {
        // Both are REQUIRED by the managed path. A managed source cannot
        // be handed to AirPlay, so attaching one to an element that
        // still allows remote playback fails; and it attaches by
        // srcObject rather than by an object URL.
        //
        // Its startstreaming / endstreaming events are deliberately NOT
        // acted on. They exist so a page that FETCHES media can stop
        // fetching; here the frames are pushed at us either way, so
        // dropping them would save nothing and would punch a timeline
        // gap into the source buffer -- trading a real stall risk for no
        // gain. Eviction is the managed source's own job, and
        // trimBuffer() below still bounds the buffer.
        try { video.disableRemotePlayback = true; } catch (e) {}
        try {
          video.srcObject = ms;
        } catch (e) {
          // Older managed implementations only took an object URL.
          objUrl = URL.createObjectURL(ms);
          video.src = objUrl;
        }
      } else {
        objUrl = URL.createObjectURL(ms);
        video.src = objUrl;
      }
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
      return true;
    }

    function pump() {
      if (gone || !sb || sb.updating || queue.length === 0) { return; }
      const buf = queue.shift();
      try {
        sb.appendBuffer(buf);
      } catch (e) {
        if (e && e.name === 'QuotaExceededError') {
          trimBuffer(true);
          queue.unshift(buf);           // retry after the pending remove()
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
      // Without a source-buffer API there is nothing to append to. Drop
      // the fragment and stay on whatever still was last shown -- the
      // panel keeps working for stills and audio.
      if (!MSE) { return; }
      if (mode === 'image') {
        // Video resumed after a still: rebuild the MediaSource from the
        // retained init so the stale pre-still buffer + the timeline gap
        // are dropped, then feed the fresh keyframe fragment.
        mode = 'video';
        media.classList.remove('show-still');
        if (lastInit) { buildMse(lastInit); }
      }
      queue.push(buf);
      pump();
      seekLive();
    }

    // A still is a full-quality PNG: show it in the <img> and let the
    // (hidden) video decoder idle. Flips back on the next fragment.
    function onImage(buf) {
      const url =
        URL.createObjectURL(new Blob([buf], { type: 'image/png' }));
      const prev = imgUrl;
      img.onload = () => {
        if (prev && prev !== url) {
          try { URL.revokeObjectURL(prev); } catch (e) {}
        }
        if (view) { view.setIntrinsic(img.naturalWidth, img.naturalHeight); }
      };
      imgUrl = url;
      img.src = url;
      mode = 'image';
      media.classList.add('show-still');
      if (view) { view.setEnabled(true); }  // a still => there is video
      setStatus('');
    }

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

    // Keep latency + memory bounded even while playing steadily.
    const maint = setInterval(() => {
      if (gone) { return; }
      trimBuffer(false);
      seekLive();
    }, 3000);

    return {
      setStatus,

      onConfig(j) {
        const hasVideo = !!(j.video && j.video.codec);
        if (hasVideo) {
          mime = 'video/mp4; codecs="' + j.video.codec + '"';
          if (view && j.video.width && j.video.height) {
            view.setIntrinsic(j.video.width, j.video.height);
          }
        }
        // View controls are only meaningful with a picture; audio-only
        // streams leave them disabled.
        if (view) { view.setEnabled(hasVideo); }
        if (!hasVideo && j.audio) { setStatus(t('preview.audio_only')); }
        // A video stream this browser cannot decode: say so ONCE, and
        // say what still works, rather than leaving a black frame with
        // no explanation. Stills and audio arrive on the same channel
        // and are unaffected.
        if (hasVideo && !MSE) { setStatus(t('preview.no_video')); }
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
      },

      onBinary(header, payload) {
        if (gone) { return; }
        switch (header && header.m) {
          case 'init':
            lastInit = payload;      // retained for image->video resume
            // Only a source that was actually built earns a cleared
            // status; otherwise the explanation has to stand.
            if (buildMse(payload)) { setStatus(''); }
            break;
          case 'fragment': onFragment(payload); break;
          case 'image':    onImage(payload);    break;
          case 'audio':    onAudio(payload);    break;
          default: break;
        }
      },

      destroy() {
        gone = true;
        clearInterval(maint);
        video.removeEventListener('click', resumeAudio);
        teardownMse();
        try { video.removeAttribute('src'); video.load(); } catch (e) {}
        if (imgUrl) {
          try { URL.revokeObjectURL(imgUrl); } catch (e) {}
          imgUrl = null;
        }
        if (audioCtx) { try { audioCtx.close(); } catch (e) {} }
      },
    };
  }

  // ---- entry --------------------------------------------------------
  // A saved designation shows the waiting screen until the backend says
  // otherwise; without one, the picker. Either way the real state
  // arrives from the backend once the channel opens.
  if (designation) {
    showWaiting(designation.pipeline, designation.stage);
  } else {
    showPicker();
  }

  return () => {
    destroyed = true;
    teardownPlayer();
    channel.close();
  };
}
