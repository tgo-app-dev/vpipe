// Shared pan/zoom controller for stage views that show a picture.
//
// This owns only the MATH and the pointer/wheel input handling; it never
// touches the thing being displayed. On every change it calls
// `onApply(state)` and the view decides how to realise it -- sizing an
// <img>/<video> box, or redrawing a <canvas>. That split is what lets
// one controller serve both a single-image view and a two-image
// comparison whose whole point is that the two share one transform.
//
//   const pz = makePanZoom({ onApply: (s) => draw(s) });
//   pz.attach(viewportEl);        // installs drag / wheel / resize
//   pz.setIntrinsic(w, h);        // natural pixel size of the content
//   pz.setEnabled(true);          // false => inert (no content yet)
//
// `state` passed to onApply:
//   k       scale relative to fit (1 = letterboxed to the viewport)
//   tx, ty  pan offset in CSS px, relative to centred
//   scale   ON-SCREEN px per content pixel (baseScale * k) -- what a
//           canvas renderer multiplies by
//   dw, dh  displayed content size in CSS px
//   vw, vh  viewport size in CSS px
//   iw, ih  intrinsic content size
//   enabled

const MIN_K = 0.1;
const MAX_K = 12;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function makePanZoom(opts) {
  const onApply = (opts && opts.onApply) || (() => {});
  // Optional: the box the content is fitted into, when that is not the
  // whole viewport. A side-by-side comparison draws each image into half
  // the viewport, and "Fit" must mean "fits its half" -- so the fit
  // scale and the pan clamp are computed against the PANE, not the
  // element. Defaults to the viewport.
  const paneSize = (opts && opts.paneSize) || null;
  // Which mouse button drags the content. 0 (left) suits a view whose
  // only pointer gesture IS the pan; an editor hands this 2 so the left
  // button is free to draw and panning moves to the right one.
  const panButton = (opts && typeof opts.panButton === 'number')
    ? opts.panButton : 0;
  let vp = null;
  let iw = 0, ih = 0;
  let k = 1, tx = 0, ty = 0;
  let enabled = false;

  function pane() {
    const vw = vp ? vp.clientWidth : 0;
    const vh = vp ? vp.clientHeight : 0;
    if (!paneSize) { return { w: vw, h: vh }; }
    const p = paneSize(vw, vh) || { w: vw, h: vh };
    return { w: p.w || vw, h: p.h || vh };
  }

  // CSS px per content pixel at k = 1: the object-fit:contain letterbox
  // scale, so "Fit" is k = 1 and "1:1" is k = 1 / baseScale.
  function baseScale() {
    if (!vp || !iw || !ih) { return 1; }
    const p = pane();
    return (p.w && p.h) ? Math.min(p.w / iw, p.h / ih) : 1;
  }

  function state() {
    const vw = vp ? vp.clientWidth : 0;
    const vh = vp ? vp.clientHeight : 0;
    const scale = baseScale() * k;
    return { k, tx, ty, scale, dw: iw * scale, dh: ih * scale,
             vw, vh, iw, ih, enabled };
  }

  function apply() {
    if (!vp) { return; }
    // Clamp the pan so the content can't be dragged out of its pane:
    // once an axis fits entirely, it stays centred on that axis.
    const p = pane();
    if (iw && ih && p.w && p.h) {
      const s = baseScale() * k;
      const mx = Math.max(0, (iw * s - p.w) / 2);
      const my = Math.max(0, (ih * s - p.h) / 2);
      tx = clamp(tx, -mx, mx);
      ty = clamp(ty, -my, my);
    }
    vp.classList.toggle('zoomable', enabled && k > 1.0001);
    onApply(state());
  }

  // Zoom about (cx, cy), given relative to the viewport CENTRE, so the
  // point under the cursor stays put.
  function zoomAt(cx, cy, factor) {
    const k2 = clamp(k * factor, MIN_K, MAX_K);
    const r = k2 / k;
    tx = cx - (cx - tx) * r;
    ty = cy - (cy - ty) * r;
    k = k2;
    apply();
  }

  function attach(viewport) {
    vp = viewport;
    let drag = false, cap = false, moved = false;
    let sx = 0, sy = 0, lx = 0, ly = 0;

    viewport.addEventListener('pointerdown', (e) => {
      if (!enabled) { return; }
      if (e.button !== undefined && e.button !== panButton) { return; }
      drag = true; cap = false; moved = false;
      sx = lx = e.clientX; sy = ly = e.clientY;
    });
    viewport.addEventListener('pointermove', (e) => {
      if (!drag) { return; }
      if (!moved) {
        // A few px of slop so a click isn't read as a pan (and, in the
        // comparison view, so it doesn't fight the wipe handle).
        if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) <= 3) {
          return;
        }
        moved = true;
        try { viewport.setPointerCapture(e.pointerId); } catch (x) {}
        cap = true;
        viewport.classList.add('panning');
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
  }

  return {
    attach,
    state,
    setIntrinsic(w, h) {
      if (w > 0 && h > 0) { iw = w; ih = h; }
      apply();
    },
    setEnabled(on) {
      enabled = !!on;
      if (vp) { vp.classList.toggle('no-view', !enabled); }
      apply();
    },
    refresh: apply,
    zoomBy: (f) => zoomAt(0, 0, f),
    actual() {
      const bs = baseScale();
      k = clamp(bs > 0 ? 1 / bs : 1, MIN_K, MAX_K);
      tx = 0; ty = 0;
      apply();
    },
    fit()    { k = 1; tx = 0; ty = 0; apply(); },
    center() { tx = 0; ty = 0; apply(); },
  };
}
