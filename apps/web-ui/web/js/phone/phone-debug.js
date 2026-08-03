// Viewport readout for the phone shell, enabled with ?debug=vv.
//
// It exists because the keyboard-layout bugs on iOS all LOOK the same --
// the UI ends up showing its bottom edge over a screen of blank -- while
// having at least three different causes: the document being taller than
// the visible area, the visual viewport being OFFSET inside the layout
// viewport, or the shell's own height simply not tracking. Telling them
// apart needs numbers, and a phone that cannot conveniently be attached
// to a Mac has nowhere to print them.
//
// Every value here is read straight from the browser; the panel computes
// nothing. What it shows, per line:
//
//   vv    visualViewport height / offsetTop / scale -- offsetTop is the
//         one that matters: non-zero means the browser has slid the
//         visible area down inside the layout viewport.
//   win   innerHeight (the LAYOUT viewport -- if this shrinks with the
//         keyboard, the browser is resizing content and there is nothing
//         to emulate), scrollY, and the document's scroll height.
//   css   what the shell published, and whether it thinks a keyboard is
//         up. If --vvh does not move, the tracking is broken; if it
//         moves and `body` does not, the CSS is.
//   body  the pinned page's real on-screen box, from
//         getBoundingClientRect -- so a top of -336 says plainly that
//         the page is sitting above the visible area.
//   app   the same for #app.
//
// The panel glues ITSELF to the visual viewport so it stays readable
// while everything else slides. That is safe here for the reason it was
// not safe for the page: this element never takes focus, so moving it
// cannot change where the browser decides to scroll, and there is no
// loop to enter.

import { el } from '../dom.js';

export function mountViewportDebug() {
  const box = el('div', { class: 'ph-vvdbg' });
  document.body.append(box);

  const rect = (node) => {
    if (!node) { return '-'; }
    const r = node.getBoundingClientRect();
    return Math.round(r.top) + '..' + Math.round(r.bottom)
         + ' h' + Math.round(r.height);
  };

  const paint = () => {
    const vv = window.visualViewport;
    const root = document.documentElement;
    const app = document.getElementById('app');
    const doc = document.scrollingElement || root;
    const cs = getComputedStyle(document.body);
    const vvh = root.style.getPropertyValue('--vvh') || '(unset)';
    const lines = [
      vv ? ('vv   h=' + Math.round(vv.height)
            + ' top=' + Math.round(vv.offsetTop)
            + ' pg=' + Math.round(vv.pageTop)
            + ' s=' + (vv.scale || 1).toFixed(2))
         : 'vv   (no visualViewport API)',
      'win  inner=' + window.innerHeight
        + ' scrollY=' + Math.round(window.scrollY)
        + ' doc=' + (doc ? doc.scrollHeight : '-'),
      'css  --vvh=' + (vvh || '(unset)').trim()
        + ' kbd=' + (app && app.classList.contains('kbd-open') ? 'Y' : 'n'),
      'body ' + rect(document.body) + ' ' + cs.position,
      'app  ' + rect(app),
    ];
    box.textContent = lines.join('\n');
    // Follow the visible area so the panel is legible even when the page
    // it is reporting on has slid off screen.
    if (vv) {
      box.style.transform =
        'translateY(' + Math.round(vv.offsetTop) + 'px)';
    }
  };

  const vv = window.visualViewport;
  if (vv) {
    vv.addEventListener('resize', paint);
    vv.addEventListener('scroll', paint);
  }
  window.addEventListener('scroll', paint, true);
  window.addEventListener('resize', paint);
  // A slow tick as well: some of these settle after the keyboard's
  // animation finishes, with no event to say so.
  const timer = setInterval(paint, 250);
  paint();

  return () => { clearInterval(timer); box.remove(); };
}
