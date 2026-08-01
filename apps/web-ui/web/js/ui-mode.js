// Which shell to run: the full desktop dashboard, or the reduced phone
// UI under js/phone/.
//
// The question is "is this a PHONE", not "is this a narrow window". A
// desktop window dragged narrow still wants the dashboard -- the
// operator can widen it again -- while a phone never will, so a media
// query is the wrong instrument: it would swap the whole app out from
// under someone resizing a window, and it would answer "desktop" for a
// phone held in landscape.
//
// No sniff is right for everyone, so the answer is overridable and the
// override persists: `?ui=phone` / `?ui=desktop` sets it (which is also
// how the phone UI is developed on a desktop), `?ui=auto` clears it, and
// the phone drawer's "Desktop layout" writes the same key. Both shells
// offer a way back, so a wrong guess is never a dead end.

const KEY = 'vpipe_ui_mode';        // 'phone' | 'desktop' (absent = auto)

function read() {
  try {
    const v = localStorage.getItem(KEY);
    return (v === 'phone' || v === 'desktop') ? v : null;
  } catch (e) { return null; }
}

// A `?ui=` parameter is an explicit instruction, so it both applies now
// and persists -- otherwise every reload during development would need
// the query string back. `?ui=auto` returns to detection.
function readQuery() {
  let v = null;
  try {
    v = new URLSearchParams(location.search).get('ui');
  } catch (e) { return; }
  if (v === 'phone' || v === 'desktop') { setUiMode(v); }
  else if (v === 'auto') { setUiMode(null); }
}

// Persist the shell choice. `null` restores auto-detection. The caller
// reloads: both shells build their whole DOM at boot, so switching in
// place would mean tearing down every live view by hand.
export function setUiMode(mode) {
  try {
    if (mode === 'phone' || mode === 'desktop') {
      localStorage.setItem(KEY, mode);
    } else {
      localStorage.removeItem(KEY);
    }
  } catch (e) { /* storage blocked -- the choice just won't persist */ }
}

export function uiModeOverride() { return read(); }

// Is the CLIENT a phone? Independent of the override, so the desktop
// shell can tell a real desktop from a phone that asked for the desktop
// layout and offer the way back only in the second case.
export function isPhoneDevice() {
  // Chromium answers directly, and its answer beats any UA guess.
  const uad = navigator.userAgentData;
  if (uad && typeof uad.mobile === 'boolean') { return uad.mobile; }
  const ua = navigator.userAgent || '';
  if (/iPhone|iPod|Windows Phone|IEMobile|BlackBerry|BB10|Opera Mini/i
      .test(ua)) {
    return true;
  }
  // "Android" alone is not enough: Android TABLETS carry it and drop the
  // "Mobile" token, and a tablet has room for the full dashboard. Same
  // reason iPadOS -- which reports a Macintosh UA -- falls through to
  // desktop, which is the answer we want anyway.
  if (/Android/i.test(ua) && /Mobile/i.test(ua)) { return true; }
  // Last resort for a UA we don't recognise or one the browser froze: a
  // touch-primary pointer on a physically small screen. `screen` is the
  // DEVICE, so unlike innerWidth it doesn't move when a window resizes.
  const coarse = !!(window.matchMedia
      && window.matchMedia('(pointer: coarse)').matches);
  const shortEdge = Math.min(screen.width || 0, screen.height || 0);
  return coarse && shortEdge > 0 && shortEdge <= 500;
}

export function isPhoneUi() {
  readQuery();
  const forced = read();
  if (forced) { return forced === 'phone'; }
  return isPhoneDevice();
}
