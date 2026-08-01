// Phone text I/O view.
//
// The console itself is the SAME module the desktop workspace mounts
// (views/user-io.js) -- it already carries everything this view has to
// offer: media attachment (picker, drag, paste), the Markdown render
// toggle, the thinking-segment toggle, and Interrupt. Reimplementing any
// of it for the phone would have meant two consoles to keep in step, so
// this file is only the frame: it hands the pane header's control strip
// to the shared module and lets the phone stylesheet lay the rest out
// (the header strip scrolls sideways rather than wrapping, and the
// desktop's Ctrl+J / Enter hint is hidden -- there is no Ctrl on a
// phone).
//
// This view is mounted with `cache: true` by the shell, so the transcript
// and the poll survive a trip to another view; user-io.js already pauses
// its own tick while its DOM is detached.

import { clear } from '../dom.js';
import { t } from '../i18n.js';
import { mountUserIo } from '../views/user-io.js';

export function mountPhoneIo({ body, actions, setTitle }) {
  clear(body);
  clear(actions);
  setTitle(t('nav.io'));
  return mountUserIo(body, actions);
}
