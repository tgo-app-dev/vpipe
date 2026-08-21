// Settings view. Language + color theme + console/log history caps
// + the wired-pool ceiling.

import { el, clear, toast } from '../dom.js';
import { THEMES, getTheme, applyTheme } from '../theme.js';
import { t, getLocale, setLocale, locales } from '../i18n.js';
import { api } from '../api.js';

// theme tag -> i18n key for its self-name label.
const THEME_KEY = { dark: 'settings.theme_dark', light: 'settings.theme_light',
  auto: 'settings.theme_auto' };

export function mountSettings(container) {
  clear(container);

  const wrap = el('div', { class: 'settings' });
  wrap.append(el('div', { class: 'pane-head' },
    el('span', { class: 'title' }, t('settings.title'))));

  const body = el('div', { class: 'settings-body' });
  body.append(languageSetting());
  body.append(themeSetting());
  body.append(consoleLimitSetting());
  body.append(logLimitSetting());
  body.append(wiredPoolSetting());
  wrap.append(body);
  container.append(wrap);
}

// UI language picker. Updating it re-localizes the app (the locale-
// change listener in app.js re-renders the nav + current view) and
// pushes the choice to the backend so server messages match.
function languageSetting() {
  const current = getLocale();
  const buttons = {};

  const group = el('div', { class: 'segmented' });
  for (const l of locales()) {
    const b = el('button', {
      class: 'seg' + (l.tag === current ? ' active' : ''),
      onclick: async () => {
        setLocale(l.tag);
        try { await api.setLanguage(l.tag); }
        catch (e) { /* older backend -- UI still localizes */ }
      },
    }, l.label);
    buttons[l.tag] = b;
    group.append(b);
  }

  return el('div', { class: 'setting' },
    el('div', { class: 'setting-label' }, t('settings.language')),
    el('div', { class: 'setting-desc' }, t('settings.language_desc')),
    group);
}

function themeSetting() {
  const current = getTheme();
  const buttons = {};

  const group = el('div', { class: 'segmented' });
  for (const th of THEMES) {
    const b = el('button', {
      class: 'seg' + (th === current ? ' active' : ''),
      onclick: () => {
        applyTheme(th);
        for (const k of THEMES) {
          buttons[k].classList.toggle('active', k === th);
        }
      },
    }, t(THEME_KEY[th] || th));
    buttons[th] = b;
    group.append(b);
  }

  return el('div', { class: 'setting' },
    el('div', { class: 'setting-label' }, t('settings.theme')),
    el('div', { class: 'setting-desc' }, t('settings.theme_desc')),
    group);
}

// User I/O console history cap. The backend keeps a deque of lines and
// drops from the front like a terminal scrollback buffer once it
// exceeds the cap. Bigger = more scrollback, more memory + bigger
// poll payloads on first connect.
function consoleLimitSetting() {
  const input = el('input', { type: 'number',
    min: '16', max: '1000000', step: '1',
    value: '16384', placeholder: '16384' });
  const status = el('span', { class: 'setting-status' });
  const saveBtn = el('button', { class: 'btn',
    onclick: async () => {
      const n = parseInt(input.value, 10);
      if (!Number.isFinite(n) || n <= 0) {
        toast(t('settings.limit_positive'), 'error');
        return;
      }
      try {
        const r = await api.ioSetLimit(n);
        input.value = String(r.max_console);
        status.textContent = t('settings.saved_current', { n: r.max_console });
        toast(t('settings.console_updated'), 'ok');
      } catch (e) {
        toast(t('settings.save_failed', { msg: e.message }), 'error');
      }
    } }, t('common.save'));

  // Load the current value from the server. Best-effort -- if the
  // endpoint is missing (older backend) the field falls back to the
  // hard-coded default so the page still renders.
  (async () => {
    try {
      const r = await api.ioGetLimit();
      input.value = String(r.max_console);
      status.textContent =
          t('settings.current_range', { n: r.max_console, min: r.min,
            max: r.max });
    } catch (e) {
      status.textContent = t('settings.no_endpoint');
    }
  })();

  return el('div', { class: 'setting' },
    el('div', { class: 'setting-label' }, t('settings.console_history')),
    el('div', { class: 'setting-desc' }, t('settings.console_history_desc')),
    el('div', { class: 'setting-row' },
      input,
      el('span', { class: 'setting-unit' }, t('settings.lines')),
      saveBtn),
    status);
}

// Session Log history cap. The backend keeps a ring (wraparound) of log
// lines and drops from the front once it exceeds the cap -- the log
// counterpart of the User I/O console limit above.
function logLimitSetting() {
  const input = el('input', { type: 'number',
    min: '16', max: '262144', step: '1',
    value: '8192', placeholder: '8192' });
  const status = el('span', { class: 'setting-status' });
  const saveBtn = el('button', { class: 'btn',
    onclick: async () => {
      const n = parseInt(input.value, 10);
      if (!Number.isFinite(n) || n <= 0) {
        toast(t('settings.limit_positive'), 'error');
        return;
      }
      try {
        const r = await api.logSetLimit(n);
        input.value = String(r.max_log);
        status.textContent = t('settings.saved_current', { n: r.max_log });
        toast(t('settings.log_updated'), 'ok');
      } catch (e) {
        toast(t('settings.save_failed', { msg: e.message }), 'error');
      }
    } }, t('common.save'));

  (async () => {
    try {
      const r = await api.logGetLimit();
      input.value = String(r.max_log);
      status.textContent =
          t('settings.current_range', { n: r.max_log, min: r.min,
            max: r.max });
    } catch (e) {
      status.textContent = t('settings.no_endpoint');
    }
  })();

  return el('div', { class: 'setting' },
    el('div', { class: 'setting-label' }, t('settings.log_history')),
    el('div', { class: 'setting-desc' }, t('settings.log_history_desc')),
    el('div', { class: 'setting-row' },
      input,
      el('span', { class: 'setting-unit' }, t('settings.lines')),
      saveBtn),
    status);
}


// The WIRED POOL ceiling -- how much memory this process may make
// unswappable (mlock'd): model weights, a streaming model's resident
// blocks, activation scratch.
//
// Editable WHILE A PIPELINE RUNS, and only upwards. The server is what
// enforces that (409), not this field: a browser check would be a race
// against a run that starts between the poll and the save, and the
// reason for the rule -- that lowering means unwiring buffers a model is
// still reading -- is the server's to know. So the input stays enabled
// and the refusal is reported when it comes.
function wiredPoolSetting() {
  const input = el('input', { type: 'number',
    min: '0', max: '4194304', step: '1024', value: '0', placeholder: '0' });
  const status = el('span', { class: 'setting-status' });
  // How to move the ceiling the status line reports. Its own line
  // rather than part of the description: it is a command to run, and it
  // is only meaningful once the GPU maximum beside it is known.
  const hint = el('div', { class: 'setting-desc' });

  // Render one server document. Always from the RESPONSE rather than
  // from what was typed: an absolute ask is clamped to what the GPU can
  // keep resident, so the field must show what was taken.
  const render = (r) => {
    input.value = String(r.mb || 0);
    const bits = [];
    if (!r.mb) {
      bits.push(t('settings.wired_off'));
    } else {
      bits.push(t('settings.wired_current',
                  { n: r.mb, used: r.used_mb || 0 }));
    }
    // The GPU ceiling, ALWAYS when it is known -- the limit above cannot
    // be read without it, since the same 48 GB is roomy or already
    // clamped depending on this number. 0 means there was no device to
    // ask, and then nothing is said rather than implying no cap exists.
    if (r.device_max_mb) {
      bits.push(r.mb >= r.device_max_mb
                  ? t('settings.wired_capped', { max: r.device_max_mb })
                  : t('settings.wired_device_max', { max: r.device_max_mb }));
    }
    if (r.running) { bits.push(t('settings.wired_running')); }
    status.textContent = bits.join(' · ');
    // The remedy, shown only where there is a ceiling to raise.
    hint.textContent = r.device_max_mb ? t('settings.wired_sysctl') : '';
  };

  const saveBtn = el('button', { class: 'btn',
    onclick: async () => {
      const n = parseInt(input.value, 10);
      if (!Number.isFinite(n) || n < 0) {
        toast(t('settings.limit_nonneg'), 'error');
        return;
      }
      try {
        render(await api.wiredPoolSet(n));
        toast(t('settings.wired_updated'), 'ok');
      } catch (e) {
        // The 409 carries the server's explanation of WHY, which is the
        // part worth showing -- a bare "save failed" would leave an
        // operator retrying a request that cannot succeed until the run
        // stops. Re-read so the field goes back to the real value.
        toast(t('settings.save_failed', { msg: e.message }), 'error');
        try { render(await api.wiredPoolGet()); } catch (e2) { /* keep */ }
      }
    } }, t('common.save'));

  (async () => {
    try {
      render(await api.wiredPoolGet());
    } catch (e) {
      status.textContent = t('settings.no_endpoint');
    }
  })();

  return el('div', { class: 'setting' },
    el('div', { class: 'setting-label' }, t('settings.wired_pool')),
    el('div', { class: 'setting-desc' }, t('settings.wired_pool_desc')),
    el('div', { class: 'setting-row' },
      input,
      el('span', { class: 'setting-unit' }, t('settings.mb')),
      saveBtn),
    status,
    hint);
}
