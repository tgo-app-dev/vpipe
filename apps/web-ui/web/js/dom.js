// Tiny DOM helpers -- no framework.

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) { continue; }
    if (k === 'class') { e.className = v; }
    else if (k === 'style') { e.style.cssText = v; }
    else if (k === 'html') { e.innerHTML = v; }
    else if (k.startsWith('on') && typeof v === 'function') {
      e.addEventListener(k.slice(2), v);
    } else { e.setAttribute(k, v === true ? '' : v); }
  }
  append(e, children);
  return e;
}

export function append(parent, children) {
  for (const c of children.flat ? children.flat(Infinity) : children) {
    if (c === null || c === undefined || c === false) { continue; }
    parent.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}

export function clear(node) {
  while (node.firstChild) { node.removeChild(node.firstChild); }
  return node;
}

// SVG element factory (namespaced).
export function svgEl(tag, attrs = {}) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) { continue; }
    if (k.startsWith('on') && typeof v === 'function') {
      e.addEventListener(k.slice(2), v);
    } else { e.setAttribute(k, v); }
  }
  return e;
}

// A small keyboard-shortcut badge, e.g. kbd('⏎') or kbd('N'). Rendered
// inside buttons to advertise their shortcut.
export function kbd(label) {
  return el('kbd', { class: 'kbd' }, label);
}

// Simple modal. actions: [{label, kind?, cancel?, onClick(close)}].
// Returns a close() function. Keyboard: Enter triggers the primary
// action (kind:'primary'); Escape / backdrop click triggers the cancel
// action (cancel:true or label "Cancel"), else just closes. The
// primary and cancel buttons render ⏎ / Esc hint badges.
export function openModal({ title, body, actions = [], className = '' }) {
  const root = document.getElementById('modal-root');
  const back = el('div', { class: 'modal-back' });
  const modal = el('div',
    { class: 'modal' + (className ? ' ' + className : '') });
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); };

  const primary = actions.find((a) => a.kind === 'primary');
  const cancel = actions.find(
    (a) => a.cancel || /^cancel$/i.test(a.label || ''));
  const dismiss = () => { if (cancel) { cancel.onClick(close); } else { close(); } };

  const onKey = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      dismiss();
    } else if (e.key === 'Enter') {
      // Let textareas keep newlines and let a focused button click itself.
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'TEXTAREA' || tag === 'BUTTON') { return; }
      if (primary) { e.preventDefault(); primary.onClick(close); }
    }
  };

  modal.append(el('h3', {}, title));
  const b = el('div', { class: 'body' });
  append(b, [body]);
  modal.append(b);
  const foot = el('div', { class: 'foot' });
  for (const a of actions) {
    const hint = a.kind === 'primary' ? '⏎' : (a === cancel ? 'Esc' : null);
    foot.append(el('button', { class: 'btn ' + (a.kind || ''),
      onclick: () => a.onClick(close) },
      a.label, hint ? kbd(hint) : null));
  }
  modal.append(foot);
  back.append(modal);
  back.addEventListener('click', (e) => { if (e.target === back) { dismiss(); } });
  document.addEventListener('keydown', onKey);
  root.append(back);
  return close;
}

// A lightweight context menu anchored at viewport point (x, y). `items`
// is a list of { label, danger?, onClick }; a null entry renders a
// divider. The menu closes on selection, click-away, Escape, scroll or
// resize, and is clamped to stay on-screen. Returns a close fn.
export function openMenu(x, y, items) {
  const root = document.getElementById('modal-root') || document.body;
  const menu = el('div', { class: 'ctx-menu' });
  const close = () => {
    menu.remove();
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('pointerdown', onAway, true);
    window.removeEventListener('resize', close);
    window.removeEventListener('scroll', close, true);
  };
  const onKey = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
  };
  const onAway = (e) => { if (!menu.contains(e.target)) { close(); } };
  for (const it of items) {
    if (!it) { menu.append(el('div', { class: 'ctx-sep' })); continue; }
    menu.append(el('button', {
      class: 'ctx-item' + (it.danger ? ' danger' : ''),
      onclick: () => { close(); it.onClick(); },
    }, it.label));
  }
  root.append(menu);
  const r = menu.getBoundingClientRect();
  menu.style.left =
    Math.max(4, Math.min(x, window.innerWidth - r.width - 6)) + 'px';
  menu.style.top =
    Math.max(4, Math.min(y, window.innerHeight - r.height - 6)) + 'px';
  document.addEventListener('keydown', onKey, true);
  // Defer the away-listener a tick so the opening click doesn't close it.
  setTimeout(() => document.addEventListener('pointerdown', onAway, true), 0);
  window.addEventListener('resize', close);
  window.addEventListener('scroll', close, true);
  return close;
}

let toastRoot = null;
export function toast(message, kind = 'info', ms = 3500) {
  if (!toastRoot) { toastRoot = document.getElementById('toast-root'); }
  const t = el('div', { class: 'toast ' + (kind === 'error' ? 'err'
                       : kind === 'ok' ? 'ok' : '') }, message);
  toastRoot.append(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 200); },
             ms);
}
