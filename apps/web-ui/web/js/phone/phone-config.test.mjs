// Checks for phone-config.js's AUTO-APPLY rule, and in particular the
// one field flavour that used not to have it: the JSON textarea that
// array/object/any fields render, which is what a multi-file path field
// (load-image `url`, load-video `input_url`) is on both shells.
//
// WHY A TEST AND NOT A GLANCE. The rule lives in TWO copies -- here and
// in views/pipeline-manager.js's configField -- because the phone sheet
// and the desktop form render their own controls. A value typed on one
// shell and not the other is the drift this directory has already seen
// once (see model-filter.test.mjs, same reason, same directory). This
// pins the phone copy, which is the exported one.
//
// The bug it pins: the JSON box had no blur-commit, on the reading that
// a half-typed blob would throw. But readConfig reads EVERY field, so an
// uncommitted edit here did not wait -- it rode out on whatever
// committed next (a neighbouring box losing focus, a path pick), or was
// dropped without a word. When it landed was decided by an unrelated
// field.
//
// No runner and no dependencies, because this tree has neither node nor
// a JS test harness. It runs on any stock macOS under JavaScriptCore:
//
//   /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/\
//     Helpers/jsc --module-file=apps/web-ui/web/js/phone/phone-config.test.mjs
//
// Exit status is not set (jsc always returns 0) -- read the output.

// ---- the smallest DOM the module can be built against ---------------
// el() needs createElement/createTextNode, attributes, className, and
// append; the field needs classList, value, and the listener pair. A
// `change` here is dispatched by hand, which is what a blur-after-edit
// does in a browser.
function makeEl(tag) {
  const e = {
    tag, nodeType: 1, className: '', children: [], attrs: {},
    style: {}, value: '', _on: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
      toggle(c, on) { if (on) { this.add(c); } else { this.remove(c); } },
    },
    setAttribute(k, v) { this.attrs[k] = v; },
    removeAttribute(k) { delete this.attrs[k]; },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener(k, fn) { (this._on[k] = this._on[k] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(ev) { (this._on[ev.type] || []).forEach((fn) => fn(ev)); },
    append(...cs) { cs.forEach((c) => this.children.push(c)); },
    insertBefore(n) { this.children.push(n); },
    remove() {},
    focus() {},
    get firstChild() { return this.children[0] || null; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    querySelectorAll() { return []; },
  };
  return e;
}
globalThis.document = {
  createElement: makeEl,
  createElementNS: (_ns, tag) => makeEl(tag),
  createTextNode: (s) => ({ nodeType: 3, text: String(s) }),
  getElementById: () => makeEl('div'),
  addEventListener() {}, removeEventListener() {},
  querySelectorAll: () => [],
  documentElement: makeEl('html'),
  body: makeEl('body'),
};
globalThis.location = { search: '', hash: '', href: 'http://localhost/',
                        origin: 'http://localhost', protocol: 'http:',
                        host: 'localhost', pathname: '/' };
globalThis.navigator = { languages: ['en'], language: 'en' };
globalThis.history = { replaceState() {} };
globalThis.localStorage = {
  _m: new Map(),
  getItem(k) { return this._m.has(k) ? this._m.get(k) : null; },
  setItem(k, v) { this._m.set(k, String(v)); },
  removeItem(k) { this._m.delete(k); },
};
globalThis.window = globalThis;

const { configField } = await import('./phone-config.js');

let failed = 0;
function check(what, got, want) {
  const ok = got === want;
  if (!ok) { failed++; }
  print(`${ok ? 'ok  ' : 'FAIL'}  ${what}`);
  if (!ok) { print(`        got  ${got}\n        want ${want}`); }
}

// A multi-file path field exactly as the backend describes load-image's
// `url`: `any` + is_path, so it renders the JSON textarea and takes a
// list of picked files.
const URL_FIELD = { key: 'url', type: 'any', is_path: true,
                    path_filter: 'image', current: ['a.png'], present: true };

// Build one, and hand back the textarea and a commit counter.
function build(field) {
  let commits = 0;
  const built = configField(field, { onCommit: () => { commits += 1; } });
  // The row holds the textarea; find it by the property only it has.
  const find = (n) => {
    if (n && n.tag === 'textarea') { return n; }
    for (const c of (n && n.children) || []) {
      const hit = find(c);
      if (hit) { return hit; }
    }
    return null;
  };
  const ta = find(built.el);
  return { built, ta, commits: () => commits,
           bad: () => built.el.classList.contains('bad-json') };
}

// ---- THE REPORTED BUG -----------------------------------------------
{
  const f = build(URL_FIELD);
  check('starts from the current value',
        f.ta.value, JSON.stringify(['a.png']));
  check('no commit before anything is edited', f.commits(), 0);

  // Edit the text BY HAND -- the case that had no auto-apply at all --
  // and blur. `change` is what a blur-after-edit fires.
  f.ta.value = '["a.png", "b.png"]';
  f.ta.dispatchEvent({ type: 'change' });
  check('a hand edit applies on blur', f.commits(), 1);
  check('and the box is not marked', f.bad(), false);
  check('read() returns the edited list',
        JSON.stringify(f.built.read()), JSON.stringify(['a.png', 'b.png']));
}

// ---- a half-typed box does NOT commit, and says so ------------------
// The reason the exclusion existed. It must not throw, must not apply,
// and must not be silent about it.
{
  const f = build(URL_FIELD);
  f.ta.value = '["a.png", "b.pn';          // mid-edit
  f.ta.dispatchEvent({ type: 'change' });
  check('a half-typed box does not apply', f.commits(), 0);
  check('and wears the warning', f.bad(), true);

  // Typing again is the correction in progress: drop the mark rather
  // than nag through every keystroke of the fix.
  f.ta.dispatchEvent({ type: 'input' });
  check('typing clears the warning', f.bad(), false);

  // Finish the edit and blur: now it applies.
  f.ta.value = '["a.png", "b.png"]';
  f.ta.dispatchEvent({ type: 'change' });
  check('the corrected box applies', f.commits(), 1);
  check('and stays unmarked', f.bad(), false);
}

// ---- an emptied box is UNSET, which is a clean value -----------------
// Blank does not parse, so this is the one "invalid JSON" that must
// still commit: read() answers `undefined`, i.e. omit the key.
{
  const f = build(URL_FIELD);
  f.ta.value = '';
  f.ta.dispatchEvent({ type: 'change' });
  check('an emptied box applies as unset', f.commits(), 1);
  check('and is not marked bad', f.bad(), false);
  check('read() omits the key', f.built.read(), undefined);
}

// ---- the gate is read(), not a second parse of its own ---------------
// A box that read() accepts must commit, and one it rejects must not --
// including a shape JSON.parse alone would wave through.
{
  const f = build(URL_FIELD);
  f.ta.value = '   ';                       // whitespace only == unset
  f.ta.dispatchEvent({ type: 'change' });
  check('whitespace reads as unset and applies', f.commits(), 1);
  f.ta.value = '{"not": "a list"}';         // parses; read() takes it
  f.ta.dispatchEvent({ type: 'change' });
  check('any parseable JSON applies (shape is the stage\'s to judge)',
        f.commits(), 2);
}

// ---- the other field types still behave -----------------------------
// The JSON branch is an `else if` on the shared wiring, so a regression
// here would show as a string field committing twice or not at all.
{
  let n = 0;
  const built = configField({ key: 's', type: 'string', current: 'x',
                              present: true },
                            { onCommit: () => { n += 1; } });
  const find = (e) => (e && (e.tag === 'input' || e.tag === 'textarea'))
    ? e : ((e && e.children) || []).map(find).find(Boolean) || null;
  const inp = find(built.el);
  inp.value = 'y';
  inp.dispatchEvent({ type: 'change' });
  check('a string field still commits once on blur', n, 1);
}

print(failed === 0 ? '\nALL PASS' : `\n${failed} FAILED`);
