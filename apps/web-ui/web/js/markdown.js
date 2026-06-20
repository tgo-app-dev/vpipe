// A tiny, dependency-free Markdown -> DOM renderer for the User I/O
// console. It is deliberately small ("simple Markdown"), built up in the
// order the feature set grew: inline emphasis (bold / italic / underline)
// and inline code first, then block constructs -- headings, lists,
// ordered lists, fenced code, and finally GFM-style pipe tables.
//
// SECURITY: every node is built with createElement + textContent (or the
// recursive inline parser, which only ever creates a fixed set of known
// elements and text nodes). We never assign innerHTML, so arbitrary
// console text -- LLM output, echoed user input -- cannot inject markup.
//
// Emphasis conventions (chosen so the three "simple" markers stay
// visually distinct, Discord-style -- note __ is underline here, NOT
// CommonMark bold):
//   **text**  -> bold        (<strong>)
//   *text*    -> italic      (<em>)
//   _text_    -> italic      (<em>, single underscore, word-bounded)
//   __text__  -> underline   (<u>)
//   `text`    -> inline code (<code>, literal, no nested parsing)
//
// renderMarkdown(text) returns a DocumentFragment ready to drop into a
// line node via node.replaceChildren(frag).

const RE_FENCE = /^\s*(`{3,}|~{3,})(.*)$/;
const RE_HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const RE_UL = /^(\s*)[-*+]\s+(.*)$/;
const RE_OL = /^(\s*)(\d+)[.)]\s+(.*)$/;

export function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  const lines = String(text == null ? '' : text)
                  .replace(/\r\n?/g, '\n').split('\n');
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line: a block separator, nothing to emit.
    if (/^\s*$/.test(line)) { i++; continue; }

    // Fenced code block: ``` or ~~~ ... matching closing fence. The body
    // is literal -- no inline parsing, whitespace preserved.
    let m = RE_FENCE.exec(line);
    if (m) {
      const fence = m[1][0];
      const len = m[1].length;
      const close = new RegExp('^\\s*' + fence + '{' + len + ',}\\s*$');
      i++;
      const body = [];
      while (i < lines.length && !close.test(lines[i])) {
        body.push(lines[i]); i++;
      }
      if (i < lines.length) { i++; }   // consume the closing fence
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = body.join('\n');
      pre.appendChild(code);
      frag.appendChild(pre);
      continue;
    }

    // ATX heading: # .. ###### .
    m = RE_HEADING.exec(line);
    if (m) {
      const h = document.createElement('h' + m[1].length);
      parseInline(m[2], h);
      frag.appendChild(h);
      i++;
      continue;
    }

    // GFM pipe table: a header row with a pipe, immediately followed by a
    // delimiter row (| --- | :--: | ...). Rows continue until a blank or
    // pipe-less line.
    if (line.indexOf('|') !== -1 && i + 1 < lines.length &&
        isTableDelimiter(lines[i + 1])) {
      const header = line;
      const align = parseTableAlign(lines[i + 1]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].indexOf('|') !== -1 &&
             !/^\s*$/.test(lines[i])) {
        rows.push(lines[i]); i++;
      }
      frag.appendChild(buildTable(header, align, rows));
      continue;
    }

    // List (ordered or unordered): gather the run of list items, then
    // build a (possibly nested) list from it. Blank lines between items
    // are tolerated (a "loose" list) as long as another item follows --
    // otherwise consecutive numbered items would each become their own
    // <ol> and every one would restart at 1.
    if (isListItem(line)) {
      const block = [];
      while (i < lines.length) {
        if (isListItem(lines[i])) { block.push(lines[i]); i++; continue; }
        if (/^\s*$/.test(lines[i])) {
          let j = i + 1;
          while (j < lines.length && /^\s*$/.test(lines[j])) { j++; }
          if (j < lines.length && isListItem(lines[j])) { i = j; continue; }
        }
        break;
      }
      frag.appendChild(buildList(block));
      continue;
    }

    // Paragraph: gather consecutive plain lines (a single newline is a
    // soft break -> <br>, which suits a chat/console transcript).
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) &&
           !isBlockStart(lines[i])) {
      para.push(lines[i]); i++;
    }
    const p = document.createElement('p');
    para.forEach((ln, idx) => {
      if (idx) { p.appendChild(document.createElement('br')); }
      parseInline(ln, p);
    });
    frag.appendChild(p);
  }

  return frag;
}

// ---- inline emphasis / code -----------------------------------------

// Append inline-parsed children of `text` to `parent`. Recurses into
// emphasis spans so bold/italic/underline can nest; inline code is a
// leaf (its content is literal).
function parseInline(text, parent) {
  let i = 0;
  let buf = '';
  const flush = () => {
    if (buf) {
      parent.appendChild(document.createTextNode(buf));
      buf = '';
    }
  };

  while (i < text.length) {
    const rest = text.slice(i);
    let m;

    // `code` -- literal, highest priority.
    if ((m = /^`([^`]+)`/.exec(rest))) {
      flush();
      const code = document.createElement('code');
      code.textContent = m[1];
      parent.appendChild(code);
      i += m[0].length;
      continue;
    }
    // **bold**
    if ((m = /^\*\*([\s\S]+?)\*\*/.exec(rest))) {
      flush();
      emit('strong', m[1], parent);
      i += m[0].length;
      continue;
    }
    // __underline__ (before single _, which is italic)
    if ((m = /^__([\s\S]+?)__/.exec(rest))) {
      flush();
      emit('u', m[1], parent);
      i += m[0].length;
      continue;
    }
    // *italic* (a lone * not opening a **)
    if ((m = /^\*(?!\*)([\s\S]+?)\*/.exec(rest))) {
      flush();
      emit('em', m[1], parent);
      i += m[0].length;
      continue;
    }
    // _italic_ -- single underscore, word-bounded so my_var_name and
    // snake_case identifiers are left alone.
    if ((m = /^_(?!_)([^_]+?)_(?![A-Za-z0-9])/.exec(rest))) {
      const prev = i > 0 ? text[i - 1] : '';
      if (!/[A-Za-z0-9]/.test(prev)) {
        flush();
        emit('em', m[1], parent);
        i += m[0].length;
        continue;
      }
    }

    buf += text[i];
    i++;
  }
  flush();
}

function emit(tag, inner, parent) {
  const e = document.createElement(tag);
  parseInline(inner, e);
  parent.appendChild(e);
}

// ---- lists ----------------------------------------------------------

function isListItem(line) {
  return RE_UL.test(line) || RE_OL.test(line);
}

function parseListItem(line) {
  let m = RE_OL.exec(line);
  if (m) {
    return { indent: m[1].length, ordered: true,
             start: parseInt(m[2], 10), content: m[3] };
  }
  m = RE_UL.exec(line);
  return { indent: m[1].length, ordered: false, start: 1, content: m[2] };
}

// Build a (possibly nested) list from a run of list-item lines. Nesting
// is by leading indentation; a deeper-indented run attaches to the
// preceding item. The list kind (ul/ol) is taken from the first item at
// each level.
function buildList(blockLines) {
  const items = blockLines.map(parseListItem);
  let pos = 0;

  const build = () => {
    const first = items[pos];
    const base = first.indent;
    const listEl = document.createElement(first.ordered ? 'ol' : 'ul');
    // Honor the source's starting number so a list that begins at N (or
    // one that got split off from a larger list) numbers from N, not 1.
    if (first.ordered && first.start > 1) { listEl.start = first.start; }
    while (pos < items.length && items[pos].indent >= base) {
      if (items[pos].indent > base) {
        const sub = build();
        const host = listEl.lastChild || listEl.appendChild(
          document.createElement('li'));
        host.appendChild(sub);
        continue;
      }
      const li = document.createElement('li');
      parseInline(items[pos].content, li);
      listEl.appendChild(li);
      pos++;
    }
    return listEl;
  };

  return build();
}

// ---- tables ---------------------------------------------------------

// The row under the header that marks a pipe table, e.g.
// "| --- | :--: | ---: |" or "--- | ---". Requires at least one dash.
function isTableDelimiter(line) {
  return line.indexOf('-') !== -1 &&
    /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(line);
}

// Split a table row into trimmed cells, tolerating optional leading and
// trailing pipes.
function splitRow(line) {
  let s = line.trim();
  if (s.startsWith('|')) { s = s.slice(1); }
  if (s.endsWith('|')) { s = s.slice(0, -1); }
  return s.split('|').map((c) => c.trim());
}

function parseTableAlign(delim) {
  return splitRow(delim).map((c) => {
    const l = c.startsWith(':');
    const r = c.endsWith(':');
    return (l && r) ? 'center' : r ? 'right' : l ? 'left' : '';
  });
}

function buildTable(header, align, rows) {
  const table = document.createElement('table');

  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  splitRow(header).forEach((cell, idx) => {
    const th = document.createElement('th');
    if (align[idx]) { th.style.textAlign = align[idx]; }
    parseInline(cell, th);
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    splitRow(row).forEach((cell, idx) => {
      const td = document.createElement('td');
      if (align[idx]) { td.style.textAlign = align[idx]; }
      parseInline(cell, td);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  return table;
}

// A line that begins a non-paragraph block -- used to stop paragraph
// accumulation. (Tables are detected by the header+delimiter pair, so
// they don't need an entry here.)
function isBlockStart(line) {
  return RE_FENCE.test(line) || RE_HEADING.test(line) || isListItem(line);
}
