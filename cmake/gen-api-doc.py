#!/usr/bin/env python3
"""Generate API.md from the public vpipe headers.

The parser is intentionally simple: it understands enough C++ to find
file-level comment blocks, top-level class / struct declarations with
their public sections, and free function declarations. Anything more
exotic (templates, nested types, friend declarations) is skipped --
the public surface of vpipe is small and lives well within these
limits.

Per header, the output contains:
  * the file-level comment block (if any),
  * each top-level class / struct, with
      - its leading comment block,
      - one entry per public method or member, with its leading
        comment block and signature,
  * each top-level free function with its leading comment block and
    signature.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

# A run of lines that look like //-comments (with optional leading
# whitespace). The body of each line has its `// ` prefix stripped.
def strip_comment(line: str) -> str | None:
    s = line.lstrip()
    if not s.startswith("//"):
        return None
    body = s[2:]
    if body.startswith(" "):
        body = body[1:]
    return body.rstrip()


def collect_comment_above(lines: list[str], i: int) -> str:
    """Walk backwards from line i (exclusive), collecting //-comments.

    Returns the joined block. Skips at most one blank line directly
    above i to handle the occasional spacer between a comment and the
    declaration it documents.
    """
    j = i - 1
    if j >= 0 and lines[j].strip() == "":
        j -= 1
    block: list[str] = []
    while j >= 0:
        body = strip_comment(lines[j])
        if body is None:
            break
        block.append(body)
        j -= 1
    block.reverse()
    while block and block[0].strip() == "":
        block.pop(0)
    while block and block[-1].strip() == "":
        block.pop()
    return "\n".join(block)


def collect_leading_comment(lines: list[str]) -> tuple[str, int]:
    """Capture the file-level //-comment block at the top.

    Returns (block, first_non_comment_line_idx). Lines that are
    `#ifndef`/`#define`/`#include`/blank are not skipped here; the
    leading block is exactly the consecutive //-comments at the top.
    """
    block: list[str] = []
    i = 0
    while i < len(lines):
        body = strip_comment(lines[i])
        if body is None:
            break
        block.append(body)
        i += 1
    while block and block[0].strip() == "":
        block.pop(0)
    while block and block[-1].strip() == "":
        block.pop()
    return ("\n".join(block), i)


# A class / struct top-level declaration: capture name and the rest
# of the line for the body-start brace search. The regex deliberately
# does not anchor with `$`, so `class Foo final {` matches and we let
# the brace-finder downstream pick up the body.
_CLASS_RE = re.compile(
    r"^\s*(class|struct)\s+(\w+)\b([^;{]*)"
)
# Access-specifier line inside a class body.
_ACCESS_RE = re.compile(r"^\s*(public|private|protected)\s*:\s*$")


@dataclass
class Member:
    comment: str
    signature: str  # the declaration text, with line breaks preserved


@dataclass
class Decl:
    kind: str       # "class" | "struct" | "free"
    name: str       # class/struct name; for free-functions, the
                    # function name as best we can extract
    comment: str
    signature: str  # the declaration text (class header / func decl)
    members: list[Member] = field(default_factory=list)


@dataclass
class Header:
    rel_path: str
    leading_comment: str
    decls: list[Decl]


def find_matching_brace(text: str, start: int) -> int:
    """Return index just past the matching `}` for the `{` at `start`.

    Naively counts braces. Handles `//`-comments and basic strings to
    avoid spurious matches inside them. Returns len(text) if no match
    is found (truncated header).
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        # Skip line comments.
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        # Skip block comments.
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        # Skip string / char literals (raw strings not handled; not
        # needed for vpipe headers).
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def split_top_level_statements(body: str) -> list[str]:
    """Split a class body into top-level statements at semicolons.

    Skips semicolons that appear inside nested braces, parentheses,
    angle brackets, comments, or string literals. Each statement is
    returned trimmed but with internal whitespace preserved.
    """
    out: list[str] = []
    cur: list[str] = []
    paren = 0
    brace = 0
    angle = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "/" and i + 1 < n and body[i + 1] == "/":
            while i < n and body[i] != "\n":
                cur.append(body[i])
                i += 1
            continue
        if c == "/" and i + 1 < n and body[i + 1] == "*":
            cur.append(body[i]); cur.append(body[i + 1])
            i += 2
            while i + 1 < n and not (body[i] == "*" and body[i + 1] == "/"):
                cur.append(body[i])
                i += 1
            if i + 1 < n:
                cur.append(body[i]); cur.append(body[i + 1])
                i += 2
            continue
        if c in ('"', "'"):
            quote = c
            cur.append(c)
            i += 1
            while i < n and body[i] != quote:
                if body[i] == "\\" and i + 1 < n:
                    cur.append(body[i]); cur.append(body[i + 1])
                    i += 2
                else:
                    cur.append(body[i])
                    i += 1
            if i < n:
                cur.append(body[i])
                i += 1
            continue
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == "{":
            brace += 1
        elif c == "}":
            brace -= 1
            if brace == 0 and paren == 0:
                # Closing brace of an inline method body (or an
                # initializer like `int x[] = {1,2,3}`). Treat it as
                # a statement terminator: the surrounding class body
                # uses these in place of the trailing `;` for inline
                # function definitions.
                cur.append(c)
                stmt = "".join(cur).strip()
                if stmt:
                    out.append(stmt)
                cur = []
                i += 1
                continue
        elif c == "<":
            angle += 1
        elif c == ">":
            if angle > 0:
                angle -= 1
        elif c == ";" and paren == 0 and brace == 0:
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def clean_member_signature(stmt: str) -> str:
    """Strip leading `//` doc comments from a captured statement.

    The body splitter starts each statement immediately after the
    previous semicolon (or `}`) and so includes any leading doc
    comment lines. We collect that doc separately via
    `collect_comment_above`; here we drop it from the rendered code
    block so the ```cpp``` snippet contains only the declaration.
    """
    block = stmt.splitlines()
    while block and block[0].lstrip().startswith("//"):
        block.pop(0)
    while block and not block[0].strip():
        block.pop(0)
    return "\n".join(block).strip()


def extract_member_name(stmt: str) -> str:
    """Best-effort: pull the declared name out of a member statement.

    Walks the statement looking for an identifier followed by `(`
    (function) or `=` / end-of-line (data member). Falls back to the
    last identifier before the first `(` or `;` boundary.
    """
    # Strip trailing default-value expressions ` = ... `.
    s = stmt
    # Function form: <ret-type> name(...)
    m = re.search(r"(\w+)\s*\(", s)
    if m:
        # Avoid matching `if (...)`, `static_cast<...>(`, etc. -- but
        # in a class body those don't appear in a declaration.
        name = m.group(1)
        if name in ("operator", "explicit", "virtual", "static",
                    "const", "noexcept", "constexpr", "inline"):
            # Not an identifier we want; try the next match.
            for m2 in re.finditer(r"(\w+)\s*\(", s):
                if m2.group(1) not in (
                        "operator", "explicit", "virtual", "static",
                        "const", "noexcept", "constexpr", "inline"):
                    return m2.group(1)
        return name
    # Data member form: <type> name [= ...]
    m = re.search(r"(\w+)\s*(?:=|$)", s)
    if m:
        return m.group(1)
    return ""


def is_doc_worthy(stmt: str, class_name: str) -> bool:
    """Skip statements that don't contribute to the public surface.

    Filters out friend / using / type alias noise plus generated
    ctors and dtors (`Foo() = default;`, `~Foo() = default;`,
    `Foo() {};`). Empty-body ctors carry no advertised semantics
    either, so they're treated the same.
    """
    s = stmt.strip()
    if s.startswith("friend "):
        return False
    if s.startswith("using "):
        return False
    if re.search(r"=\s*(default|delete)\b", s):
        return False
    # Skip dtors outright -- they're never part of an interface
    # users have to read.
    if s.lstrip().startswith("~"):
        return False
    name = extract_member_name(s)
    if name and name == class_name:
        # Ctor (with empty body or otherwise). Constructors of public
        # value types are typically not interesting to a library
        # user; the factory methods that hand out instances are.
        return False
    return True


def parse_class(text: str, header_idx: int, lines: list[str],
                line_offsets: list[int]) -> tuple[Decl, int] | None:
    """Parse a class / struct starting at `text[header_idx]`.

    Returns (Decl, end_index_in_text). `end_index` is the index
    immediately after the closing `;` of the class declaration.
    """
    line_no = text.count("\n", 0, header_idx)
    line = lines[line_no]
    m = _CLASS_RE.match(line)
    if not m:
        return None
    kind = m.group(1)
    name = m.group(2)

    # Forward declaration check: if the next significant token after
    # the class header is `;` (and no `{` precedes it), this is a
    # forward decl with no body to document.
    brace = text.find("{", header_idx)
    semi  = text.find(";", header_idx)
    if semi != -1 and (brace == -1 or semi < brace):
        return None
    if brace == -1:
        return None
    body_end = find_matching_brace(text, brace)
    # Class header text: from header_idx to brace (exclusive).
    header_text = text[header_idx:brace].strip()
    body_text = text[brace + 1:body_end - 1]

    leading = collect_comment_above(lines, line_no)

    decl = Decl(kind=kind, name=name, comment=leading,
                signature=header_text)

    # Walk the body, tracking access. Public sections only.
    access = "public" if kind == "struct" else "private"
    stmts = []
    body_lines = body_text.splitlines()
    cur_access = access
    line_idx = 0
    section_lines: list[tuple[int, str]] = []  # (line_no_within_body, text)
    for ln, raw in enumerate(body_lines):
        am = _ACCESS_RE.match(raw)
        if am:
            cur_access = am.group(1)
            continue
        section_lines.append((ln, raw, cur_access))

    # Re-split per access block so we can preserve comments above
    # public-only declarations. Easier: produce the full body with
    # section markers and split by semicolons, tracking which access
    # block each statement belongs to.
    rebuilt: list[str] = []
    access_for_line: list[str] = []
    cur = "public" if kind == "struct" else "private"
    for raw in body_lines:
        am = _ACCESS_RE.match(raw)
        if am:
            cur = am.group(1)
            rebuilt.append("")
            access_for_line.append(cur)
            continue
        rebuilt.append(raw)
        access_for_line.append(cur)

    # Use split_top_level_statements on the joined body, but tag each
    # statement with the access of its first line. To do that we
    # split on the rebuilt text and find each statement's start line
    # by scanning.
    body_joined = "\n".join(rebuilt)
    stmts = split_top_level_statements(body_joined)

    # For comment lookup we need original-line indices in body_lines.
    # Rebuild a cumulative offset map and re-locate each statement.
    pos = 0
    for stmt in stmts:
        # Find this statement's starting line in body_joined.
        idx = body_joined.find(stmt, pos)
        if idx < 0:
            pos += len(stmt)
            continue
        start_line = body_joined.count("\n", 0, idx)
        pos = idx + len(stmt)
        if start_line >= len(access_for_line):
            continue
        if access_for_line[start_line] != "public":
            continue
        cleaned = clean_member_signature(stmt)
        if not cleaned:
            continue
        if not is_doc_worthy(cleaned, name):
            continue
        member_comment = collect_comment_above(body_lines, start_line)
        decl.members.append(Member(comment=member_comment,
                                   signature=cleaned))

    # Find trailing `;` after the body's closing brace, if any.
    end = body_end
    while end < len(text) and text[end] in " \t\r\n":
        end += 1
    if end < len(text) and text[end] == ";":
        end += 1
    return decl, end


def parse_free_function(text: str, lines: list[str],
                        line_no: int) -> Decl | None:
    """Try to parse a top-level free-function declaration.

    Accepts simple forms `<ret> name(args);`. Returns None if the
    line does not look like a function declaration.
    """
    line = lines[line_no].strip()
    # Skip preprocessor / namespace / using / typedef / forward decls,
    # destructors, and `= default` / `= delete` shorthand lines.
    if line.startswith(("#", "}", "{", "namespace", "using ",
                         "typedef ", "extern ", "//", "/*", "~")):
        return None
    if re.search(r"=\s*(default|delete)\b", line):
        return None
    if "(" not in line:
        return None
    if not line.endswith(";"):
        return None
    if line.startswith(("class ", "struct ", "enum ")):
        return None
    name = extract_member_name(line)
    if not name:
        return None
    return Decl(kind="free", name=name,
                comment=collect_comment_above(lines, line_no),
                signature=line)


def parse_header(path: str, source_root: str) -> Header:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    line_offsets = [0]
    for ln in lines:
        line_offsets.append(line_offsets[-1] + len(ln) + 1)

    leading, _ = collect_leading_comment(lines)

    decls: list[Decl] = []

    # Walk the file looking for top-level class declarations OR
    # free function declarations. We do this brace-naively: the
    # public headers don't have anonymous namespaces or other
    # depth-0 brace blocks besides `namespace vpipe { ... }`, which
    # we treat as transparent.
    i = 0
    depth_namespace = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip include / preprocessor lines fast.
        if stripped.startswith(("#include", "#ifndef", "#define",
                                "#endif", "#pragma")):
            i += 1
            continue

        # namespace open / close. We don't track names; vpipe headers
        # only nest one level (`namespace vpipe`).
        if stripped.startswith("namespace ") and stripped.endswith("{"):
            depth_namespace += 1
            i += 1
            continue
        if stripped == "}" and depth_namespace > 0:
            depth_namespace -= 1
            i += 1
            continue

        m = _CLASS_RE.match(line)
        if m:
            line_start = line_offsets[i]
            parsed = parse_class(text, line_start, lines, line_offsets)
            if parsed is None:
                i += 1
                continue
            decl, end_off = parsed
            decls.append(decl)
            # Advance i to the line after end_off.
            new_line = text.count("\n", 0, end_off)
            i = max(i + 1, new_line + 1)
            continue

        # Free function?
        fn = parse_free_function(text, lines, i)
        if fn is not None:
            decls.append(fn)
            i += 1
            continue

        i += 1

    rel = os.path.relpath(path, source_root)
    return Header(rel_path=rel, leading_comment=leading, decls=decls)


def render_decl(decl: Decl, out: list[str]) -> None:
    if decl.kind in ("class", "struct"):
        out.append(f"### {decl.kind} `{decl.name}`")
        out.append("")
        if decl.comment:
            out.append(decl.comment)
            out.append("")
        out.append("```cpp")
        out.append(decl.signature)
        out.append("```")
        out.append("")
        if decl.members:
            for m in decl.members:
                if m.comment:
                    out.append(m.comment)
                    out.append("")
                # Inline-defined methods already end in `}`. Only
                # append a trailing `;` for plain declarations.
                sig = m.signature
                if not sig.rstrip().endswith("}"):
                    sig = sig + ";"
                out.append("```cpp")
                out.append(sig)
                out.append("```")
                out.append("")
        else:
            out.append("_(no documented public members)_")
            out.append("")
    else:
        out.append(f"### `{decl.name}`")
        out.append("")
        if decl.comment:
            out.append(decl.comment)
            out.append("")
        out.append("```cpp")
        out.append(decl.signature)
        out.append("```")
        out.append("")


def render(headers: list[Header]) -> str:
    out: list[str] = []
    out.append("# vpipe public API reference")
    out.append("")
    out.append(
        "Auto-generated from the headers under `include/vpipe/`. "
        "Do not edit by hand -- re-run the build to regenerate."
    )
    out.append("")
    out.append("## Headers")
    out.append("")
    for h in headers:
        anchor = re.sub(r"[^a-z0-9]+", "-",
                        h.rel_path.lower()).strip("-")
        out.append(f"- [`{h.rel_path}`](#{anchor})")
    out.append("")
    for h in headers:
        out.append(f"## `{h.rel_path}`")
        out.append("")
        if h.leading_comment:
            out.append(h.leading_comment)
            out.append("")
        if not h.decls:
            out.append("_(no documented declarations)_")
            out.append("")
            continue
        for decl in h.decls:
            render_decl(decl, out)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="Path to write the Markdown file to.")
    ap.add_argument("--source-root", required=True,
                    help="Project source root, used to render "
                         "relative paths in the Markdown output.")
    ap.add_argument("headers", nargs="+",
                    help="Public header files to scan, in the order "
                         "they should appear in the output.")
    args = ap.parse_args()

    parsed = [parse_header(h, args.source_root) for h in args.headers]
    md = render(parsed)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    new_bytes = md.encode("utf-8")
    if os.path.exists(out):
        with open(out, "rb") as f:
            if f.read() == new_bytes:
                return 0
    with open(out, "wb") as f:
        f.write(new_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
