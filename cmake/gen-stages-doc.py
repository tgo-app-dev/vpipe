#!/usr/bin/env python3
"""Generate STAGES.md from vpipe stage headers.

Scans each header for class declarations of the form

    class FooStage final : public TypedStage<FooStage>

(or the multi-line variant) plus the matching

    static constexpr const char* kTypeName = "...";

and the comment block that immediately precedes the class. The
collected entries are sorted by type name and emitted as a Markdown
file with one section per stage.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

# Match the class line(s) that identify a stage. Tolerates the form
#   class FooStage final : public TypedStage<FooStage>
# and the line-broken form
#   class FooStage final
#     : public TypedStage<FooStage>
_CLASS_RE = re.compile(
    r"class\s+(\w+Stage)\s+final\b[^{;]*?TypedStage<\1>",
    re.DOTALL,
)
_TYPENAME_RE = re.compile(
    r'kTypeName\s*=\s*"([^"]+)"'
)


@dataclass
class Stage:
    type_name: str
    class_name: str
    rel_path: str
    description: str


def _strip_comment(line: str) -> str | None:
    """Strip a leading C++ // comment marker, returning the body.

    Returns None if `line` is not a // comment line. Preserves the
    rest of the content verbatim, including any leading whitespace
    after the marker, so indentation in code/config blocks survives.
    """
    s = line.lstrip()
    if not s.startswith("//"):
        return None
    body = s[2:]
    # Drop exactly one space after // when present, the way Doxygen
    # and most code styles assume.
    if body.startswith(" "):
        body = body[1:]
    return body.rstrip("\n")


def _extract_block(lines: list[str], class_line_idx: int) -> str:
    """Walk backwards from a class declaration collecting comments.

    Skips blank lines immediately above the class to handle the
    occasional spacer between the comment and the class.
    """
    i = class_line_idx - 1
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    block: list[str] = []
    while i >= 0:
        body = _strip_comment(lines[i])
        if body is None:
            break
        block.append(body)
        i -= 1
    block.reverse()
    # Trim leading/trailing blank lines from the captured block.
    while block and block[0].strip() == "":
        block.pop(0)
    while block and block[-1].strip() == "":
        block.pop()
    return "\n".join(block)


def _find_class_line(text: str, match_start: int) -> int:
    """Return the 0-based line index containing `match_start` in `text`."""
    return text.count("\n", 0, match_start)


def _scan_header(path: str, source_root: str) -> list[Stage]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    stages: list[Stage] = []
    for m in _CLASS_RE.finditer(text):
        class_name = m.group(1)
        class_line_idx = _find_class_line(text, m.start())

        # Search forward from the class match for kTypeName. Cap the
        # search at the next class declaration so we never pick up a
        # neighbouring stage's type name.
        next_match = _CLASS_RE.search(text, m.end())
        end = next_match.start() if next_match else len(text)
        type_name_match = _TYPENAME_RE.search(text, m.end(), end)
        if not type_name_match:
            sys.stderr.write(
                f"warning: {path}: class {class_name} has no kTypeName "
                f"-- skipped\n"
            )
            continue
        type_name = type_name_match.group(1)
        description = _extract_block(lines, class_line_idx)
        rel = os.path.relpath(path, source_root)
        stages.append(Stage(type_name, class_name, rel, description))
    return stages


def _format(stages: list[Stage]) -> str:
    out: list[str] = []
    out.append("# vpipe stage reference")
    out.append("")
    out.append(
        "Auto-generated from stage headers. Do not edit by hand -- "
        "re-run the build to regenerate."
    )
    out.append("")
    out.append(f"Stage count: **{len(stages)}**.")
    out.append("")
    out.append("## Index")
    out.append("")
    for s in stages:
        anchor = re.sub(r"[^a-z0-9]+", "-", s.type_name.lower()).strip("-")
        out.append(f"- [`{s.type_name}`](#{anchor}) -- `{s.class_name}`")
    out.append("")
    for s in stages:
        out.append(f"## `{s.type_name}`")
        out.append("")
        out.append(f"- **Class:** `vpipe::{s.class_name}`")
        out.append(f"- **Header:** `{s.rel_path}`")
        out.append("")
        if s.description:
            out.append(s.description)
            out.append("")
        else:
            out.append("_(no description)_")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="Path to write the Markdown file to.")
    ap.add_argument("--source-root", required=True,
                    help="Project source root, used to render relative "
                         "paths in the Markdown output.")
    ap.add_argument("headers", nargs="+",
                    help="Stage header files to scan.")
    args = ap.parse_args()

    all_stages: list[Stage] = []
    seen_types: dict[str, str] = {}
    for h in args.headers:
        for st in _scan_header(h, args.source_root):
            if st.type_name in seen_types:
                sys.stderr.write(
                    f"warning: duplicate type_name '{st.type_name}' in "
                    f"{st.rel_path} (first seen in "
                    f"{seen_types[st.type_name]}) -- keeping first\n"
                )
                continue
            seen_types[st.type_name] = st.rel_path
            all_stages.append(st)

    all_stages.sort(key=lambda s: s.type_name)
    md = _format(all_stages)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Write only when the contents change so re-builds with no stage
    # changes don't perturb downstream timestamps.
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
