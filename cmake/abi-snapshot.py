#!/usr/bin/env python3
"""The plugin ABI, written down: what an already-built plugin depends on.

A plugin compiled against the SDK bakes in three things the host must keep
still for as long as that plugin's ABI is inside the support window
(plugin/plugin-abi.h):

  layouts   the size, alignment, bases and field offsets of every record
            the stable headers declare -- from the compiler's own record
            layout dump, so it is the layout the plugin actually compiled;
  virtuals  every class's virtual functions, in declaration (= vtable)
            order, with their signatures;
  symbols   the functions libvpipe exports that those headers declare;
  keys      every string constant those headers declare -- the FlexData and
            SpecExtra keys (and their enumerated values) a plugin and the
            host exchange. A key is a name a built binary already asks for,
            so renaming or dropping one is a break like any other.

The STABLE headers are the installed SDK headers that mark their
declarations VPIPE_API_BEGIN (plus plugin/plugin-abi.h). Toolkit headers
are compiled into the plugin and are not part of it; the host-only
headers the in-tree apps link (VPIPE_HOST_API_BEGIN) are not installed.

  abi-snapshot.py write  --prefix P --lib L --cxx C --out FILE
  abi-snapshot.py check  --prefix P --lib L --cxx C --snapshot FILE

`check` fails on anything REMOVED or CHANGED against the snapshot and
reports additions, which are what a feature flag is for. A failure means
either the change was a mistake, or it is a new ABI version: regenerate
the snapshot (`write`) and decide the number -- see docs/PLUGINS.md,
"Versioning".
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def stable_headers(prefix):
    inc = os.path.join(prefix, "include")
    out = []
    for root, _, files in os.walk(inc):
        for f in files:
            if not f.endswith(".h"):
                continue
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if ("VPIPE_API_BEGIN\n" in text or
                    "define VPIPE_PLUGIN_ABI_VERSION" in text):
                out.append(os.path.relpath(path, inc))
    return sorted(out)


def probe(prefix, headers, cxx, extra):
    inc = os.path.join(prefix, "include")
    with tempfile.NamedTemporaryFile("w", suffix=".cc", delete=False) as t:
        for h in headers:
            t.write('#include "%s"\n' % h)
        name = t.name
    try:
        r = subprocess.run([cxx, "-std=c++20", "-fsyntax-only", "-I" + inc,
                            name] + extra, capture_output=True, text=True)
    finally:
        os.unlink(name)
    if r.returncode != 0:
        sys.exit("abi-snapshot: the probe did not compile:\n" + r.stderr)
    return r.stdout


def walk_ast(text, stable):
    """Records declared in a stable header: qualified name -> virtuals, and
    the qualified names of the functions they declare."""
    dec = json.JSONDecoder()
    docs, i = [], 0
    while i < len(text):
        while i < len(text) and text[i] in " \n\r\t":
            i += 1
        if i >= len(text):
            break
        if text[i] != "{":          # "Dumping ...:" separators
            j = text.find("\n", i)
            i = len(text) if j < 0 else j + 1
            continue
        obj, i = dec.raw_decode(text, i)
        docs.append(obj)

    records, functions, opaque, keys = {}, set(), set(), {}

    # WHICH FILE EACH NODE IS IN. Clang's JSON writes a location's "file"
    # only when it differs from the LAST file it wrote, across every
    # location in dump order: a node's `loc`, then its range's `begin`
    # and `end`, then its children -- and a macro location is two, its
    # `spellingLoc` then its `expansionLoc`. So the file is state, and it
    # has to be replayed in exactly that order, over every node. Reading
    # only some of them (the spelling half, or a range's begin) goes
    # wrong silently after the first macro: VPIPE_ABI_OPAQUE is SPELLED
    # in vpipe/export.h, so everything after an opaque marker was taken
    # to be in export.h -- not a stable header -- and a struct nested
    # there (MetalCompute::MemoryBudget, handed to plugins by value)
    # dropped out of the snapshot. A node belongs where it is EXPANDED.
    state = {"file": ""}

    def track(sl):
        if not isinstance(sl, dict):
            return
        if "spellingLoc" in sl or "expansionLoc" in sl:
            track(sl.get("spellingLoc", {}))
            track(sl.get("expansionLoc", {}))
            return
        f = sl.get("file")
        if f:
            state["file"] = f

    def attribute(node):
        track(node.get("loc", {}))
        node["_file"] = state["file"]
        rng = node.get("range", {})
        track(rng.get("begin", {}))
        track(rng.get("end", {}))
        for c in node.get("inner", []):
            attribute(c)

    def file_of(node):
        return node.get("_file", "")

    def is_stable(f):
        f = f.replace("\\", "/")
        return any(f.endswith("/include/" + h) for h in stable)

    # `internal`: inside a NON-PUBLIC section of a VPIPE_ABI_OPAQUE class.
    # A record declared there is the host's own bookkeeping -- a plugin
    # cannot name it, and an opaque class keeps its code out of line, so
    # no plugin has its layout compiled in. Everywhere else a nested
    # record stays in, whatever its access: a frozen class's inline code
    # may well have baked a private type's layout into a plugin.
    def visit(node, scope, in_template, internal=False):
        kind = node.get("kind", "")
        f = file_of(node)
        name = node.get("name", "")
        if kind == "NamespaceDecl":
            for c in node.get("inner", []):
                visit(c, scope + [name or "(anonymous)"], in_template)
            return
        if kind in ("ClassTemplateDecl", "FunctionTemplateDecl"):
            for c in node.get("inner", []):
                visit(c, scope, True, internal)
            return
        if kind == "CXXRecordDecl":
            q = "::".join(scope + [name])
            ok = (name and not in_template and not internal and
                  node.get("completeDefinition") and is_stable(f) and
                  "(anonymous)" not in scope)
            inner = node.get("inner", [])
            is_opaque = any(c.get("kind") == "TypeAliasDecl" and
                            c.get("name") == "vpipe_abi_opaque"
                            for c in inner)
            if is_opaque:
                opaque.add(q)
            access = ("private" if node.get("tagUsed") == "class"
                      else "public")
            virt = []
            for c in inner:
                ck = c.get("kind", "")
                if ck == "AccessSpecDecl":
                    access = c.get("access", access)
                    continue
                if ck in ("CXXMethodDecl", "CXXDestructorDecl"):
                    if ok and not c.get("isImplicit"):
                        functions.add(q + "::" + c.get("name", ""))
                    if c.get("virtual"):
                        sig = c.get("type", {}).get("qualType", "")
                        pure = " = 0" if c.get("pure") else ""
                        virt.append("%s %s%s" % (c.get("name", ""), sig,
                                                 pure))
                elif ck in ("CXXConstructorDecl",):
                    if ok:
                        functions.add(q + "::" + name)
                else:
                    visit(c, scope + [name], in_template,
                          internal or (is_opaque and access != "public"))
            if ok:
                records[q] = virt
            return
        if kind == "FunctionDecl":
            if not in_template and is_stable(f):
                functions.add("::".join(scope + [name]))
            return      # a body's locals are nobody's contract
        if (kind == "VarDecl" and node.get("constexpr") and not in_template
                and not internal and is_stable(f) and scope and
                "(anonymous)" not in scope):
            lit = string_literal(node)
            if lit is not None:
                keys["::".join(scope + [name])] = lit
            return
        for c in node.get("inner", []):
            visit(c, scope, in_template, internal)

    for d in docs:
        attribute(d)
    for d in docs:
        visit(d, [], False)
    return records, functions, opaque, keys


def string_literal(node):
    """The string literal a constant is initialised with, or None."""
    if node.get("kind") == "StringLiteral":
        return node.get("value")
    for c in node.get("inner", []):
        v = string_literal(c)
        if v is not None:
            return v
    return None


def layouts(text, names):
    """Each record's OWN layout: its direct bases and fields with their
    offsets, and its size and alignment. Nested expansions -- a base's
    fields, a std::string's internals -- are dropped: a base that matters
    has its own entry, and a library type's internals are the toolchain's
    (and would pin the snapshot to one Xcode)."""
    out = {}
    for block in text.split("*** Dumping AST Record Layout")[1:]:
        lines = [l.rstrip() for l in block.strip("\n").split("\n")]
        if not lines:
            continue
        head = re.sub(r"^\s*\d+\s*\|\s*", "", lines[0])
        rec = re.sub(r"^(struct|class|union)\s+", "", head)
        if rec not in names:
            continue
        body = []
        for l in lines:
            if not l.strip():
                break
            off, _, rest = l.partition("|")
            depth = (len(rest) - len(rest.lstrip(" ")) - 1) // 2
            if rest.lstrip().startswith("[") or rest.lstrip().startswith(
                    "nvsize"):
                depth = 0
            if depth > 1:
                continue
            item = re.sub(r"\(anonymous at [^)]*\)", "(anonymous)",
                          rest.strip())
            body.append(("%s | %s" % (off.strip(), item)).strip())
        out[rec] = body
    return out


def exported(lib, functions, records):
    nm = subprocess.run(["nm", "-gU", lib], capture_output=True, text=True)
    if nm.returncode != 0:
        sys.exit("abi-snapshot: nm failed on %s" % lib)
    raw = [l.split()[-1] for l in nm.stdout.splitlines() if l.strip()]
    dem = subprocess.run(["c++filt"], input="\n".join(raw),
                         capture_output=True, text=True).stdout.splitlines()
    keep = set()
    for s in dem:
        m = re.match(r"^(vtable|typeinfo|typeinfo name) for (.*)$", s)
        if m:
            if m.group(2) in records:
                keep.add(s)
            continue
        base = re.sub(r"\(.*$", "", s)
        # Strip a leading return type of a template specialisation.
        base = base.split(" ")[-1] if " " in base else base
        if base in functions:
            keep.add(s)
    return sorted(keep)


def snapshot(args):
    stable = stable_headers(args.prefix)
    ast = probe(args.prefix, stable, args.cxx,
                ["-Xclang", "-ast-dump=json", "-Xclang",
                 "-ast-dump-filter=vpipe"])
    records, functions, opaque, keys = walk_ast(ast, stable)
    lay = probe(args.prefix, stable, args.cxx,
                ["-Xclang", "-fdump-record-layouts-complete"])
    # A host-owned type (VPIPE_ABI_OPAQUE) keeps its virtuals and its
    # symbols in the contract; only its layout is the host's own.
    layout = layouts(lay, set(records) - opaque)
    syms = exported(args.lib, functions, set(records))
    return stable, records, layout, syms, keys


def render(stable, records, layout, syms, keys):
    out = ["# The plugin ABI snapshot -- generated by cmake/abi-snapshot.py.",
           "# Do not edit; regenerate with the vpipe_abi_snapshot target.",
           "", "[headers]"]
    out += stable
    out += ["", "[layouts]"]
    for r in sorted(layout):
        out.append("record " + r)
        out += ["  " + l for l in layout[r]]
    out += ["", "[virtuals]"]
    for r in sorted(records):
        if records[r]:
            out.append("class " + r)
            out += ["  " + v for v in records[r]]
    out += ["", "[symbols]"]
    out += syms
    out += ["", "[keys]"]
    out += ["%s = %s" % (k, keys[k]) for k in sorted(keys)]
    return "\n".join(out) + "\n"


def parse(text):
    sec, cur, data = None, None, {"layouts": {}, "virtuals": {},
                                   "symbols": set(), "headers": set(),
                                   "keys": {}}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec, cur = line[1:-1], None
            continue
        if sec in ("layouts", "virtuals"):
            if not line.startswith("  "):
                cur = line.split(" ", 1)[1]
                data[sec][cur] = []
            else:
                data[sec][cur].append(line.strip())
        elif sec in ("symbols", "headers"):
            data[sec].add(line)
        elif sec == "keys":
            k, _, v = line.partition(" = ")
            data["keys"][k] = v
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "check"])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cxx", required=True)
    ap.add_argument("--out")
    ap.add_argument("--snapshot")
    args = ap.parse_args()
    text = render(*snapshot(args))
    if args.mode == "write":
        with open(args.out, "w") as f:
            f.write(text)
        print("abi-snapshot: wrote %s" % args.out)
        return 0
    with open(args.snapshot) as f:
        old = parse(f.read())
    new = parse(text)
    problems, added = [], []
    for sec in ("layouts", "virtuals"):
        for r, body in old[sec].items():
            if r not in new[sec]:
                problems.append("%s: %s was removed" % (sec, r))
            elif new[sec][r] != body:
                problems.append("%s: %s changed" % (sec, r))
        added += ["%s: %s" % (sec, r) for r in new[sec] if r not in old[sec]]
    for s in sorted(old["symbols"] - new["symbols"]):
        problems.append("symbols: %s is no longer exported" % s)
    for k, v in sorted(old["keys"].items()):
        if k not in new["keys"]:
            problems.append("keys: %s (%s) was removed" % (k, v))
        elif new["keys"][k] != v:
            problems.append("keys: %s changed from %s to %s"
                            % (k, v, new["keys"][k]))
    added += ["keys: %s = %s" % (k, v) for k, v in
              sorted(new["keys"].items()) if k not in old["keys"]]
    added += ["symbols: " + s for s in sorted(new["symbols"] -
                                               old["symbols"])]
    for a in added:
        print("abi-snapshot: added   " + a)
    for p in problems:
        print("abi-snapshot: BREAKS  " + p)
    if problems:
        print("abi-snapshot: %d change(s) to the plugin ABI against %s. If "
              "intended, this is a new ABI version: regenerate the snapshot "
              "(target vpipe_abi_snapshot) and see docs/PLUGINS.md, "
              "\"Versioning\"." % (len(problems), args.snapshot))
        return 1
    print("abi-snapshot: the plugin ABI matches %s (%d addition(s))"
          % (os.path.basename(args.snapshot), len(added)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
