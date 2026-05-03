#!/usr/bin/env python3
"""Single-header amalgamation generator (stb-style).

Walks neuralnethack/ for .hh and .cc files, resolves project-local include
deps via topological sort, and emits single_include/neuralnethack.hh.

Usage:
    python scripts/amalgamate.py [--out <path>]

Consumer pattern (stb_image-style):

    // exactly one .cc in your project:
    #define NNH_IMPLEMENTATION
    #include "neuralnethack.hh"

    // every other TU just:
    #include "neuralnethack.hh"

The implementation section is gated by NNH_IMPLEMENTATION so non-inline
function definitions are emitted exactly once. BLAS/OpenMP support is
*not* baked in -- if you want them, define USE_BLAS / NNH_USE_OPENMP
before the implementation include and link the appropriate libraries.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB_ROOT = ROOT / "neuralnethack"
DEFAULT_OUT = ROOT / "single_include" / "neuralnethack.hh"

INCLUDE_GUARD_RE = re.compile(r"^\s*#\s*ifndef\s+__\w+_hh__\s*$|^\s*#\s*define\s+__\w+_hh__\s*$")
LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
SYSTEM_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+<([^>]+)>")
CONFIG_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+["<]config\.h[">]')
HAVE_CONFIG_RE = re.compile(r"^\s*#\s*ifdef\s+HAVE_CONFIG_H\s*$")
PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\s*$")


def collect_files(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{suffix}"))


def normalize_local_include(include_path: str, file_dir: Path) -> Path | None:
    """Resolve `#include "foo/bar.hh"` to an absolute path under LIB_ROOT.
    Returns None if the include doesn't resolve to a project header.
    """
    candidates = [
        (file_dir / include_path).resolve(),
        (LIB_ROOT / include_path).resolve(),
    ]
    # Headers like `../datatools/Foo.hh` -- compute relative to file_dir.
    for cand in candidates:
        if cand.exists() and cand.is_relative_to(LIB_ROOT):
            return cand
    return None


def parse_file(path: Path) -> dict:
    """Returns {local_includes: list[Path], body: str}.

    Strips include guards, project local includes (those are wired up via
    the topological order), config.h includes, and `#pragma once`. System
    includes (`#include <...>`) are kept inline in the body so any
    surrounding `#ifdef USE_BLAS` etc. context survives in the output.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=False)
    file_dir = path.parent

    # Strip the trailing endif of the include guard. The include-guard
    # pattern is: leading #ifndef + #define + body + final #endif. Counting
    # nested directives is overkill -- we drop the matching final #endif by
    # tracking depth.
    depth_in_guard = 0
    stripped: list[str] = []
    local_includes: list[Path] = []
    skip_have_config_block = False

    for line in lines:
        if HAVE_CONFIG_RE.match(line):
            skip_have_config_block = True
            continue
        if skip_have_config_block:
            if re.match(r"^\s*#\s*endif", line):
                skip_have_config_block = False
            continue

        if INCLUDE_GUARD_RE.match(line):
            depth_in_guard += 1 if line.lstrip().startswith("#ifndef") else 0
            continue
        if PRAGMA_ONCE_RE.match(line):
            continue
        if CONFIG_INCLUDE_RE.match(line):
            continue

        m = LOCAL_INCLUDE_RE.match(line)
        if m:
            resolved = normalize_local_include(m.group(1), file_dir)
            if resolved is not None:
                local_includes.append(resolved)
                continue
            # Fall through to keep unresolved local includes as-is (rare).

        # System includes are kept inline (not deduped to a top block).
        # The dedup approach used to strip the surrounding `#ifdef
        # USE_BLAS` / `#ifdef HAVE_CONFIG_H` context and turn conditional
        # includes (like <cblas.h>) into unconditional ones, which then
        # fails to compile on hosts without those headers. Inline
        # duplicates are harmless because every standard header has its
        # own include guard.
        stripped.append(line)

    # Drop the matching final #endif of the include guard, if any.
    if path.suffix == ".hh" and depth_in_guard:
        for i in range(len(stripped) - 1, -1, -1):
            if re.match(r"^\s*#\s*endif", stripped[i]):
                stripped.pop(i)
                break

    return {
        "local_includes": local_includes,
        "body": "\n".join(stripped).strip("\n"),
    }


def topo_sort(headers: list[Path], deps: dict[Path, list[Path]]) -> list[Path]:
    """Kahn's algorithm. Stable: ties broken alphabetically by relpath."""
    indegree: dict[Path, int] = {h: 0 for h in headers}
    edges: dict[Path, set[Path]] = defaultdict(set)
    for h in headers:
        for dep in deps.get(h, []):
            if dep in indegree and dep != h:
                if h not in edges[dep]:
                    edges[dep].add(h)
                    indegree[h] += 1

    ready = sorted([h for h, d in indegree.items() if d == 0])
    out: list[Path] = []
    while ready:
        node = ready.pop(0)
        out.append(node)
        for nxt in sorted(edges[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    if len(out) != len(headers):
        cycle = [h for h, d in indegree.items() if d > 0]
        raise RuntimeError(f"include cycle detected involving: {cycle}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output path")
    args = ap.parse_args()
    out_path = Path(args.out).resolve()

    headers = collect_files(LIB_ROOT, ".hh")
    # Only keep .cc files that have a sibling .hh -- this filters out
    # orphans like TestWeights.cc that aren't part of LIB_SOURCES.
    sources = [s for s in collect_files(LIB_ROOT, ".cc") if s.with_suffix(".hh").exists()]

    parsed_headers = {h: parse_file(h) for h in headers}
    parsed_sources = {s: parse_file(s) for s in sources}

    header_deps = {h: parsed_headers[h]["local_includes"] for h in headers}
    ordered_headers = topo_sort(headers, header_deps)

    # Match each source to its header (same stem in same dir) for ordering.
    source_order: list[Path] = []
    seen: set[Path] = set()
    for h in ordered_headers:
        candidate = h.with_suffix(".cc")
        if candidate.exists():
            source_order.append(candidate)
            seen.add(candidate)
    for s in sources:
        if s not in seen:
            source_order.append(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    out.append("// neuralnethack.hh -- single-header amalgamation")
    out.append("// Generated by scripts/amalgamate.py. Do not edit by hand.")
    out.append("//")
    out.append("// Usage (stb-style):")
    out.append("//   #define NNH_IMPLEMENTATION")
    out.append("//   #include \"neuralnethack.hh\"")
    out.append("// in exactly ONE translation unit. Every other TU just")
    out.append("// `#include \"neuralnethack.hh\"`.")
    out.append("//")
    out.append("// Optional: define USE_BLAS / NNH_USE_OPENMP and link the")
    out.append("// matching libraries before the implementation include.")
    out.append("#ifndef NEURALNETHACK_AMALGAMATED_HH")
    out.append("#define NEURALNETHACK_AMALGAMATED_HH")
    out.append("")
    out.append("// System includes are kept inline at each header / source where they")
    out.append("// originally appeared, including ones inside #ifdef USE_BLAS or other")
    out.append("// preprocessor blocks. Duplicates are harmless (every standard header")
    out.append("// has its own guard), and this preserves the original conditionality")
    out.append("// so building without optional deps (BLAS, OpenMP) still works.")
    out.append("")
    out.append("// ---- declarations --------------------------------------------")
    for h in ordered_headers:
        rel = h.relative_to(LIB_ROOT)
        out.append(f"// ===== {rel} =====")
        out.append(parsed_headers[h]["body"])
        out.append("")

    out.append("// ---- implementation (gated) ----------------------------------")
    out.append("#ifdef NNH_IMPLEMENTATION")
    for s in source_order:
        rel = s.relative_to(LIB_ROOT)
        out.append(f"// ===== {rel} =====")
        out.append(parsed_sources[s]["body"])
        out.append("")
    out.append("#endif // NNH_IMPLEMENTATION")
    out.append("")
    out.append("#endif // NEURALNETHACK_AMALGAMATED_HH")

    out_path.write_text("\n".join(out))

    n_lines = len(out_path.read_text().splitlines())
    print(f"wrote {out_path} ({len(headers)} headers + {len(sources)} sources, {n_lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
