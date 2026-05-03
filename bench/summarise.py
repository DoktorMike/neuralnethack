#!/usr/bin/env python3
"""Read per-trial CSV from stdin (or argv[1]) and emit a summary table:
median ± std over trials per (lib, dataset). Header row is detected by
the literal first column "lib"."""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict


def main() -> int:
    src = sys.stdin if len(sys.argv) < 2 else open(sys.argv[1])
    rows = [r for r in csv.reader(src) if r and r[0] != "lib"]
    if not rows:
        print("(no rows)")
        return 0

    by_lib = defaultdict(lambda: {
        "train_s": [], "infer_us": [], "test_acc": [],
        "threads": "", "blas": "",
    })
    for r in rows:
        lib, dataset, arch, epochs, batch, threads, blas, trial, train_s, infer_us, acc = r
        by_lib[lib]["train_s"].append(float(train_s))
        by_lib[lib]["infer_us"].append(float(infer_us))
        by_lib[lib]["test_acc"].append(float(acc))
        by_lib[lib]["threads"] = threads
        by_lib[lib]["blas"] = blas

    def fmt(xs: list[float], pct: int = 4) -> str:
        med = st.median(xs)
        std = st.pstdev(xs) if len(xs) > 1 else 0.0
        return f"{med:.{pct}f} ± {std:.{pct}f}"

    cols = [
        ("lib", 15), ("threads", 8), ("blas", 9),
        ("train_s", 22), ("infer_us", 22), ("test_acc", 22),
    ]
    print("  ".join(f"{name:<{w}}" for name, w in cols))
    for lib, d in sorted(by_lib.items()):
        print("  ".join([
            f"{lib:<15}",
            f"{d['threads']:<8}",
            f"{d['blas']:<9}",
            f"{fmt(d['train_s'], 4):<22}",
            f"{fmt(d['infer_us'], 2):<22}",
            f"{fmt(d['test_acc'], 4):<22}",
        ]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
