#!/usr/bin/env python3
"""Translate a legacy NeuralNetHack config (key/value text with `%` comments)
into the TOML format used since version 3.0.

Usage:
    scripts/migrate-config.py old-config.txt > new-config.toml
    scripts/migrate-config.py old-config.txt -o new-config.toml

The output is intentionally close to what the user would write by hand:
sections grouped logically, with one value per line.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path


def tokens(line: str) -> list[str]:
    """Strip `% comment` tail and split on whitespace."""
    pct = -1
    for i, ch in enumerate(line):
        if ch == "%":
            pct = i
            break
    body = line if pct < 0 else line[:pct]
    return body.split()


def parse_legacy(text: str) -> dict[str, list[str]]:
    """Return {key_lowercase: [args...]}. Last write wins, like the old parser."""
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        toks = tokens(raw)
        if not toks:
            continue
        key = toks[0].lower()
        out[key] = toks[1:]
    return out


def split_mode(s: str) -> str:
    # Legacy accepts "rnd"/"ser" or "1"/"0".
    if s in ("rnd", "1"):
        return "rnd"
    return "ser"


def bool_str(s: str) -> str:
    return "true" if s in ("1", "true") else "false"


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def fmt_array_str(xs: Iterable[str]) -> str:
    return "[" + ", ".join(quote(x) for x in xs) + "]"


def fmt_array_int(xs: Iterable[str]) -> str:
    return "[" + ", ".join(xs) + "]"


def emit(cfg: dict[str, list[str]]) -> str:
    out: list[str] = []
    p = lambda s="": out.append(s)

    # Top-level
    if "suffix" in cfg:
        p(f'suffix = {quote(cfg["suffix"][0])}')
    if "seed" in cfg:
        p(f'seed = {cfg["seed"][0]}')
    if "normalization" in cfg:
        p(f'normalization = {quote(cfg["normalization"][0])}')
    if "ptype" in cfg:
        p(f'problem_type = {quote(cfg["ptype"][0])}')
    p()

    # Data
    p("[data.train]")
    if "filename" in cfg:
        p(f'file = {quote(cfg["filename"][0])}')
    if "idcol" in cfg:
        p(f'id_col = {cfg["idcol"][0]}')
    if "incol" in cfg:
        p(f'in_cols = {quote(cfg["incol"][0])}')
    if "outcol" in cfg:
        p(f'out_cols = {quote(cfg["outcol"][0])}')
    if "rowrange" in cfg:
        p(f'row_range = {quote(cfg["rowrange"][0])}')
    p()

    test_keys = ("filenamet", "idcolt", "incolt", "outcolt", "rowranget")
    if any(k in cfg for k in test_keys):
        p("[data.test]")
        if "filenamet" in cfg:
            p(f'file = {quote(cfg["filenamet"][0])}')
        if "idcolt" in cfg:
            p(f'id_col = {cfg["idcolt"][0]}')
        if "incolt" in cfg:
            p(f'in_cols = {quote(cfg["incolt"][0])}')
        if "outcolt" in cfg:
            p(f'out_cols = {quote(cfg["outcolt"][0])}')
        if "rowranget" in cfg:
            p(f'row_range = {quote(cfg["rowranget"][0])}')
        p()

    # Network
    p("[network]")
    if "size" in cfg:
        p(f'size = {fmt_array_int(cfg["size"])}')
    if "actfcn" in cfg:
        p(f'activations = {fmt_array_str(cfg["actfcn"])}')
    if "errfcn" in cfg:
        p(f'error_fcn = {quote(cfg["errfcn"][0])}')
    p()

    # Training
    p("[training]")
    if "minmethod" in cfg:
        p(f'method = {quote(cfg["minmethod"][0])}')
    if "maxepochs" in cfg:
        p(f'max_epochs = {cfg["maxepochs"][0]}')
    p()

    if "gdparam" in cfg:
        gd = cfg["gdparam"]
        p("[training.gd]")
        if len(gd) >= 1:
            p(f"batch_size = {gd[0]}")
        if len(gd) >= 2:
            p(f"learning_rate = {gd[1]}")
        if len(gd) >= 3:
            p(f"lr_decay = {gd[2]}")
        if len(gd) >= 4:
            p(f"momentum = {gd[3]}")
        p()

    if "adamparam" in cfg:
        a = cfg["adamparam"]
        p("[training.adam]")
        if len(a) >= 1:
            p(f"learning_rate = {a[0]}")
        if len(a) >= 2:
            p(f"beta1 = {a[1]}")
        if len(a) >= 3:
            p(f"beta2 = {a[2]}")
        if len(a) >= 4:
            p(f"epsilon = {a[3]}")
        if len(a) >= 5:
            p(f"weight_decay = {a[4]}")
        p()

    if "weightelim" in cfg:
        we = cfg["weightelim"]
        p("[regularization.weight_elim]")
        if len(we) >= 1:
            p(f"enabled = {bool_str(we[0])}")
        if len(we) >= 2:
            p(f"alpha = {we[1]}")
        if len(we) >= 3:
            p(f"w0 = {we[2]}")
        p()

    if "ensparam" in cfg:
        e = cfg["ensparam"]
        p("[ensemble]")
        if len(e) >= 1:
            p(f"method = {quote(e[0])}")
        if len(e) >= 2:
            p(f"runs = {e[1]}")
        if len(e) >= 3:
            p(f"parts = {e[2]}")
        if len(e) >= 4:
            p(f"split = {quote(split_mode(e[3]))}")
        if len(e) >= 5:
            p(f"vary_weights = {bool_str(e[4])}")
        p()

    if "msparam" in cfg:
        m = cfg["msparam"]
        p("[model_selection]")
        if len(m) >= 1:
            p(f"method = {quote(m[0])}")
        if len(m) >= 2:
            p(f"runs = {m[1]}")
        if len(m) >= 3:
            p(f"parts = {m[2]}")
        if len(m) >= 4:
            p(f"split = {quote(split_mode(m[3]))}")
        if len(m) >= 5:
            p(f"fraction = {m[4]}")
        p()

    if "msgparam" in cfg:
        g = cfg["msgparam"]
        p("[model_selection.msg]")
        if len(g) >= 1:
            p(f"runs = {g[0]}")
        if len(g) >= 2:
            p(f"parts = {g[1]}")
        if len(g) >= 3:
            p(f"split = {quote(split_mode(g[2]))}")
        if len(g) >= 4:
            p(f"fraction = {g[3]}")
        p()

    if "vary" in cfg:
        v = cfg["vary"]
        # v: parameter subparam start stop step
        p("[[model_selection.vary]]")
        if len(v) >= 1:
            p(f"parameter = {quote(v[0])}")
        if len(v) >= 2:
            p(f"subparam = {v[1]}")
        if len(v) >= 3:
            p(f"start = {v[2]}")
        if len(v) >= 4:
            p(f"stop = {v[3]}")
        if len(v) >= 5:
            p(f"step = {v[4]}")
        p()

    out_keys = ("savesession", "saveoutputlist", "info")
    if any(k in cfg for k in out_keys):
        p("[output]")
        if "savesession" in cfg:
            p(f'save_session = {bool_str(cfg["savesession"][0])}')
        if "saveoutputlist" in cfg:
            p(f'save_output_list = {bool_str(cfg["saveoutputlist"][0])}')
        if "info" in cfg:
            p(f'info = {cfg["info"][0]}')

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Legacy config file (e.g. config.txt)")
    ap.add_argument("-o", "--output", type=Path, help="Write to file instead of stdout")
    args = ap.parse_args()

    text = args.input.read_text()
    cfg = parse_legacy(text)
    toml = emit(cfg)

    if args.output:
        args.output.write_text(toml)
    else:
        sys.stdout.write(toml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
