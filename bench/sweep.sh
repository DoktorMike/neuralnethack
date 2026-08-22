#!/usr/bin/env bash
# Width sweep: where does CPU PyTorch overtake neuralnethack?
# Synthetic regression, arch IN-H-1, Adam + MSE, double precision,
# identical protocol both sides. Emits CSV: lib,in,H,epoch_s,infer_us.
#
# Defaults: IN=64, N=4096 samples, 3 epochs, batch 64,
# widths 32 128 512 1024 2048 4096. Override with IN / N / EPOCHS /
# BATCH / WIDTHS env vars.
#
# Requires: built libneuralnethack.a; torch importable or uv available.

set -euo pipefail

IN="${IN:-64}"
N="${N:-4096}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-64}"
WIDTHS="${WIDTHS:-32 128 512 1024 2048 4096}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/bench"
LIB="$ROOT/build/libneuralnethack.a"

if [[ ! -f "$LIB" ]]; then
    echo "missing $LIB. Run 'make' from repo root first." >&2
    exit 1
fi

echo "Building bench_sweep_nnh ..." >&2
g++ -std=c++23 -O3 -march=native -DNDEBUG -DUSE_BLAS=1 \
    -I"$ROOT/neuralnethack" -I"$BENCH" \
    "$BENCH/bench_sweep_nnh.cc" "$LIB" -lopenblas -fopenmp \
    -o "$BENCH/bench_sweep_nnh"

if python3 -c 'import torch' 2>/dev/null; then
    PYRUN=(python3)
elif command -v uv >/dev/null; then
    PYRUN=(uv run --quiet --with torch --with numpy python)
else
    echo "neither torch nor uv found; skipping torch side." >&2
    PYRUN=()
fi

echo "lib,in,H,epoch_s,infer_us"
for H in $WIDTHS; do
    "$BENCH/bench_sweep_nnh" "$IN" "$H" "$N" "$EPOCHS" "$BATCH"
done
if [[ ${#PYRUN[@]} -gt 0 ]]; then
    for H in $WIDTHS; do
        "${PYRUN[@]}" "$BENCH/bench_sweep_torch.py" "$IN" "$H" "$N" "$EPOCHS" "$BATCH"
    done
fi
