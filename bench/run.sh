#!/usr/bin/env bash
# Build + run all three benchmarks. Emits CSV to stdout (header first).
# Defaults: 100 epochs, batch 32. Override with EPOCHS / BATCH env vars.
#
# Requires: built libneuralnethack.a, vendored tiny-dnn under
# bench/third_party/tiny-dnn (with submodules), system mlpack >= 4.7.

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
TRIALS="${TRIALS:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/bench"
LIB="$ROOT/build/libneuralnethack.a"
DATA="$ROOT/test/pima-indians-diabetes"

if [[ ! -f "$LIB" ]]; then
    echo "missing $LIB. Run 'make' from repo root first." >&2
    exit 1
fi

cd "$BENCH"

echo "Building bench_nnh ..." >&2
g++ -std=c++23 -O3 -march=native -DNDEBUG -DUSE_BLAS=1 \
    -I"$ROOT/neuralnethack" -I. \
    bench_nnh.cc "$LIB" -lopenblas -fopenmp -o bench_nnh

echo "Building bench_tinydnn ..." >&2
g++ -std=c++14 -O3 -march=native -DNDEBUG \
    -I. -Ithird_party/tiny-dnn -Ithird_party/tiny-dnn/third_party \
    bench_tinydnn.cc -o bench_tinydnn

if pkg-config --exists mlpack 2>/dev/null || [[ -f /usr/include/mlpack/core.hpp ]]; then
    echo "Building bench_mlpack ..." >&2
    g++ -std=c++17 -O3 -march=native -DNDEBUG \
        -I. -I/usr/include \
        bench_mlpack.cc -o bench_mlpack -larmadillo -lopenblas -fopenmp
    HAS_MLPACK=1
else
    echo "mlpack not found; skipping bench_mlpack." >&2
    HAS_MLPACK=0
fi

RAW="$(mktemp)"
trap 'rm -f "$RAW"' EXIT

echo "lib,dataset,arch,epochs,batch,threads,blas,trial,train_s,infer_us,test_acc" | tee "$RAW"
"$BENCH/bench_nnh" "$DATA" "$EPOCHS" "$BATCH" "$TRIALS" | tee -a "$RAW"
"$BENCH/bench_tinydnn" "$DATA" "$EPOCHS" "$BATCH" "$TRIALS" | tee -a "$RAW"
if [[ "$HAS_MLPACK" == 1 ]]; then
    "$BENCH/bench_mlpack" "$DATA" "$EPOCHS" "$BATCH" "$TRIALS" | tee -a "$RAW"
fi

echo
echo "=== summary (median ± stdev over $TRIALS trials) ==="
python3 "$BENCH/summarise.py" "$RAW"
