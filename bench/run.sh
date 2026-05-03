#!/usr/bin/env bash
# Build + run all three benchmarks across the configured datasets.
# Emits per-trial CSV to stdout, then a median ± stdev summary table.
#
# Defaults: pima 100 epochs, covtype 5 epochs, batch 32, 10 trials (pima),
# 3 trials (covtype). Override with EPOCHS_PIMA / EPOCHS_COVTYPE / BATCH /
# TRIALS_PIMA / TRIALS_COVTYPE / DATASETS env vars.
#
# DATASETS=pima or DATASETS=covtype runs only that one. Default runs both.
#
# Requires: built libneuralnethack.a, vendored tiny-dnn under
# bench/third_party/tiny-dnn (with submodules), system mlpack >= 4.7.

set -euo pipefail

EPOCHS_PIMA="${EPOCHS_PIMA:-100}"
EPOCHS_COVTYPE="${EPOCHS_COVTYPE:-5}"
BATCH="${BATCH:-32}"
TRIALS_PIMA="${TRIALS_PIMA:-10}"
TRIALS_COVTYPE="${TRIALS_COVTYPE:-3}"
DATASETS="${DATASETS:-pima covtype}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/bench"
LIB="$ROOT/build/libneuralnethack.a"
PIMA_DIR="$ROOT/test/pima-indians-diabetes"
COVTYPE_DIR="$BENCH/datasets/covtype"

if [[ ! -f "$LIB" ]]; then
    echo "missing $LIB. Run 'make' from repo root first." >&2
    exit 1
fi

cd "$BENCH"

build_pima() {
    echo "Building bench_nnh + bench_tinydnn (pima) ..." >&2
    g++ -std=c++23 -O3 -march=native -DNDEBUG -DUSE_BLAS=1 \
        -I"$ROOT/neuralnethack" -I. \
        bench_nnh.cc "$LIB" -lopenblas -fopenmp -o bench_nnh
    g++ -std=c++14 -O3 -march=native -DNDEBUG \
        -I. -Ithird_party/tiny-dnn -Ithird_party/tiny-dnn/third_party \
        bench_tinydnn.cc -o bench_tinydnn
    if [[ "$HAS_MLPACK" == 1 ]]; then
        echo "Building bench_mlpack (pima) ..." >&2
        g++ -std=c++17 -O3 -march=native -DNDEBUG \
            -I. -I/usr/include \
            bench_mlpack.cc -o bench_mlpack -larmadillo -lopenblas -fopenmp
    fi
}

build_covtype() {
    echo "Building bench_*_covtype ..." >&2
    g++ -std=c++23 -O3 -march=native -DNDEBUG -DUSE_BLAS=1 \
        -I"$ROOT/neuralnethack" -I. \
        bench_nnh_covtype.cc "$LIB" -lopenblas -fopenmp -o bench_nnh_covtype
    g++ -std=c++14 -O3 -march=native -DNDEBUG \
        -I. -Ithird_party/tiny-dnn -Ithird_party/tiny-dnn/third_party \
        bench_tinydnn_covtype.cc -o bench_tinydnn_covtype
    if [[ "$HAS_MLPACK" == 1 ]]; then
        g++ -std=c++17 -O3 -march=native -DNDEBUG \
            -I. -I/usr/include \
            bench_mlpack_covtype.cc -o bench_mlpack_covtype -larmadillo -lopenblas -fopenmp
    fi
}

if pkg-config --exists mlpack 2>/dev/null || [[ -f /usr/include/mlpack/core.hpp ]]; then
    HAS_MLPACK=1
else
    echo "mlpack not found; skipping bench_mlpack*." >&2
    HAS_MLPACK=0
fi

RAW="$(mktemp)"
trap 'rm -f "$RAW"' EXIT
echo "lib,dataset,arch,epochs,batch,threads,blas,trial,train_s,infer_us,test_acc" | tee "$RAW"

for dset in $DATASETS; do
    case "$dset" in
        pima)
            build_pima
            "$BENCH/bench_nnh" "$PIMA_DIR" "$EPOCHS_PIMA" "$BATCH" "$TRIALS_PIMA" | tee -a "$RAW"
            "$BENCH/bench_tinydnn" "$PIMA_DIR" "$EPOCHS_PIMA" "$BATCH" "$TRIALS_PIMA" | tee -a "$RAW"
            if [[ "$HAS_MLPACK" == 1 ]]; then
                "$BENCH/bench_mlpack" "$PIMA_DIR" "$EPOCHS_PIMA" "$BATCH" "$TRIALS_PIMA" \
                    | tee -a "$RAW"
            fi
            ;;
        covtype)
            if [[ ! -f "$COVTYPE_DIR/covtype.trn.csv" ]]; then
                echo "covtype splits missing; running bench/datasets/get_covtype.sh ..." >&2
                "$BENCH/datasets/get_covtype.sh"
            fi
            build_covtype
            "$BENCH/bench_nnh_covtype" "$COVTYPE_DIR" "$EPOCHS_COVTYPE" "$BATCH" \
                "$TRIALS_COVTYPE" | tee -a "$RAW"
            "$BENCH/bench_tinydnn_covtype" "$COVTYPE_DIR" "$EPOCHS_COVTYPE" "$BATCH" \
                "$TRIALS_COVTYPE" | tee -a "$RAW"
            if [[ "$HAS_MLPACK" == 1 ]]; then
                "$BENCH/bench_mlpack_covtype" "$COVTYPE_DIR" "$EPOCHS_COVTYPE" "$BATCH" \
                    "$TRIALS_COVTYPE" | tee -a "$RAW"
            fi
            ;;
        *)
            echo "unknown dataset: $dset (expected pima or covtype)" >&2
            exit 1
            ;;
    esac
done

echo
echo "=== summary (median ± stdev) ==="
python3 "$BENCH/summarise.py" "$RAW"
