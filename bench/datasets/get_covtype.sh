#!/usr/bin/env bash
# Fetch the UCI Covertype dataset (~7 MB compressed, ~70 MB uncompressed)
# and write a simple 80/20 train/test split to bench/datasets/covtype/.
# Idempotent: re-running is a no-op if the splits already exist.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/covtype"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

mkdir -p "$DIR"
if [[ -f "$DIR/covtype.trn.csv" && -f "$DIR/covtype.tst.csv" ]]; then
    echo "covtype splits already present at $DIR; skipping fetch."
    exit 0
fi

if [[ ! -f "$DIR/covtype.data" ]]; then
    echo "downloading covtype.data.gz ..."
    curl -fsSL "$URL" -o "$DIR/covtype.data.gz"
    gunzip -f "$DIR/covtype.data.gz"
fi

# Shuffle then split 80/20. awk + sort -R is reproducible if we seed.
echo "shuffling + splitting ..."
awk 'BEGIN{srand(42)} {print rand() "\t" $0}' "$DIR/covtype.data" \
    | sort -k1,1n \
    | cut -f2- > "$DIR/covtype.shuffled"

N=$(wc -l < "$DIR/covtype.shuffled")
TRAIN_N=$(( N * 80 / 100 ))
head -n "$TRAIN_N" "$DIR/covtype.shuffled" > "$DIR/covtype.trn.csv"
tail -n +"$((TRAIN_N + 1))" "$DIR/covtype.shuffled" > "$DIR/covtype.tst.csv"

rm -f "$DIR/covtype.shuffled"
echo "wrote $DIR/covtype.{trn,tst}.csv ($TRAIN_N train / $((N - TRAIN_N)) test)"
