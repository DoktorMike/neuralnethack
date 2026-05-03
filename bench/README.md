# Bench

Three-way speed test on Pima Indians Diabetes: neuralnethack vs tiny-dnn
vs mlpack. Same architecture (8-32-1, tansig + logsig), same optimizer
(Adam, lr=0.01), same epochs/batch. Each lib uses its default
threading and linear-algebra backend, since that's the apples-to-apples
for "what you get out of the box".

## Setup

- Build the static library at the repo root: `make` (produces
  `build/libneuralnethack.a`).
- Vendor tiny-dnn into `bench/third_party/tiny-dnn`:
  ```sh
  cd bench/third_party
  git clone --depth 1 --recurse-submodules https://github.com/tiny-dnn/tiny-dnn.git
  ```
  (The AUR package ships an incomplete dep tree; cloning upstream with
  submodules is the path of least resistance.)
- Install mlpack (Arch: `paru -S mlpack`). Note: the AUR PKGBUILD pins
  `CMAKE_CXX_STANDARD=14` but mlpack 4.7 needs 17; patch locally to 17.
  Also disable `BUILD_PYTHON_BINDINGS` to avoid the ccache path and
  speed up the build.

## Run

```sh
./bench/run.sh                         # 10 trials, 100 epochs, batch 32
TRIALS=20 EPOCHS=200 ./bench/run.sh    # override
```

Per-trial CSV goes to stdout, then a median ± stdev summary table.

## What the numbers mean

The published comparison-doc table claims neuralnethack is in the
"distinctive strength = uncertainty + conformal" niche, not the
"distinctive strength = raw speed" niche. The benchmark backs that up:
mlpack wins on speed and on accuracy out of the box. tiny-dnn's
single-thread no-BLAS default makes it look slow; flip those config
flags and most of the gap closes.

The point is to be honest about it, not to win.
