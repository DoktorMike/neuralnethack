# Bench

Speed test on Pima Indians Diabetes and UCI Covertype: neuralnethack vs
tiny-dnn vs mlpack vs PyTorch (eager and torch.compile). Same
architecture (8-32-1 tansig + logsig on Pima; 54-128-7 tansig + softmax
on Covertype), same optimizer (Adam, lr=0.01), same epochs/batch. Each
lib uses its default threading and linear-algebra backend, since that's
the apples-to-apples for "what you get out of the box".

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
- PyTorch: `run.sh` uses a system `torch` if importable, otherwise runs
  `bench_pytorch.py` through an ephemeral `uv` environment. Skipped if
  neither torch nor uv is available. `pytorch-compiled` rows train and
  infer through `torch.compile` (compile/warmup time excluded).

## Run

```sh
./bench/run.sh                         # 10 trials, 100 epochs, batch 32
TRIALS=20 EPOCHS=200 ./bench/run.sh    # override
```

Per-trial CSV goes to stdout, then a median ± stdev summary table.

## Width sweep (PyTorch crossover)

```sh
./bench/sweep.sh                       # IN=64, widths 32..4096, batch 64
WIDTHS="128 256 512" ./bench/sweep.sh  # override
```

Synthetic IN-H-1 regression, identical Adam + MSE protocol on both
sides, emits `lib,in,H,epoch_s,infer_us`. Locates where CPU PyTorch's
per-step dispatch tax amortizes and its multithreaded fused kernels
overtake nnh on training time. Measured on Zen 5: training crossover
at roughly H 150-250 (~10-30k params); batch-1 inference had no
crossover up to H=4096 (nnh still 1.6x ahead).

## What the numbers mean

The published comparison-doc table claims neuralnethack is in the
"distinctive strength = uncertainty + conformal" niche, not the
"distinctive strength = raw speed" niche. The benchmark backs that up:
mlpack wins on speed and on accuracy out of the box. tiny-dnn's
single-thread no-BLAS default makes it look slow; flip those config
flags and most of the gap closes.

PyTorch is the exception where raw speed IS the story: on small CPU
MLPs its per-op dispatch (Python, autograd graph, kernel launch)
dominates the arithmetic, and `torch.compile` makes it worse at this
scale because per-call guard checks outweigh any fusion win. That is a
structural claim about op-by-op frameworks on tiny models, not a knock
on PyTorch at the scale it's built for.

The point is to be honest about it, not to win.
