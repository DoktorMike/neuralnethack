# neuralnethack vs tiny-dnn / mlpack / flashlight

A candid feature comparison so you can pick the right tool. None of these libraries are bad. Each is excellent at what it set out to do, including this one. I just want to be honest about what overlaps and what doesn't, so you don't end up with the wrong hammer for your nail.

Last reviewed 2026-05-03. Anything in the table that smells of marketing (star counts, "production track record") drifts faster than I update this doc, so re-check upstream before basing a decision on it.

## Table

| | **neuralnethack** | **tiny-dnn** | **mlpack** | **flashlight** |
|---|---|---|---|---|
| **Scope** | MLP only | MLP + CNN + basic RNN | Full ML (NN, trees, SVM, k-means, HMM, …) | Full DL (Transformers, ASR, vision) |
| **Architectures** | Dense + residual skip | Dense, conv, pooling | Dense, conv, RNN, LSTM | Anything you wire (torch-like) |
| **GPU** | None | Optional cuDNN | Bandicoot (GPU Armadillo), partial | First-class CUDA, ArrayFire |
| **Bindings** | None (C++ only) | None | Python, Julia, R, Go, CLI | Python |
| **LinAlg backend** | Hand-rolled + optional CBLAS | Hand-rolled + optional Eigen | Armadillo (LAPACK/BLAS) | ArrayFire |
| **Activity (2026)** | Hobby, single-maintainer | Stalled since ≈2020 | Active, multi-maintainer | Active, Meta-driven |
| **GitHub stars (approx)** | <100 | ≈14k | ≈5k | ≈5k |
| **Production track record** | Personal lab | Embedded demos, hobbyists | Used in research and industry | Wav2letter, Meta speech |
| **Build pain** | Low (CMake + BLAS, or a single header) | Lowest (header-only) | Medium (Armadillo deps) | High (ArrayFire, CUDA) |
| **Binary size** | ~1 MB lib | Header-only | ~10–50 MB | 100s of MB with CUDA |
| **Conformal prediction** | **Yes (split, regression + LAC)** | No | No | No |
| **Ensemble framework** | **First-class (samplers, OOB)** | No | No | No |
| **Uncertainty decomposition** | **Yes (total / aleatoric / epistemic)** | No | No | No |
| **Calibration (Platt / temperature)** | Planned | No | No | No |
| **Tabular focus** | **Yes** | Mixed | Yes | No (designed for big tensors) |
| **Test coverage** | 22 tests, 61% line | Sparse | Solid | Solid |
| **License** | MIT | BSD | BSD | MIT |
| **Distinctive strength** | Uncertainty + conformal in pure-C++ MLP | Tiny, header-only, easy demo | Breadth (non-NN ML too) | Big-model GPU training |
| **Distinctive weakness** | MLP only, no Python, niche audience | Stale, CPU-only realistically | NN module less polished than core | Heavy deps, overkill for tabular |

## Where this library actually wins

The audience that lands here and not on one of the others is someone who needs **prediction sets or intervals with coverage guarantees**, in a **pure C++ binary**, on a **tabular regression or classification** problem. That's a small slice (regulated industries, embedded telemetry, anywhere Python is forbidden) but it's a real one, and as far as I can tell the other three don't really cover it.

Everywhere else, the others are better and you should use them:

- **flashlight** for anything large-scale, GPU-bound, or Transformer-shaped. Speech, vision, the works.
- **mlpack** for non-NN classical ML in C++ with bindings out the back to Python, Julia, R, Go. The breadth is hard to beat.
- **tiny-dnn** for the lowest-friction CNN demo (caveat: it's been quiet since around 2020, so kick the tyres before you commit).
- **scikit-learn / PyTorch** for anything tabular where Python is allowed. The ecosystem alone wins. Don't be a hero.

## Speed and accuracy on real benchmarks

Tabular MLP head-to-head on two datasets, identical architecture and
optimiser per dataset (Adam, lr=0.01, batch=32). Each lib uses its
default threading / linear-algebra config that a normal user would
build with: BLAS + OpenMP for nnh and mlpack, OpenMP + AVX intrinsics
for tiny-dnn (`CNN_USE_OMP` + `CNN_USE_AVX`). 16 cores. Numbers are
medians over multiple trials; the harness lives in `bench/`.

**Pima Indians Diabetes** (768 x 8, binary, arch 8-32-1, 100 epochs):

| lib | train (s) | inference (us / sample) | test accuracy |
|---|---|---|---|
| mlpack | 0.001 | 0.10 | 0.768 |
| neuralnethack | 0.010 | 0.15 | 0.744 |
| tiny-dnn | 0.262 | 0.93 | 0.741 |

mlpack wins. On 8x32 GEMMs the per-call BLAS dispatch overhead in nnh
(and the kernel-launch overhead in tiny-dnn's small AVX loops) costs
more than the actual compute. Armadillo's expression-template path
inlines the matmul at the call site and skips that overhead. The
accuracy column is statistically a tie inside one sigma.

**UCI Covertype** (581k x 54, 7-class, arch 54-128-7, 5 epochs):

| lib | train (s) | inference (us / sample) | test accuracy |
|---|---|---|---|
| neuralnethack | 5.56 | **1.29** | **0.828** |
| mlpack | 1.56* | 5.64 | 0.754 |
| tiny-dnn | 52.8 | 2.02 | 0.823 |

\* mlpack hit ensmallen's default convergence tolerance (1e-8) and
stopped early, which is also why its accuracy is ~7 points lower.
Apples-to-apples timing on covtype needs the tolerance loosened or
more epochs forced; until that's done, treat mlpack's 1.56s as a
lower bound rather than a fair 5-epoch result.

**Honest read.** At Pima scale we're 10x off mlpack on training time
because the GEMMs are too small to amortise BLAS dispatch. At
Covertype scale the dispatch overhead vanishes, and nnh comes out
with the **lowest inference latency** and the **highest test
accuracy** of the three. The story isn't "we're faster than mlpack",
it's "the gap depends entirely on whether your matrices are big
enough for BLAS to earn its overhead, and if you're doing
realistic-sized tabular work, this library is in the same league."

## Where the table is unfair

A few caveats so this doesn't read as a leaderboard:

- **Stars and activity drift.** mlpack and flashlight are multi-org maintainer efforts. This one is me, doing it for fun, on weekends, since 2004. Treat star counts as a proxy for ecosystem reach, not quality.
- **Conformal and uncertainty decomposition aren't in the other three** because their users tend to reach for a separate library (MAPIE in Python, e.g.). That's a real gap for *C++-only* workflows specifically. It's not a knock on the libraries themselves.
- **Layer `.cc` files at low coverage** in this repo is an artifact: the layer logic is inlined in the headers, so the `.cc` files are mostly ctor/dtor. Real coverage is healthier than that row suggests.
- **The "niche audience" weakness is the honest one.** This library doesn't compete with flashlight or mlpack on reach, and never will. The whole point of this doc is to help you decide whether you're in the slice where it wins.
