# neuralnethack vs tiny-dnn / mlpack / flashlight

A candid feature comparison. The aim is to help you pick the right tool, not to rank libraries — each is excellent at what it set out to do.

Last reviewed: 2026-05-03. Project metadata (stars, activity) drifts; treat the headline numbers as approximate and re-check upstream before basing a decision on them.

## Table

| | **neuralnethack** | **tiny-dnn** | **mlpack** | **flashlight** |
|---|---|---|---|---|
| **Scope** | MLP only | MLP + CNN + basic RNN | Full ML (NN, trees, SVM, k-means, HMM, …) | Full DL (Transformers, ASR, vision) |
| **Architectures** | Dense + residual skip | Dense, conv, pooling | Dense, conv, RNN, LSTM | Anything you wire (torch-like) |
| **GPU** | None | Optional cuDNN | Bandicoot (GPU Armadillo) — partial | First-class CUDA, ArrayFire |
| **Bindings** | None (C++ only) | None | Python, Julia, R, Go, CLI | Python |
| **LinAlg backend** | Hand-rolled + optional CBLAS | Hand-rolled + optional Eigen | Armadillo (LAPACK/BLAS) | ArrayFire |
| **Activity (2026)** | Hobby, single-maintainer | Stalled since ≈2020 | Active, multi-maintainer | Active, Meta-driven |
| **GitHub stars (approx)** | <100 | ≈14k | ≈5k | ≈5k |
| **Production track record** | Personal lab | Embedded demos, hobbyists | Used in research and industry | Wav2letter, Meta speech |
| **Build pain** | Low (CMake + BLAS) | Lowest (header-only) | Medium (Armadillo deps) | High (ArrayFire, CUDA) |
| **Binary size** | ~1 MB lib | Header-only | ~10–50 MB | 100s of MB with CUDA |
| **Conformal prediction** | **Yes (split, regression + LAC)** | No | No | No |
| **Ensemble framework** | **First-class (samplers, OOB)** | No | No | No |
| **Uncertainty decomposition** | **Yes (total / aleatoric / epistemic)** | No | No | No |
| **Calibration (Platt / temperature)** | Planned | No | No | No |
| **Tabular focus** | **Yes** | Mixed | Yes | No (designed for big tensors) |
| **Test coverage** | 22 tests, 61% line | Sparse | Solid | Solid |
| **License** | GPL-2 | BSD | BSD | MIT |
| **Distinctive strength** | Uncertainty + conformal in pure-C++ MLP | Tiny, header-only, easy demo | Breadth (non-NN ML too) | Big-model GPU training |
| **Distinctive weakness** | MLP only, GPL, no Python | Stale, CPU-only realistically | NN module less polished than core | Heavy deps, overkill for tabular |

## Honest niche

The audience that lands on neuralnethack and *not* the others is someone who needs **prediction sets / intervals with coverage guarantees** in a **pure C++ binary** for a **tabular regression or classification** problem. Niche but real (regulated industries, embedded telemetry, environments where Python is forbidden).

For everything else, the libraries above do it better:

- **flashlight** — anything large-scale or GPU-bound. Speech, vision, Transformers.
- **mlpack** — non-NN classical ML in C++ with multi-language bindings.
- **tiny-dnn** — header-only CNN demos, lowest possible friction (though development has slowed).
- **scikit-learn / PyTorch** — anything tabular where Python is acceptable; far larger ecosystem and tooling.

## Where the comparison gets unfair

A few caveats so the table isn't read as a leaderboard:

- *Stars* and *activity* drift. mlpack and flashlight have multi-org maintainers; neuralnethack is one person doing it for fun. Treat star counts as proxies for ecosystem reach, not quality.
- *Conformal / uncertainty decomposition* aren't implemented in tiny-dnn / mlpack / flashlight, but in those ecosystems users typically reach out to a separate library (e.g. MAPIE in Python) — the gap is real for *C++*-only workflows specifically.
- *Layer .cc files at low coverage* in neuralnethack are an artifact (the layer logic is inlined in headers). The lib's coverage is healthier than the per-file numbers suggest in those rows.
- *GPL-2* on neuralnethack is the loudest weakness for commercial adoption. The other three are permissive.
