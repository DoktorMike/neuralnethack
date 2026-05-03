# TODO

## Worth doing (fits mission, cheap)

### Class weights / sample weights in cross-entropy
Real tabular data is rarely balanced (Pima already isn't). Currently the
only fix is resampling. Add `theError->classWeights({...})` that scales the
per-pattern delta. ~50 LOC. Big practitioner-side win, low risk.

### Calibration
Predicted probabilities don't match observed frequencies out of the box,
and nobody reports this. Platt scaling, temperature scaling, reliability
diagram, ECE / MCE. ~200 LOC. Pairs naturally with the existing uncertainty
decomposition (calibrated total uncertainty is what users actually want).

## Statistics
- Calculate a P-value for the ROC curve. Bootstrap CI on AUC is the
  cheap version (resample dataset → recompute AUC many times → percentile
  / one-sided p-value). DeLong's test is the principled binary
  comparison; only worth it if a concrete user asks.

## Nice but more work

### Conformal prediction follow-ups
Split conformal v1 shipped (`evaltools/Conformal`): per-dim regression
residual quantiles + LAC non-conformity for classification. Outstanding:
- APS (adaptive prediction sets) for classification: smaller sets in easy
  regions, ~30 LOC on top of LAC.
- Class-conditional (Mondrian) conformal: separate quantile per class for
  imbalanced problems. Pairs with class weights.
- Worked example wired into one of the synthetic datasets (cubic / residual
  for regression, iris / spiral for classification) showing empirical
  coverage at α = 0.1 and a plot of intervals / set sizes.

### Quantile regression / pinball loss
Explicit per-pattern prediction intervals as an alternative to ensemble-based
epistemic uncertainty. New loss class plus minor adjustments to evaluation.

### Python binding (pybind11)
Massive reach gain. Couple of weeks of real effort. Takes the project from
"C++ research artifact" to "thing people actually use day to day". Worth it
only if there's an audience asking for it; otherwise speculative.

### JSON model export
Interop with other languages and inspection tools. Currently binary-only.
Format: arch + types + softmax + skip pairs + weights, in a stable schema.

## Documentation
- End-to-end walkthrough doc: CSV in to calibrated predictions out, on Pima
  or Iris. Right now the path from "I have a CSV" to "I have a useful model"
  requires reading three examples and the README. One narrative document
  would help adoption a lot.

## Probably skip (libtorch territory)
Listed for clarity so they don't keep coming up:
- SHAP / KernelSHAP for tabular interpretability.
- Categorical embeddings.
- Bayesian / TPE hyperparameter optimization beyond the existing grid
  search.
- Mixed precision / templated Matrix library.

## Performance ceiling notes
On the Pima 8-32-1 / batch-32 benchmark, this lib trains in ~10 ms / 100
epochs vs mlpack at ~0.5 ms (20x gap). The gap is structural, not a
specific bottleneck. mlpack's Armadillo expression templates inline a
compile-time-sized GEMM directly at the call site, fusing forward +
backward + Adam update with no per-call dispatch overhead. nnh's
runtime-sized `Layer` has to dispatch through OpenBLAS's small-kernel
path on every call, which costs ~1 us per dgemm and is the floor.

Things that didn't close the gap (verified, save the time next time):
- **Eigen Dynamic-size embed.** Eigen's `Map<MatrixXd>` ends up calling
  the same `general_matrix_matrix_product` path as OpenBLAS for tiny
  matrices. AVX-512 microkernels in both, near-identical perf. Tried
  and rolled back. Static-size templates (`Matrix<double, M, N>`) are
  the only Eigen path that actually inlines, but they require making
  `Layer` template-parameterised by dimensions -- big rewrite.
- **Skipping BLAS for small sizes.** OpenBLAS already has tuned
  `dgemm_small_kernel_*` paths that beat naive C loops; gating to the
  manual fallback was strictly slower.

Things that *did* help (already shipped in 4.1.0):
- Polynomial tanh / sigmoid (5-6x training speedup; libm tanh wasn't
  vectorising and ate 23% of profile).
- Pre-pack biases for vectorised per-row add.

Plausible further wins (not yet attempted):
- Hand-coded AVX-512 microkernels for the 4-6 hot GEMM shapes that
  dominate small MLPs. ~200 LOC of intrinsics. Probably gets to within
  3-5x of mlpack.
- Compile-time `Layer<InDim, OutDim>` templates. Highest ceiling but
  rewrites Mlp/Factory/parser/serialisation. Loses the runtime-arch
  property. Reach for this only if the benchmark gap is biting a real
  user.

Pre-existing build bug while we're at it: `Layer.cc` and
`MatrixTools.cc` call `cblas_ddot` / `cblas_dcopy` without `#ifdef
USE_BLAS` guards, so `NNH_USE_BLAS=OFF` doesn't link. Easy fix when
someone needs a BLAS-free build.
