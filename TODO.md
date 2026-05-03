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
