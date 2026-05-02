# TODO

## Worth doing (fits mission, cheap)

### Conformal prediction (split conformal)
The headline addition. Distribution-free prediction sets / intervals with
coverage guarantees. Hold out a calibration set, compute residuals (regression)
or non-conformity scores (classification), take the (1-α) quantile, adjust
predictions. ~100 LOC. Plugs straight into the existing ensemble framework
and aligns with the uncertainty-as-first-class theme already established by
the iris / spiral uncertainty examples. Most tabular libraries don't have it.

### Early stopping on validation loss
Learning-curve infra is in place. Add a "stop if val hasn't improved in N
epochs" check inside the trainer loop. ~30 LOC. Every practitioner expects
it. Removes the current dependency on hand-tuning epoch count.

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
