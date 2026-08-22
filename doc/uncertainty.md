# Uncertainty quantification

A point prediction without a sense of how much to trust it is half an answer.
NeuralNetHack treats uncertainty as a first-class output, not an afterthought.

## Epistemic vs aleatoric

For an ensemble of classifiers, the entropy of the averaged prediction
decomposes into the part that comes from genuine class overlap (aleatoric,
irreducible) and the part that comes from the members disagreeing (epistemic,
which shrinks with more data and grows out of distribution). This is the
Depeweg et al. 2018 decomposition:

```cpp
#include "evaltools/Uncertainty.hh"
using namespace EvalTools::Uncertainty;

// Per-member probability vectors (e.g. softmax outputs), or pass an Ensemble.
auto d = decomposeEntropy(ensemble, input);
std::cout << "total=" << d.total
          << " aleatoric=" << d.aleatoric
          << " epistemic=" << d.epistemic << "\n";
```

High epistemic with low aleatoric is the classic "the model is guessing
because it has not seen anything like this" signal. See
`examples/iris_ensemble_uncertainty.cc` and `spiral_ensemble_uncertainty.cc`
for the full per-grid-point version, and `cubic_ensemble_uncertainty.cc` for
the regression-spread analogue ([examples.md](examples.md) has the full list).

## Conformal prediction

When you need a distribution-free coverage guarantee rather than a heuristic
score, calibrate a `Conformal` predictor on held-out data and get prediction
sets (classification) or intervals (regression) that contain the truth at the
requested rate. See `evaltools/Conformal.hh`.

## AUC confidence

`Roc::aucBootstrapCI` resamples the evaluation set to put a confidence
interval and a one-sided p-value around the AUC, so "0.82" comes with "and
here is how sure we are it beats chance."

## Adstock kernel uncertainty

Bands on fitted carryover kernels across ensemble members:
[adstock.md](adstock.md#kernel-parameter-uncertainty).
