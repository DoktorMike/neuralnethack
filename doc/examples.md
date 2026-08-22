# Examples

Worked examples live in [`examples/`](../examples/) and build as separate
executables:

```sh
cmake --build build --target xor_residual_ensemble
./build/xor_residual_ensemble        # default ensemble size
./build/xor_residual_ensemble 11     # custom ensemble size
```

`make examples` builds all of them (target `nnh_examples`). The ensemble
examples take an optional positional argument: the number of ensemble members.

| Example | What it shows |
|---|---|
| `xor_residual_ensemble.cc` | Residual MLP (2-4-4-1 with skip 0→1) trained five times from different inits and combined into an `Ensemble` with uniform 1/N weighting. Reports per-member outputs and the ensemble's averaged prediction on each XOR pattern. |
| `residual_vs_plain.cc` | A 12-layer tanh MLP on a synthetic regression task, trained twice with identical init: with and without 5 residual blocks. The residual variant converges to roughly half the MSE of the plain one, because tanh's saturating activation makes gradients vanish across 12 layers without the skip identity path. Loss curves go to `residual_vs_plain.csv`. |
| `residual_ensemble_uncertainty.cc` | Ensemble of 7 residual MLPs trained on `x ∈ [-3, 3]` and evaluated on `x ∈ [-6, 6]`. Inside the training range the members agree (std ≈ 0.01); outside it they extrapolate to wildly different functions (std ≈ 0.5, 30× wider). The growing spread is epistemic uncertainty, made visible. |
| `cubic_ensemble_uncertainty.cc` | Same uncertainty story on the canonical Amini *Deep Evidential Regression* cubic benchmark: `y = x^3 + N(0, 3)` trained on `x ∈ [-4, 4]` and evaluated on `x ∈ [-6, 6]`. ReLU members extrapolate piecewise-linearly into OOD where the truth is super-linear, so the mean prediction undershoots dramatically and the spread balloons. |
| `multiclass_synthetic.cc` | Tiny softmax demo on a synthetic 3-region planar split. No data files, no fuss. Prints train/test accuracy. |
| `multiclass_iris.cc` | Softmax MLP on the UCI Iris dataset (3 classes, 4 features). Loads `datasets/iris/iris.{trn,tst}.tab`, Z-normalises, trains, reports accuracy. |
| `multiclass_wine.cc` | Same for the UCI Wine dataset (3 classes, 13 features). |
| `iris_ensemble_uncertainty.cc` | Ensemble of softmax MLPs on the petal-length / petal-width pair, with the full Depeweg et al. 2018 entropy decomposition: total, aleatoric, and epistemic per grid point. Plot via `scripts/plotexamplesresultdata.r`. |
| `spiral_ensemble_uncertainty.cc` | Three-arm Archimedean spiral, same decomposition. Useful as a sanity check that the network is doing what you think it's doing. |
| `mmm_adstock.cc` | Marketing-mix regression with the `Adstock` lag-kernel stage: recovers known geometric/Weibull carryover kernels through a saturating net, reports holdout R². See [adstock.md](adstock.md). |
| `mmm_boxed.cc` | Boxed adstock at scale: 50 insertion types (10 media × 5 messages), 156 weekly obs, seasonality + unemployment + trend. Recovers the delayed peak and the S-shaped Hill exponent, and shows where 156 rows stop identifying middle carryover regimes — the stability readout flags exactly those channels. See [adstock.md](adstock.md#boxed-mode). |

Related deep dives: [residual-connections.md](residual-connections.md),
[multiclass.md](multiclass.md), [uncertainty.md](uncertainty.md),
[adstock.md](adstock.md).
