# NeuralNetHack

[![CI](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml/badge.svg)](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml)
![Coverage](./coverage-badge.svg)
![Code Style](./format-badge.svg)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is the MLP and ensemble-of-MLPs library I've kept maintained, however infrequent, since 2004. It's small, fast, and stays out of your way: a C++23 core, an optional BLAS dependency, and nothing else. I reach for it on tabular problems where libtorch is overkill and I actually want to see what the optimizer is doing. If that sounds like your kind of thing, read on.

## Features

- **Activations**: Sigmoid, TanH, Linear, ReLU, Leaky ReLU, ELU
- **Topology**: sequential MLP with optional residual (skip) connections, merged pre-activation between same-width layers
- **Lag structure (adstock)**: optional differentiable input stage collapsing per-channel lag windows through parametric carryover kernels (geometric or Weibull), 1-2 trained parameters per channel — built for marketing-mix-style time-series regression
- **Output heads**: linear or sigmoid output, plus optional softmax for multi-class classification
- **Optimizers**: SGD with momentum, Adam/AdamW, L-BFGS
- **Loss functions**: cross-entropy, summed square error, with optional per-class weights for imbalanced data
- **Normalization**: batch normalization, layer normalization
- **Regularization**: dropout (inverted), weight elimination
- **Ensembles**: weighted ensemble of MLPs with bootstrap, cross-split, or hold-out sampling, trained in parallel via OpenMP
- **Model selection**: grid search over regularization with cross-validation
- **Feature selection**: backward elimination via saliency / clamping
- **Evaluation**: ROC/AUC (with bootstrap confidence interval and a one-sided p-value), Hosmer-Lemeshow goodness of fit, confusion matrix (binary and multi-class) with accuracy / precision / recall / F1 / MCC / balanced accuracy / macro variants, regression metrics (MAE, MAPE, sMAPE, RMSE, R²)
- **Uncertainty**: ensemble spread, total/aleatoric/epistemic entropy decomposition (Depeweg et al. 2018), and split-conformal prediction sets and intervals with coverage guarantees
- **Diagnostics**: per-trainer learning-curve files (train and validation error per epoch), gnuplot-friendly
- **Serialization**: binary save/load for models and ensembles
- **Performance**: BLAS-accelerated batch GEMM training, devirtualized activations, SIMD-friendly loops
- **Distribution**: ships as a CMake static library *and* a generated single-header amalgamation (stb-style) for drop-in use

## Who is this for?

If you're doing tabular regression or classification in C++ and you actually care about *how confident* the model is (ensembles for spread, conformal sets for coverage guarantees, an explicit aleatoric/epistemic split), this is one of the few C++ libraries that treats that as the point rather than an afterthought. I built it for that and I keep using it for that.

It's not a libtorch replacement and I'm not going to pretend it is. Reach for something else if:

- you need GPUs, big tensors, or anything Transformer-shaped → [**flashlight**](https://github.com/flashlight/flashlight).
- you want trees, SVMs, k-means, or Python/Julia bindings alongside the NN bits → [**mlpack**](https://github.com/mlpack/mlpack).
- you just want a header-only CNN demo → [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) (caveat: it's been quiet since around 2020).
- you're allowed to use Python → **scikit-learn** or **PyTorch**. Don't be a hero.

If you want the receipts, a full feature-by-feature comparison with the same libraries lives in [`doc/comparison.md`](doc/comparison.md).

## Speed

At realistic data scale this library is fast. Head-to-head against mlpack, tiny-dnn, and PyTorch under identical config (same architecture, Adam lr=0.01, batch 32, medians over trials; harness in [`bench/`](bench/)):

**Pima Indians Diabetes** (768 x 8, binary, arch 8-32-1, 100 epochs):

| lib | train (s) | inference (us / sample) | test accuracy |
|---|---|---|---|
| mlpack | 0.001 | 0.10 | 0.768 |
| neuralnethack | 0.010 | **0.08** | 0.744 |
| tiny-dnn | 0.262 | 0.93 | 0.741 |
| PyTorch (eager) | 0.280 | 7.76 | 0.744 |
| PyTorch (compile) | 0.538 | 13.47 | 0.744 |

**UCI Covertype** (581k x 54, 7-class softmax, arch 54-128-7, 5 epochs):

| lib | train (s) | inference (us / sample) | test accuracy |
|---|---|---|---|
| neuralnethack | 5.56 | **0.82** | **0.829** |
| mlpack | 1.56* | 5.64 | 0.754 |
| tiny-dnn | 52.5 | 2.01 | 0.823 |
| PyTorch (eager) | 14.8 | 7.85 | 0.832 |
| PyTorch (compile) | 21.5 | 15.43 | 0.832 |

\* stopped early on ensmallen's convergence tolerance; treat as a lower bound (see [`doc/comparison.md`](doc/comparison.md#speed-and-accuracy-on-real-benchmarks)).

Against PyTorch on small CPU MLPs the gap is structural: **~28x faster training and ~100x lower inference latency** on Pima, and still ~2.7x faster training at Covertype scale, because an op-by-op framework pays per-op dispatch costs that compiled C++ doesn't (`torch.compile` makes it worse at these sizes — per-call guard checks outweigh fusion). A width sweep (`bench/sweep.sh`) puts the training crossover around 10-30k parameters, far above typical tabular nets, and finds no batch-1 inference crossover even at 268k parameters:

| hidden H | params | nnh s/epoch | torch s/epoch | nnh infer (us) | torch infer (us) |
|---|---|---|---|---|---|
| 32 | 2.1k | 0.0075 | 0.0246 | 0.4 | 8.3 |
| 128 | 8.4k | 0.0113 | 0.0135 | 0.8 | 8.5 |
| 512 | 33k | 0.0436 | 0.0233 | 5.3 | 12.3 |
| 1024 | 67k | 0.0817 | 0.0299 | 5.0 | 17.9 |
| 4096 | 268k | 0.260 | 0.152 | 33.6 | 52.5 |

mlpack still wins small-net training via expression-template fusion. Full analysis and caveats live in [`doc/comparison.md`](doc/comparison.md#speed-and-accuracy-on-real-benchmarks).

## Build

```sh
make          # configure + build
make test     # run all tests
make coverage # build with gcov, run tests, generate HTML report
make format   # apply clang-format to all source files
make clean    # remove build directories
```

You'll need GCC 13+ or Clang 17+ for C++23. BLAS is auto-detected (install `libopenblas-dev` or similar for best performance), and you can switch it off with `cmake -B build -DNNH_USE_BLAS=OFF` if you really want to.

OpenMP is also auto-detected and used to train ensemble members in parallel. Control with `OMP_NUM_THREADS` at run time, or disable at configure time with `cmake -B build -DNNH_OPENMP=OFF`.

## Single-header amalgamation

If you'd rather not depend on the CMake build, the whole library is also shipped as a single header at `single_include/neuralnethack.hh`. Drop it into your project, follow the stb-style consumer pattern, and you're done -- no library to build, no CMake target to link against:

```cpp
// in exactly ONE translation unit:
#define NNH_IMPLEMENTATION
#include "neuralnethack.hh"

// every other TU just:
#include "neuralnethack.hh"
```

Compile with `g++ -std=c++23 -O2 your_app.cc`. The amalgamation is self-contained: BLAS and OpenMP are *optional*, not required to compile -- if you want them, define `USE_BLAS` / `NNH_USE_OPENMP` and link the matching libraries (`-lopenblas` / `-fopenmp`).

The header is regenerated by `scripts/amalgamate.py` (topo-sorts the public headers by include deps, dedupes system includes, gates the implementation under `NNH_IMPLEMENTATION`):

```sh
make single-include   # regenerate + smoke-compile
```

CI runs the same target on every PR and fails if `single_include/neuralnethack.hh` ends up out of sync with the source tree, so the committed artifact always matches the rest of the repo.

## Quick start: learning XOR

```cpp
#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/Serialization.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

int main()
{
    // Build the XOR dataset
    auto core = std::make_shared<CoreDataSet>();
    double xor_in[][2]  = {{0,0}, {0,1}, {1,0}, {1,1}};
    double xor_out[][1] = {{0},   {1},   {1},   {0}};
    for (int i = 0; i < 4; ++i) {
        std::vector<double> in(xor_in[i], xor_in[i] + 2);
        std::vector<double> out(xor_out[i], xor_out[i] + 1);
        core->addPattern(Pattern(std::to_string(i), in, out));
    }
    DataSet data;
    data.coreDataSet(core);

    // 2-4-1 network with ReLU hidden and sigmoid output
    std::vector<uint> arch = {2, 4, 1};
    std::vector<std::string> types = {"relu", "logsig"};
    Mlp mlp(arch, types, false);

    // Optional: enable BatchNorm and a bit of dropout
    mlp.normType(NormType::BatchNorm);
    mlp.dropoutRate(0.1);

    // Train with Adam for 2000 epochs
    SummedSquare error(mlp, data);
    Adam trainer(mlp, data, error, 0.001, 4 /*batch*/, 0.01 /*lr*/);
    trainer.numEpochs(2000);
    trainer.train(std::cout);

    // Evaluate
    for (int i = 0; i < 4; ++i) {
        const auto& out = mlp.propagate(data.pattern(i).input());
        std::cout << xor_in[i][0] << " XOR " << xor_in[i][1]
                  << " = " << out[0] << std::endl;
    }

    // Save and reload
    saveMlpBinary(mlp, "xor.nnh");
    auto loaded = loadMlpBinary("xor.nnh");
    std::cout << "Loaded: " << loaded->propagate(data.pattern(1).input())[0] << std::endl;
}
```

## Residual (skip) connections

Each layer can optionally take a residual input from an earlier layer. The skip source's output is added element-wise into the target layer's pre-activation, before the activation function:

```
z = W · y_prev + b + y_skip       // skip added before activation
y = act(z)
```

Pre-activation rather than post-activation, because the existing activation-derivative formulas all express f'(z) in terms of f(z). Putting the skip in pre-activation means that bookkeeping keeps working without any extra plumbing.

Two hard constraints:

- **Source must come earlier in the chain.** A layer can only skip from a layer with a smaller index. `skipFrom()` aborts otherwise.
- **Source and target must have the same width.** The merge is element-wise, so the shapes have to line up.

### Layer indexing

This is the part that trips people up. Indices count up from the first hidden layer. The input vector is *not* a layer. So for an architecture `[n_in, n_h1, n_h2, n_h3, n_out]`:

```
arch:    [n_in,    n_h1,    n_h2,    n_h3,    n_out]
                    ^        ^        ^        ^
                  layer 0  layer 1  layer 2  layer 3 (output)
```

Which means in `arch = [2, 4, 4, 1]` (input plus two width-4 hidden plus width-1 output), layers 0 and 1 are both width 4 and can be wired together with a skip.

### From C++

```cpp
std::vector<uint> arch = {2, 4, 4, 1};
std::vector<std::string> types = {"tansig", "tansig", "logsig"};
Mlp mlp(arch, types, false);

// Layer 1's pre-activation gets layer 0's output added in.
mlp.skipFrom(/*target=*/1, /*source=*/0);
```

Pass `-1` as the source to clear an existing skip on a given target.

### From a TOML config

Under `[network]`, add `skip_connections` as an array of `[target, source]` pairs:

```toml
[network]
size = [2, 4, 4, 1]
activations = ["tansig", "tansig", "logsig"]
error_fcn = "kullback"
skip_connections = [[1, 0]]   # layer 1 receives skip from layer 0
```

One skip source per target layer (later entries for the same target overwrite earlier ones). Multiple targets are free to share the same source.

For a full worked example with an ensemble of residual MLPs, see `examples/xor_residual_ensemble.cc`.

## Multi-class classification (softmax)

For K-way classification, use a linear output layer of width K and turn softmax on. Pair it with the cross-entropy loss and the (target - output) shortcut at the output layer gives you exactly the right gradient (no derivative on softmax to apply explicitly, the math cancels).

From C++:

```cpp
std::vector<uint> arch = {4, 8, 3};                  // 4-feature input, 3 classes
std::vector<std::string> types = {"tansig", "purelin"};
Mlp mlp(arch, types, /*softmax=*/true);
```

From a TOML config:

```toml
[network]
size = [4, 8, 3]
activations = ["tansig", "purelin"]
softmax = true
error_fcn = "kullback"
```

Targets should be one-hot encoded (one column per class in the data file, `out_cols = "6-8"` for example). Worked examples in `examples/multiclass_iris.cc`, `examples/multiclass_wine.cc`, and `examples/multiclass_synthetic.cc`.

## Lag structures / adstock (marketing-mix models)

Time-series regression where today's input keeps affecting the response for many periods (ad spend, promotions, pricing) usually gets modeled by feeding L lags per channel into the network — L free weights per channel per neuron that then need pruning. The `Adstock` stage collapses each channel's lag window through a **normalized parametric carryover kernel** instead, with the kernel parameters trained jointly with the weights by any of the optimizers:

```cpp
#include "mlp/Adstock.hh"

// Input per pattern: [c0 lag0..lag27, c1 lag0..lag27, c2 lag0..lag27, seasonality]
// arch[0] = channels + passthrough covariates = 4
Mlp mlp({4, 8, 1}, {"tansig", "purelin"}, false);
mlp.adstock(Adstock(/*channels=*/3, /*lags=*/28, /*passthrough=*/1,
                    Adstock::Kernel::Geometric)); // or Kernel::Weibull

// train exactly as usual -- Adam / SGD / L-BFGS all update the kernel params
SummedSquare loss(mlp, data);
Adam opt(mlp, data, loss, 0.0, 32, 0.01);
opt.train(std::cout);

auto w = mlp.adstock()->kernelWeights(0); // fitted carryover curve, channel 0
```

**Boxed mode** scales this to many channels (say 100 channels, 156 weekly observations — where 100 free lambdas would be hopeless): K shared kernel "boxes" (short / medium / long carryover) plus a learned per-channel routing `pi_c = softmax(logits_c / tau)` that mixes the boxes. One routing gates both the carryover kernel and an optional per-box Hill saturation `a^n / (a^n + s^n)` (trainable half-saturation and exponent; n > 1 learns an S-shaped response), so 100 media insertion types collapse to K effective ones. Each box's parameters pool ~C/K channels' worth of data — the pooling is the prior:

```cpp
Mlp mlp({100 + P, 1}, {"purelin"}, false);
mlp.adstock(Adstock(/*channels=*/100, /*lags=*/28, /*passthrough=*/P,
                    Adstock::Kernel::Weibull, /*nBoxes=*/3,
                    Adstock::Saturation::Hill));

// train with the entropy penalty OFF until routing stabilizes...
opt.train(std::cout);
// ...then enable it to harden the assignments toward one-hot
mlp.adstock()->entropyPenalty(0.01);
opt.train(std::cout);

auto box = mlp.adstock()->boxAssignments(); // channel -> box
auto pi  = mlp.adstock()->routingProbs(7);  // soft routing, channel 7
```

Do not enable the entropy penalty from the first epoch — it hardens the routing before the boxes separate and locks in chance-level assignments; warm up with it off (measured: 12/12 channels routed correctly with warmup, chance without). `summarizeBoxedAdstock` extends the ensemble summary to boxed stages with label-switching-safe box bands plus **assignment stability** — "channel 7 routed long in 9/10 members". Full worked example: `examples/mmm_boxed.cc` (50 insertion types = 10 media × 5 messages, 156 weekly obs, seasonality + unemployment + trend; recovers the delayed peak and the S-shaped Hill exponent, and shows where 156 rows stop identifying middle carryover regimes — the stability readout flags exactly those channels). Design rationale and V2 (feature-based routing) in [`doc/spec-boxed-adstock.md`](doc/spec-boxed-adstock.md).

Kernel-parameter uncertainty comes from the ensemble machinery: train members on bootstrap resamples, then summarize the spread of fitted kernels across members:

```cpp
#include "evaltools/Uncertainty.hh"

auto s = EvalTools::Uncertainty::summarizeAdstock(ensemble, /*alpha=*/0.1);
// s.paramMean / s.paramLower / s.paramUpper: 90% band per channel on the
// natural scale (geometric lambda, or Weibull shape and scale)
// s.weightMean / s.weightLower / s.weightUpper: per-lag bands on the
// carryover curve itself
```

Kernels: **geometric** (1 param/channel, monotone decay) and **Weibull** (2 params/channel, allows a delayed peak). 84 lagged inputs above cost 3 trained lag parameters instead of hundreds of free weights — nothing to prune, and the fitted kernels are directly interpretable as carryover curves. The MLP behind the stage learns saturation and channel interactions. Gradients flow through one extra GEMM; the stage serializes with the model (`NNH2` format, old `NNH1` files still load). Worked example with recovered kernels and holdout R²: `examples/mmm_adstock.cc`.

**Choosing the window length.** The kernel is truncated and renormalized over L lags, so carryover beyond the window is invisible to the model. For geometric decay the truncated tail mass is ≈ λ^L: λ = 0.8 loses 4.4% at L = 14 but 0.2% at L = 28; λ = 0.9 needs L ≈ 44 for a 1% tail. Size L for the slowest decay you consider plausible (L ≥ ln ε / ln λ_max) — the cost is only `channels × L` inputs per pattern and O(L) per GEMM row, and an over-long window costs nothing statistically since the kernel stays a 1-2 parameter family regardless of L. The fixed window is a deliberate trade against recursive adstock (`a_t = x_t + λa_{t-1}`): infinite memory, but stateful patterns would break bootstrap/cross-split resampling and shuffled batches.

## Uncertainty quantification

A point prediction without a sense of how much to trust it is half an answer.
NeuralNetHack treats uncertainty as a first-class output, not an afterthought.

**Epistemic vs aleatoric.** For an ensemble of classifiers, the entropy of
the averaged prediction decomposes into the part that comes from genuine class
overlap (aleatoric, irreducible) and the part that comes from the members
disagreeing (epistemic, which shrinks with more data and grows out of
distribution). This is the Depeweg et al. 2018 decomposition:

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
the regression-spread analogue.

**Conformal prediction.** When you need a distribution-free coverage
guarantee rather than a heuristic score, calibrate a `Conformal` predictor on
held-out data and get prediction sets (classification) or intervals
(regression) that contain the truth at the requested rate. See
`evaltools/Conformal.hh`.

**AUC confidence.** `Roc::aucBootstrapCI` resamples the evaluation set to put
a confidence interval and a one-sided p-value around the AUC, so "0.82" comes
with "and here is how sure we are it beats chance."

## Examples

Worked examples live in `examples/` and build as separate executables:

```sh
cmake --build build --target xor_residual_ensemble
./build/xor_residual_ensemble        # default ensemble size
./build/xor_residual_ensemble 11     # custom ensemble size
```

The ensemble examples take an optional positional argument: the number of ensemble members.

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

## Run from a config file

Don't want to write any C++? You don't have to. The `neuralnethack` binary takes a single config file and does the whole thing: parses the data, normalises it, trains an ensemble (with model selection if you ask for one), evaluates on the test set, and writes everything to disk.

```sh
./build/neuralnethack config.toml
```

There's a working example under `datasets/pima/` if you want something to run right now:

```sh
cd datasets/pima
../../build/neuralnethack config-pima.toml
```

For multi-class classification, similar configs ship with the iris and wine datasets:

```sh
cd datasets/iris   && ../../build/neuralnethack config-iris.toml
cd datasets/wine   && ../../build/neuralnethack config-wine.toml
```

Every output file is suffixed with whatever you put in the `suffix` field, so you can run a few experiments side by side without clobbering each other:

- `result.<suffix>.txt`: train/test AUC (binary) or accuracy (multi-class).
- `networks.<suffix>.xml`: the trained ensemble, ready to reload.
- `outputlist.<suffix>.txt`: per-pattern model outputs (toggle with `save_output_list`).
- `saliencies.<suffix>.txt`: input saliencies, handy for feature selection.
- `myconfig.debug`: the parsed config, so you can sanity-check what was actually used.
- `<curve>_NNN.dat` (when `output.learning_curve_file` is set): per-member learning curves, one row per epoch with `epoch  trainErr  valErr`. The validation error comes from each member's out-of-bag split.

The other CLI tools (`ann`, `modelselector`, `featureselector`, `saliency`, `auc`) all read the same config format. Pick the one that matches what you're after.

### Config file format

Configs are TOML. Sections group related settings, named keys replace the old positional tuples (no more counting arguments), and comments use `#`. A minimal binary-classification config looks like this:

```toml
suffix = "myrun"
seed = 42
normalization = "Z"          # "Z" or "no"
problem_type = "class"       # "class" or "regr"

[data.train]
file = "data/train.tab"
id_col = 0                   # 0 = no id column
in_cols = "1-8"              # range string, 1-indexed
out_cols = "9"
row_range = "0"              # "0" = all rows

[data.test]
file = "data/test.tab"
id_col = 0
in_cols = "1-8"
out_cols = "9"
row_range = "0"

[network]
size = [8, 4, 1]
activations = ["relu", "logsig"]   # one per non-input layer
error_fcn = "kullback"             # "sumsqr" or "kullback"
softmax = false                    # true for multi-class with linear output
weight_init = "glorot"             # "glorot" (default) or "legacy_uniform".
                                   # glorot picks Xavier uniform for saturating
                                   # activations and He uniform for ReLU-family,
                                   # both scaled to fan-in / fan-out. Biases
                                   # initialise to zero. legacy_uniform is the
                                   # pre-4.1.0 U(-0.5, 0.5) draw, kept for
                                   # back-compat with serialised models.
# Optional residual connections: each entry is [target_layer, source_layer]
# (0-indexed, source < target, both layers must have matching width).
# skip_connections = [[2, 0]]

[training]
method = "adam"              # "gd", "adam", "qn"
max_epochs = 2000

[training.adam]
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01

[training.early_stopping]
patience = 0                 # 0 disables (default). When > 0 the trainer stops
min_delta = 0.0              # if val loss has not improved by min_delta for
                             # `patience` recorded epochs, and the model weights
                             # are restored to the best-val snapshot.

[regularization.weight_elim]
enabled = false
alpha = 0.01
w0 = 1.0

[ensemble]
method = "bagg"              # "bagg", "cs"
runs = 5
parts = 2
split = "rnd"                # "rnd" or "ser"
vary_weights = false

[model_selection]
method = "cv"                # "cv", "boot", "hold", "none"
runs = 3
parts = 5
split = "rnd"
fraction = 0.2

[output]
save_session = true
save_output_list = true
# learning_curve_file = "curve.dat"   # optional, per-member files <stem>_NNN.<ext>
```

See `datasets/pima/config-pima.toml` for a fully commented version with every field.

#### Migrating from the legacy format

Configs from version 2.x and earlier used a space-separated `{Identifier} {Value} {Value} ...` format with `%` comments. There's a script for that:

```sh
scripts/migrate-config.py old-config.txt -o new-config.toml
```

It handles the field rename, splits the positional tuples (`GDParam`, `AdamParam`, `EnsParam`, `MSParam`, `WeightElim`, `Vary`) into named keys, and drops the result into the right section. Eyeball the output before running it for real, since the legacy format had a few oddities.

## License

MIT, Copyright (c) 2004-2026 Michael Green
