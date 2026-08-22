# NeuralNetHack

[![CI](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml/badge.svg)](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml)
![Coverage](./coverage-badge.svg)
![Code Style](./format-badge.svg)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is the MLP and ensemble-of-MLPs library I've kept maintained, however infrequent, since 2004. It's small, fast, and stays out of your way: a C++23 core, an optional BLAS dependency, and nothing else. I reach for it on tabular problems where libtorch is overkill and I actually want to see what the optimizer is doing. If that sounds like your kind of thing, read on.

## Features

- **Activations**: Sigmoid, TanH, Linear, ReLU, Leaky ReLU, ELU
- **Topology**: sequential MLP with optional [residual (skip) connections](doc/residual-connections.md), merged pre-activation between same-width layers
- **Lag structure (adstock)**: optional differentiable input stage collapsing per-channel lag windows through parametric carryover kernels (geometric or Weibull), 1-2 trained parameters per channel — built for [marketing-mix-style time-series regression](doc/adstock.md)
- **Output heads**: linear or sigmoid output, plus optional [softmax for multi-class classification](doc/multiclass.md)
- **Optimizers**: SGD with momentum, Adam/AdamW, L-BFGS
- **Loss functions**: cross-entropy, summed square error, with optional per-class weights for imbalanced data
- **Normalization**: batch normalization, layer normalization
- **Regularization**: dropout (inverted), weight elimination, optional non-negative weight constraints (projected gradient, per layer and column range)
- **Ensembles**: weighted ensemble of MLPs with bootstrap, cross-split, or hold-out sampling, trained in parallel via OpenMP
- **Model selection**: grid search over regularization with cross-validation
- **Feature selection**: backward elimination via saliency / clamping
- **Evaluation**: ROC/AUC (with bootstrap confidence interval and a one-sided p-value), Hosmer-Lemeshow goodness of fit, confusion matrix (binary and multi-class) with accuracy / precision / recall / F1 / MCC / balanced accuracy / macro variants, regression metrics (MAE, MAPE, sMAPE, RMSE, R²)
- **Uncertainty**: ensemble spread, total/aleatoric/epistemic entropy decomposition (Depeweg et al. 2018), and split-conformal prediction sets and intervals with [coverage guarantees](doc/uncertainty.md)
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

Against PyTorch on small CPU MLPs the gap is structural: **~28x faster training and ~100x lower inference latency** on Pima, and still ~2.7x faster training at UCI Covertype scale (581k rows, where nnh also posts the lowest inference latency and highest accuracy of the C++ field), because an op-by-op framework pays per-op dispatch costs that compiled C++ doesn't. mlpack still wins small-net training via expression-template fusion. The Covertype table, a width sweep locating the PyTorch crossover (~10-30k parameters, far above typical tabular nets), and all caveats live in [`doc/comparison.md`](doc/comparison.md#speed-and-accuracy-on-real-benchmarks).

## Build

```sh
make          # configure + build
make test     # run all tests
make examples # build the worked examples
make clean    # remove build directories
```

You'll need GCC 13+ or Clang 17+ for C++23. BLAS and OpenMP are auto-detected and optional (install `libopenblas-dev` or similar for best performance). Build options, coverage, formatting, and the release process are in [`doc/development.md`](doc/development.md).

## Single-header amalgamation

If you'd rather not depend on the CMake build, the whole library is also shipped as a single header at `single_include/neuralnethack.hh`. Drop it into your project, follow the stb-style consumer pattern, and you're done -- no library to build, no CMake target to link against:

```cpp
// in exactly ONE translation unit:
#define NNH_IMPLEMENTATION
#include "neuralnethack.hh"

// every other TU just:
#include "neuralnethack.hh"
```

Compile with `g++ -std=c++23 -O2 your_app.cc`. The amalgamation is self-contained: BLAS and OpenMP are *optional*, not required to compile -- if you want them, define `USE_BLAS` / `NNH_USE_OPENMP` and link the matching libraries (`-lopenblas` / `-fopenmp`). How the header is generated and kept in sync: [`doc/development.md`](doc/development.md#single-header-amalgamation).

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

More worked examples (ensembles, residual nets, multi-class, uncertainty, marketing-mix models) are catalogued in [`doc/examples.md`](doc/examples.md).

## Run from a config file

Don't want to write any C++? You don't have to. The `neuralnethack` binary takes a single TOML config and does the whole thing: parses the data, normalises it, trains an ensemble (with model selection if you ask for one), evaluates on the test set, and writes everything to disk.

```sh
cd datasets/pima
../../build/neuralnethack config-pima.toml
```

The full config format, the output files, the other CLI tools, and the legacy-config migration script are documented in [`doc/configuration.md`](doc/configuration.md).

## Documentation

- [Documentation index](doc/README.md)
- [Architecture and design decisions](doc/architecture.md)
- [Configuration and CLI](doc/configuration.md)
- [Development guide](doc/development.md)
- [Examples](doc/examples.md)
- [Uncertainty quantification](doc/uncertainty.md)
- [Adstock / marketing-mix models](doc/adstock.md)
- [Library comparison and benchmarks](doc/comparison.md)

## License

MIT, Copyright (c) 2004-2026 Michael Green
