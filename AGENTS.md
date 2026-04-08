# NeuralNetHack

A fast, lightweight C++23 library for training and evaluating ensembles of multi-layer perceptrons. Designed for research, teaching, and embedding in larger systems with minimal dependencies.

## Build

```sh
cmake -B build                    # detects BLAS automatically
cmake --build build -j$(nproc)    # parallel build
ctest --test-dir build            # run tests
```

To disable BLAS: `cmake -B build -DNNH_USE_BLAS=OFF`

Requires a C++23 compiler (GCC 13+ or Clang 17+). Optional: cblas/openblas for BLAS-accelerated training.

## Architecture

```
neuralnethack/
  mlp/               Core MLP engine
    Layer.hh/cc         Base layer with batch GEMM propagation, function-pointer activations, dropout
    SigmoidLayer        logsig activation
    TanHypLayer         tansig activation
    LinearLayer         purelin activation
    ReLULayer           relu activation
    LeakyReLULayer      leakyrelu activation (alpha=0.01)
    ELULayer            elu activation (alpha=1.0)
    Mlp.hh/cc           MLP container (unique_ptr<Layer> ownership, batch propagate)
    Trainer.hh/cc       Abstract trainer base (trainNew returns unique_ptr<Mlp>)
    GradientDescent     SGD with momentum and adaptive learning rate
    Adam                Adam/AdamW optimizer with per-weight moments
    QuasiNewton         L-BFGS optimizer (O(mn) memory, two-loop recursion)
    Error.hh/cc         Abstract error base (packBatch utility)
    CrossEntropy        Cross-entropy loss (batch GEMM gradient)
    SummedSquare        Summed square error loss (batch GEMM gradient)
    Serialization       Binary save/load for Mlp and Ensemble
    Weights             Weight storage (value semantics)
  datatools/          Data handling
    DataSet             Index-based view into CoreDataSet
    Pattern             Single input/output pair
    CoreDataSet         Owns the raw pattern data
    Sampler             Abstract sampler (Bootstrap, CrossSplit, HoldOut, Dummy)
    Normaliser          Z-score normalization
  evaltools/          Evaluation
    Roc                 ROC curve computation
    Gof                 Goodness of fit
  matrixtools/        Vector/matrix operations (BLAS-accelerated)
  parser/             Config file and network XML parsing
  Ensemble.hh/cc      Weighted ensemble of MLPs (unique_ptr ownership)
  EnsembleBuilder     Builds ensembles via resampling + training
  ModelEstimator      Cross-validation / bootstrap model estimation
  ModelSelector       Grid search over regularization parameters
  FeatureSelector     Backward elimination feature selection
  Factory             Creates Mlp, Trainer, Error, Sampler from Config
  Config              All configuration parameters
src/                CLI binaries (neuralnethack, ann, modelselector, etc.)
test/               Test suite (7 tests)
```

## Key design decisions

**Activation functions** are devirtualized on the hot path. Layer stores function pointers (`ActivationFn`, `DerivScaleFn`) set at construction time. Virtual `fire()`/`firePrime()` methods remain for per-element access (used by Saliency). The batch path uses the function pointers exclusively.

**Training uses batch GEMM.** `CrossEntropy::gradient()` and `SummedSquare::gradient()` pack the DataSet into contiguous matrices, then use `cblas_dgemm` for forward pass, backpropagation, and gradient accumulation (one call per layer per phase). Non-BLAS fallback uses triple loops. Single-pattern `propagate()` is retained for inference and line search.

**Ownership uses unique_ptr.** Mlp owns its Layers, Ensemble owns its Mlps, Session owns its Ensemble and DataSets. Trainer/Error hold non-owning raw pointers to their collaborators. `trainNew()` and `clone()` return `unique_ptr`.

**L-BFGS** replaces full BFGS. Stores the last 20 (s,y) pairs in a circular buffer. O(mn) memory and compute instead of O(n^2). The two-loop recursion computes H*g without materializing the inverse Hessian.

**Dropout** uses inverted dropout (scale by `1/(1-p)` during training). Applied after activation in both single-pattern and batch paths. Mask is propagated through backprop. Only applied to hidden layers. Toggled via `Mlp::training(bool)`.

## Type strings

Used in config files and architecture specification:

| String | Activation | Optimizer | Error |
|---|---|---|---|
| `logsig` | Sigmoid | | |
| `tansig` | TanH | | |
| `purelin` | Linear | | |
| `relu` | ReLU | | |
| `leakyrelu` | Leaky ReLU | | |
| `elu` | ELU | | |
| `gd` | | SGD+momentum | |
| `adam` | | Adam/AdamW | |
| `qn` | | L-BFGS | |
| `sumsqr` | | | SSE |
| `kullback` | | | Cross-entropy |

## Serialization

```cpp
#include "mlp/Serialization.hh"

// Save
MultiLayerPerceptron::saveMlpBinary(mlp, "model.nnh");
MultiLayerPerceptron::saveEnsembleBinary(ensemble, "ensemble.nne");

// Load
auto mlp = MultiLayerPerceptron::loadMlpBinary("model.nnh");
auto ens = MultiLayerPerceptron::loadEnsembleBinary("ensemble.nne");
```

Binary format: magic bytes + architecture + type strings + softmax flag + weights. Exact double precision preserved.

## Compiler flags

`-std=c++23 -O3 -march=native -ffast-math -ftree-vectorize -funroll-loops -fno-math-errno`

`-ffast-math` relaxes IEEE 754 for better vectorization. This means normalize/unnormalize roundtrips may not be bit-exact (testNormaliser exercises this edge case).

## Adding a new activation function

1. Add type string to `MultiLayerPerceptron.hh` (e.g. `#define MYACT "myact"`)
2. Add static `myactActivation` and `myactDerivScale` functions in `Layer.cc`
3. Wire them in the `Layer` constructor's if-chain
4. Create `MyActLayer.hh/cc` (copy from `ReLULayer`, change the math)
5. Add `make_unique<MyActLayer>` case in `Mlp::createLayers()`
6. Add to `Makefile.am`

## License

MIT
