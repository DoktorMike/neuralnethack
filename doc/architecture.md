# Architecture

High-level layout and the design decisions behind it. For full class diagrams
and data-flow charts, see [design/ARCHITECTURE.md](design/ARCHITECTURE.md).

## Directory layout

```
neuralnethack/
  mlp/               Core MLP engine
    Activation.hh/cc    Activation tags (Sigmoid/TanH/Linear/ReLU/LeakyReLU/ELU) as a std::variant; scalar + batch free functions; tag<->string round-trip
    Layer.hh/cc         Concrete layer with batch GEMM propagation, std::visit activation dispatch, dropout
    Mlp.hh/cc           MLP container (vector<Layer>, batch propagate)
    Trainer.hh/cc       Abstract trainer base (trainNew returns unique_ptr<Mlp>)
    GradientDescent     SGD with momentum and adaptive learning rate
    Adam                Adam/AdamW optimizer with per-weight moments
    QuasiNewton         L-BFGS optimizer (O(mn) memory, two-loop recursion)
    Error.hh/cc         Abstract error base (packBatch utility)
    CrossEntropy        Cross-entropy loss (batch GEMM gradient)
    SummedSquare        Summed square error loss (batch GEMM gradient)
    Serialization       Binary save/load for Mlp and Ensemble (NNH1; NNH2 adds adstock block)
    Weights             Weight storage (value semantics)
    Adstock             Differentiable parametric lag-kernel input stage (geometric/Weibull), params train jointly via all optimizers; boxed mode routes C channels into K shared kernel+saturation boxes via softmax routing (see spec-boxed-adstock.md)
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
test/               Test suite
```

## Key design decisions

**Activation functions** are a `std::variant` tag (`Activation = variant<Sigmoid, TanH, Linear, ReLU, LeakyReLU, ELU>`), not a class hierarchy. Each `Layer` holds one `Activation`. Dispatch goes through `std::visit`, so the compiler inlines the per-element kernel inside each branch. Scalar (`fire`/`firePrime`/`firePrimePrime` + `*FromOutput`) and batch (`applyActivation`/`applyDerivScale`) overloads live as free functions in `Activation.hh/cc`. Parameterized activations carry their own params (`LeakyReLU::alpha=0.01`, `ELU::alpha=1.0`).

**Training uses batch GEMM.** `CrossEntropy::gradient()` and `SummedSquare::gradient()` pack the DataSet into contiguous matrices, then use `cblas_dgemm` for forward pass, backpropagation, and gradient accumulation (one call per layer per phase). Non-BLAS fallback uses triple loops. Single-pattern `propagate()` is retained for inference and line search.

**Ownership uses unique_ptr.** Mlp holds its Layers by value (`vector<Layer>`); Ensemble owns its Mlps, Session owns its Ensemble and DataSets via `unique_ptr`. Trainer/Error hold non-owning raw pointers to their collaborators. `trainNew()` and `clone()` return `unique_ptr`.

**L-BFGS** replaces full BFGS. Stores the last 20 (s,y) pairs in a circular buffer. O(mn) memory and compute instead of O(n^2). The two-loop recursion computes H*g without materializing the inverse Hessian.

**Adstock** is an optional `std::optional<Adstock>` stage on Mlp: raw input carries `channels*lags + passthrough` values, the stage collapses each channel's lag window through a normalized parametric kernel (1-2 unconstrained params per channel), output feeds `arch[0]`. Params are appended to `Mlp::weights()/gradients()` (L-BFGS support for free); Adam and GD have explicit update blocks. `Error::chainAdstock()` backpropagates into the stage via one extra GEMM (deltas w.r.t. layer-0 inputs). Library gradient convention is (1/2) dE/dparam for SSE (delta = t - o, no factor 2) — the adstock grads match it (see testAdstock gradient check). Usage guide: [adstock.md](adstock.md); boxed-mode design: [spec-boxed-adstock.md](spec-boxed-adstock.md).

**Non-negative weights** are an optional projection constraint: `Mlp::nonNegative(layer, colFrom, colTo)` clamps that column range to >= 0 after every trainer update (Adam/GD call `projectNonNegative()`; the flat `weights()` setter projects too, which covers L-BFGS -- whose local weight copy must be read back after the setter, see QuasiNewton.cc). Chosen over softplus/exp reparameterization deliberately: projection reaches exact zeros and has no vanishing gradient at the boundary. Not serialized.

**Dropout** uses inverted dropout (scale by `1/(1-p)` during training). Applied after activation in both single-pattern and batch paths. Mask is propagated through backprop. Only applied to hidden layers. Toggled via `Mlp::training(bool)`.

**Residual (skip) connections** merge the skip source's output into the target layer's pre-activation (`z = W · y_prev + b + y_skip`), before the activation function. Pre-activation rather than post-activation, because the existing activation-derivative formulas all express f'(z) in terms of f(z); putting the skip in pre-activation means that bookkeeping keeps working without extra plumbing. Usage guide: [residual-connections.md](residual-connections.md).

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

Binary format: magic bytes + architecture + type strings + softmax flag + weights. Exact double precision preserved. `NNH2` adds the adstock block; old `NNH1` files still load.

## Compiler flags

`-std=c++23 -O3 -march=native -ffast-math -ftree-vectorize -funroll-loops -fno-math-errno`

`-ffast-math` relaxes IEEE 754 for better vectorization. This means normalize/unnormalize roundtrips may not be bit-exact (testNormaliser exercises this edge case).
