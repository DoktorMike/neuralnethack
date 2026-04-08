# NeuralNetHack

A fast, lightweight C++23 library for training and evaluating ensembles of multi-layer perceptrons. Zero external dependencies beyond an optional BLAS library. Designed for research, teaching, and embedding in larger systems.

## Features

- **Activations**: Sigmoid, TanH, Linear, ReLU, Leaky ReLU, ELU
- **Optimizers**: SGD with momentum, Adam/AdamW, L-BFGS
- **Loss functions**: Cross-entropy, Summed square error
- **Regularization**: Dropout (inverted), weight elimination
- **Ensembles**: Weighted ensemble of MLPs with bootstrap/cross-split/hold-out sampling
- **Model selection**: Grid search over regularization with cross-validation
- **Feature selection**: Backward elimination via saliency/clamping
- **Evaluation**: ROC/AUC, goodness of fit
- **Serialization**: Binary save/load for models and ensembles
- **Performance**: BLAS-accelerated batch GEMM training, devirtualized activations, SIMD-friendly loops

## Build

```sh
cmake -B build
cmake --build build -j$(nproc)
ctest --test-dir build
```

Requires GCC 13+ or Clang 17+ (C++23). BLAS is auto-detected (install `libopenblas-dev` or similar for best performance). To disable: `cmake -B build -DNNH_USE_BLAS=OFF`.

## Quick start

```cpp
#include "mlp/Mlp.hh"
#include "mlp/Serialization.hh"
#include "Ensemble.hh"

using namespace MultiLayerPerceptron;

// Create a 4-input, 8-hidden (ReLU), 1-output (sigmoid) network
std::vector<uint> arch = {4, 8, 1};
std::vector<std::string> types = {"relu", "logsig"};
Mlp mlp(arch, types, false);

// After training...
saveMlpBinary(mlp, "model.nnh");

// Later...
auto loaded = loadMlpBinary("model.nnh");
const auto& output = loaded->propagate(input);
```

## License

GPL v2+ -- Copyright (C) Michael Green
