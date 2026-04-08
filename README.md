# NeuralNetHack

[![CI](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml/badge.svg)](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml)
![Coverage](./coverage-badge.svg)

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
#include <vector>
#include <string>

using namespace MultiLayerPerceptron;
using namespace DataTools;

int main()
{
    // -- Build the XOR dataset --
    CoreDataSet core;
    double xor_in[][2]  = {{0,0}, {0,1}, {1,0}, {1,1}};
    double xor_out[][1] = {{0},   {1},   {1},   {0}};
    for (int i = 0; i < 4; ++i) {
        std::vector<double> in(xor_in[i], xor_in[i] + 2);
        std::vector<double> out(xor_out[i], xor_out[i] + 1);
        core.addPattern(Pattern(std::to_string(i), in, out));
    }
    DataSet data;
    data.coreDataSet(core);

    // -- Create a 2-4-1 network (ReLU hidden, sigmoid output) --
    std::vector<uint> arch = {2, 4, 1};
    std::vector<std::string> types = {"relu", "logsig"};
    Mlp mlp(arch, types, false);

    // -- Train with Adam for 2000 epochs --
    SummedSquare error(mlp, data);
    Adam trainer(mlp, data, error, 0.001, 4 /*batch*/, 0.01 /*lr*/);
    trainer.numEpochs(2000);
    trainer.train(std::cout);

    // -- Evaluate --
    for (int i = 0; i < 4; ++i) {
        const auto& out = mlp.propagate(data.pattern(i).input());
        std::cout << xor_in[i][0] << " XOR " << xor_in[i][1]
                  << " = " << out[0] << std::endl;
    }

    // -- Save and reload --
    saveMlpBinary(mlp, "xor.nnh");
    auto loaded = loadMlpBinary("xor.nnh");
    std::cout << "Loaded: " << loaded->propagate(data.pattern(1).input())[0] << std::endl;
}
```

## License

MIT -- Copyright (c) 2004-2026 Michael Green
