# NeuralNetHack

[![CI](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml/badge.svg)](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml)
![Coverage](./coverage-badge.svg)
![Code Style](./format-badge.svg)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A fast, lightweight C++23 library for training and evaluating ensembles of multi-layer perceptrons. Zero external dependencies beyond an optional BLAS library. Designed for research, teaching, and embedding in larger systems.

## Features

- **Activations**: Sigmoid, TanH, Linear, ReLU, Leaky ReLU, ELU
- **Optimizers**: SGD with momentum, Adam/AdamW, L-BFGS
- **Loss functions**: Cross-entropy, Summed square error
- **Normalization**: Batch normalization, Layer normalization
- **Regularization**: Dropout (inverted), weight elimination
- **Ensembles**: Weighted ensemble of MLPs with bootstrap/cross-split/hold-out sampling
- **Model selection**: Grid search over regularization with cross-validation
- **Feature selection**: Backward elimination via saliency/clamping
- **Evaluation**: ROC/AUC, goodness of fit
- **Serialization**: Binary save/load for models and ensembles
- **Performance**: BLAS-accelerated batch GEMM training, devirtualized activations, SIMD-friendly loops

## Build

```sh
make          # configure + build
make test     # run all tests
make coverage # build with gcov, run tests, generate HTML report
make format   # apply clang-format to all source files
make clean    # remove build directories
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

    // -- Optional: enable BatchNorm and dropout --
    mlp.normType(NormType::BatchNorm);
    mlp.dropoutRate(0.1);

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

## Configuration file format

For use with the CLI tools (`neuralnethack`, `ann`, `modelselector`, etc.):

```
Suffix      myrun
Filename    data/train.tab
IdCol       0
InCol       1-8
OutCol      9
RowRange    0
FilenameT   data/test.tab
IdColT      0
InColT      1-8
OutColT     9
RowRangeT   0
PType       class
NLay        3
Size        8 4 1
ActFcn      relu logsig
ErrFcn      kullback
MinMethod   adam
MaxEpochs   2000
AdamParam   0.001 0.9 0.999 1e-8 0.01
WeightElim  0 0.01 1
EnsParam    bagg 5 2 rnd 0
MSParam     cv 3 5 rnd 0.2
Seed        42
Normalization Z
```

## License

MIT -- Copyright (c) 2004-2026 Michael Green
