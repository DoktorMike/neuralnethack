# NeuralNetHack

[![CI](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml/badge.svg)](https://github.com/DoktorMike/neuralnethack/actions/workflows/ci.yml)
![Coverage](./coverage-badge.svg)
![Code Style](./format-badge.svg)
![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A fast, lightweight C++23 library for training and evaluating ensembles of multi-layer perceptrons. Zero external dependencies beyond an optional BLAS library. Designed for research, teaching, and embedding in larger systems.

## Features

- **Activations**: Sigmoid, TanH, Linear, ReLU, Leaky ReLU, ELU
- **Topology**: sequential MLP with optional residual (skip) connections — pre-activation sum merge between same-width layers
- **Optimizers**: SGD with momentum, Adam/AdamW, L-BFGS
- **Loss functions**: Cross-entropy, Summed square error
- **Normalization**: Batch normalization, Layer normalization
- **Regularization**: Dropout (inverted), weight elimination
- **Ensembles**: Weighted ensemble of MLPs with bootstrap/cross-split/hold-out sampling
- **Model selection**: Grid search over regularization with cross-validation
- **Feature selection**: Backward elimination via saliency/clamping
- **Evaluation**: ROC/AUC, Hosmer-Lemeshow goodness of fit, confusion matrix (binary + multi-class) with accuracy / precision / recall / F1 / MCC / balanced accuracy / macro variants, regression metrics (MAE, MAPE, sMAPE, RMSE, R²)
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

## Run from a config file

Don't want to write any C++? You don't have to. The `neuralnethack` binary takes a single config file and does the whole thing: parses the data, normalises it, trains an ensemble (with model selection if you ask for one), evaluates on the test set, and writes everything to disk.

```sh
./build/neuralnethack config.toml
```

There's a working example under `test/pima-indians-diabetes/` if you want something to run right now:

```sh
cd test/pima-indians-diabetes
../../build/neuralnethack config-pima.toml
```

Every output file is suffixed with whatever you put in the `Suffix` field, so you can run a few experiments side by side without clobbering each other:

- `result.<suffix>.txt` -- train/test AUC (cross-entropy for multi-class)
- `networks.<suffix>.xml` -- the trained ensemble, ready to reload
- `outputlist.<suffix>.txt` -- per-pattern model outputs (toggle with `SaveOutputList`)
- `saliencies.<suffix>.txt` -- input saliencies, handy for feature selection
- `myconfig.debug` -- the parsed config, so you can sanity-check what was actually used

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
```

See `test/pima-indians-diabetes/config-pima.toml` for a fully commented version with every field.

#### Migrating from the legacy format

Configs from version 2.x and earlier used a space-separated `{Identifier} {Value} {Value} ...` format with `%` comments. There's a script for that:

```sh
scripts/migrate-config.py old-config.txt -o new-config.toml
```

It handles the field rename, splits the positional tuples (`GDParam`, `AdamParam`, `EnsParam`, `MSParam`, `WeightElim`, `Vary`) into named keys, and drops the result into the right section. Eyeball the output before running it for real.

## License

MIT -- Copyright (c) 2004-2026 Michael Green
