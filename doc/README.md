# Documentation

Deep-dive reference for NeuralNetHack. Start with the project
[README](../README.md) for the overview and quick start.

## Guides

| Doc | Topic |
|---|---|
| [architecture.md](architecture.md) | Directory layout, key design decisions, type strings, serialization, compiler flags |
| [configuration.md](configuration.md) | TOML config format, CLI tools, output files, legacy-config migration |
| [development.md](development.md) | Build options, amalgamation, adding activations, release process, scripts |
| [examples.md](examples.md) | The worked examples and what each one demonstrates |
| [residual-connections.md](residual-connections.md) | Skip connections: semantics, layer indexing, C++ and TOML usage |
| [multiclass.md](multiclass.md) | Softmax multi-class classification |
| [adstock.md](adstock.md) | Lag structures / adstock for marketing-mix models, boxed mode, window sizing |
| [uncertainty.md](uncertainty.md) | Entropy decomposition, conformal prediction, AUC confidence |
| [comparison.md](comparison.md) | Feature and speed comparison vs tiny-dnn, mlpack, flashlight, PyTorch |

## Design documents

| Doc | Topic |
|---|---|
| [design/ARCHITECTURE.md](design/ARCHITECTURE.md) | Full class diagrams and data-flow charts (Mermaid) |
| [spec-boxed-adstock.md](spec-boxed-adstock.md) | Boxed-adstock design rationale and V2 (feature-based routing) |
