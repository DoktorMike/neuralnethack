# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-08

### Added
- **ReLU activation family**: ReLU, Leaky ReLU (alpha=0.01), ELU (alpha=1.0) layer types
- **Adam/AdamW optimizer**: per-weight adaptive learning rates with configurable weight decay
- **Dropout**: inverted dropout on hidden layers with training/inference mode toggle
- **Model serialization**: binary save/load for Mlp and Ensemble (exact weight preservation)
- **Batch GEMM training**: forward pass, backpropagation, and gradient accumulation via `cblas_dgemm` — one call per layer instead of per-pattern loops
- **L-BFGS optimizer**: replaces full BFGS; O(mn) memory via two-loop recursion instead of O(n^2) inverse Hessian
- **BLAS integration**: optional cblas acceleration for vector/matrix operations with auto-detection
- **Devirtualized activations**: function pointers replace per-neuron virtual dispatch on the hot path
- **CMake build system**: replaces Autotools; single `CMakeLists.txt`, out-of-tree builds, CTest
- **XOR integration test**: trains, evaluates (accuracy/sensitivity/specificity), and tests serialization roundtrip
- Top-level `Makefile` wrapper: `make`, `make test`, `make clean`
- `AGENTS.md` developer guide
- `.gitignore` for build directory

### Changed
- **C++23**: bumped from C++03-era code to C++23 throughout
- **Compiler flags**: `-O3 -march=native -ffast-math -ftree-vectorize -funroll-loops`
- **Ownership model**: raw `new`/`delete` replaced with `unique_ptr` (Mlp layers, Ensemble MLPs, Session members, Trainer::trainNew, Trainer::clone)
- Weights class uses value semantics instead of heap-allocated `vector<double>*`
- Weight update loops (GradientDescent, QuasiNewton) use `__restrict__` raw pointers for SIMD auto-vectorization
- QuasiNewton matrices use contiguous flat storage instead of `vector<vector<double>>`
- Move constructors/assignment added to Mlp and Ensemble
- `std::random_shuffle` replaced with `std::shuffle` + `std::mt19937`
- `std::bind2nd`, `std::unary_function`, `std::binary_function` replaced with lambdas and plain structs
- `testNormaliser` compares numerically with tolerance instead of exact text diff

### Fixed
- Matrix `sub()` (2-arg) was adding instead of subtracting
- Variable shadowing in Normaliser (`uint i` redeclared in inner scope)
- Missing 3rd argument to `Saliency::saliency()` in tests
- `make_pair` with explicit template args (invalid in C++17+)
- Dangling-else warning in CrossEntropy output error

### Removed
- Autotools build system (configure.ac, Makefile.am, m4/, autotools/, aclocal.m4, INSTALL, bootstrap)
- Full n*n BFGS inverse Hessian (replaced by L-BFGS)

---

## [0.9.5] - 2016

- Travis CI integration with Codecov
- README with markdown

## [0.9.0] - 2007-05-09

### Added
- ModelSelector: grid search over weight elimination with cross-validation
- Saliency: true gradient-based saliency derivatives
- RowRange attribute in config files for skipping headers
- Hold-out sampler
- NetworkParser for XML-based model loading
- Parser test, saliency test, GOF test, matrix test
- Normalization option in config (Z-score)
- `modelselector`, `saliency`, `auc` CLI programs
- PrintUtils for formatted output of ensembles and sessions
- Trainer now outputs progress to configurable `ostream`
- `trainNew()` method for training fresh copies of an MLP

### Changed
- Error functions enforce full-batch mode; Trainer handles mini-batching
- Restructured into subdirectories: mlp/, datatools/, evaltools/, matrixtools/, parser/
- DataSet uses index indirection into CoreDataSet (no data copying for cross-validation)

## [0.2.2] - 2004-11-02

### Changed
- DataSet refactored to use CoreDataSet for zero-copy cross-validation views

## [0.2.1] - 2004-10-13

### Fixed
- Documentation and copy constructor fixes in Error/SummedSquare

## [0.2.0] - 2004-09-16

### Added
- Quasi-Newton (BFGS) optimizer with Brent line search
- SummedSquare error function fully tested
- XOR and ECG test datasets

## [0.1.0] - 2004-09-13

### Added
- Gradient descent optimizer with momentum
- Sigmoid, TanH, Linear activation layers
- Multi-layer perceptron (Mlp) with configurable architecture
- Weight management (Weights class)
- DataSet and Pattern classes
- Configuration file parser
- Initial project structure

## [0.0.0] - 2004-06-21

- Initial import: MLP prototype, perceptron prototype, C-based MLP reference implementation
