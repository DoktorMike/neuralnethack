# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### [2.0.1](https://github.com/DoktorMike/neuralnethack/compare/v2.0.0...v2.0.1) (2026-04-08)


### Bug Fixes

* add missing sstream include for Clang compatibility ([47ac96f](https://github.com/DoktorMike/neuralnethack/commit/47ac96fef1cfa8226bedf6348ec436becf273533))
* unhide base Sampler::operator= in all Sampler subclasses ([f893152](https://github.com/DoktorMike/neuralnethack/commit/f8931528e90eecd1e408219c46be2a4a6db36b7b))


### Other

* add GitHub Actions workflow for GCC and Clang ([5310a4f](https://github.com/DoktorMike/neuralnethack/commit/5310a4fafa552f90af64a0a3d90577e151118212))


### Documentation

* versioning. ([b3201c6](https://github.com/DoktorMike/neuralnethack/commit/b3201c629e02619e5267f55e112ea87849349ef1))

## 2.0.0 (2026-04-08)


### ⚠ BREAKING CHANGES

* replace Autotools with CMake

### Features

* add ReLU activations, Adam optimizer, dropout, and serialization ([f49e20c](https://github.com/DoktorMike/neuralnethack/commit/f49e20c430e385328e4888428944eab9edeca32d))
* modernize to C++17 with BLAS, unique_ptr, and vectorized hot paths ([62dcc10](https://github.com/DoktorMike/neuralnethack/commit/62dcc100ffdd2a2a5ef70a65f2c0e607e2e486d5))
* wire Adam/AdamW parameters through Config and Parser ([0b068e7](https://github.com/DoktorMike/neuralnethack/commit/0b068e7d4aa4b9b129182021e11c27624dda3551))


### Bug Fixes

* compare numerically with tolerance in testNormaliser ([b06d3a8](https://github.com/DoktorMike/neuralnethack/commit/b06d3a81a4d9166cf958921bbd234dd45f6a7a7b))


### build

* replace Autotools with CMake ([524b7e6](https://github.com/DoktorMike/neuralnethack/commit/524b7e66c4ef9edac69e023795e03da1fb646921))


### Other

* add classification metrics to XOR test ([9bccce8](https://github.com/DoktorMike/neuralnethack/commit/9bccce8d79a6affa48f03fc0b66f613aa090f9f2))
* add XOR integration test ([5637c96](https://github.com/DoktorMike/neuralnethack/commit/5637c96f7c4391d58028158f83e78cb01ae555a2))
* batch forward/backward propagation using GEMM ([885019c](https://github.com/DoktorMike/neuralnethack/commit/885019c4ad361c3cbda7695419aec6b65717f61b))
* contiguous matrix storage and devirtualized activations ([d894caf](https://github.com/DoktorMike/neuralnethack/commit/d894caf8696c7d0b5b3f646837d9e3df5a49a3a5))
* replace full BFGS with L-BFGS (O(mn) memory) ([e17408a](https://github.com/DoktorMike/neuralnethack/commit/e17408a86782453ff2c2e567fba3ca9b7db2effc))


### Documentation

* add AGENTS.md with architecture overview ([7f26a90](https://github.com/DoktorMike/neuralnethack/commit/7f26a90a252a8f2d5b22c161582427df4e5376a5))
* add markdown README ([a4ea3d3](https://github.com/DoktorMike/neuralnethack/commit/a4ea3d3a078db689bb352e012f098c62a30bc52e))
* extend README with XOR training example ([c3b94d2](https://github.com/DoktorMike/neuralnethack/commit/c3b94d28b9b8ca5ef3967e7b874ff87410e7f295))
* merge old ChangeLog into semver CHANGELOG.md ([290c0e6](https://github.com/DoktorMike/neuralnethack/commit/290c0e621dba23ac7e3894a3e4093f8ebb6de1f4))
* update AUTHORS ([467faa1](https://github.com/DoktorMike/neuralnethack/commit/467faa16138b2979503d874e0c8a006d6731909f))
* update README ([d3cf6a7](https://github.com/DoktorMike/neuralnethack/commit/d3cf6a7ec656a1238c82a0f72cf401c0a6138af5))
* update README with features and quick start ([834ff92](https://github.com/DoktorMike/neuralnethack/commit/834ff92ee42f179df664d49aec89753891656e8a))

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
