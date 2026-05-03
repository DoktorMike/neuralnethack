# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [4.1.0](https://github.com/DoktorMike/neuralnethack/compare/v4.0.0...v4.1.0) (2026-05-03)


### Features

* **mlp:** glorot / he weight init by default ([354e5ef](https://github.com/DoktorMike/neuralnethack/commit/354e5ef3e5f2857ee00a7741d91ac035511d1f05))


### Bug Fixes

* **bench:** count mlpack iterations in batch steps, not samples ([b8004db](https://github.com/DoktorMike/neuralnethack/commit/b8004dbd51d0eb24b40cae215e39e0053ad64981))


### Documentation

* **readme:** add "Who is this for?" section + comparison doc ([20b1699](https://github.com/DoktorMike/neuralnethack/commit/20b169957754d042cb8fd9db41992cdd5f59b078))
* **readme:** document single-header amalgamation ([4fd0bb0](https://github.com/DoktorMike/neuralnethack/commit/4fd0bb0531e3e5416b5943fe92f5698ac8dcf62a))
* rewrite "Who is this for?" + comparison in mike's voice ([5322d14](https://github.com/DoktorMike/neuralnethack/commit/5322d140171856eb0729f0172c75319b53c03445))
* strip em-dashes from prose ([cac2085](https://github.com/DoktorMike/neuralnethack/commit/cac20858ee7c5f65b4f13348b5e54cb44f8401ae))
* updated small section in readme. ([d7f83de](https://github.com/DoktorMike/neuralnethack/commit/d7f83de3ff02fb5622067cc718a06eb73a65a0cc))


### Other

* **coverage:** exclude CLI binaries + add backward-pass + pipeline tests ([9457541](https://github.com/DoktorMike/neuralnethack/commit/94575412325347e06036af1e65b9f5a98b9fc42a))
* formatting ([de45d3d](https://github.com/DoktorMike/neuralnethack/commit/de45d3d6b66694ff55b6a9174ea5df5abb591ada))
* formatting ([dcb00b4](https://github.com/DoktorMike/neuralnethack/commit/dcb00b43368a26a264beb957d68cea7887d9ef96))
* **mlp:** polynomial tanh + vectorised bias add ([c9bfc3f](https://github.com/DoktorMike/neuralnethack/commit/c9bfc3f50e59a76baec253906d30258855cb1f48))
* update badges [skip ci] ([828c519](https://github.com/DoktorMike/neuralnethack/commit/828c5191633fe9dd050942e9ec02b1348145f734))
* update badges [skip ci] ([35263a1](https://github.com/DoktorMike/neuralnethack/commit/35263a1493e52be13045ddf00b49d04b865f2931))
* update badges [skip ci] ([2965a18](https://github.com/DoktorMike/neuralnethack/commit/2965a18d5cc39a7c20c18a1eeb4606170aafcfda))
* update badges [skip ci] ([4f69ce4](https://github.com/DoktorMike/neuralnethack/commit/4f69ce4f9a97ef7ac313f4e245386ca38fd93b80))
* update badges [skip ci] ([5f47b7a](https://github.com/DoktorMike/neuralnethack/commit/5f47b7a29dfb901e4789e2ec358cf33b1c671cbc))

## [4.0.0](https://github.com/DoktorMike/neuralnethack/compare/v3.0.0...v4.0.0) (2026-05-03)


### ⚠ BREAKING CHANGES

* **parallel:** random number generation is now nnh::rand (thread-
local mt19937_64) rather than libc drand48. Existing srand48(s) calls
became nnh::rand::seed(s); reproducibility holds for the same seed,
but values differ from the drand48-seeded build.

NNH_OPENMP CMake option (default ON, auto-disabled under NNH_ASAN).
* DataManager::split overloads and Sampler::next now
return by value (std::pair<DataSet, DataSet> or std::vector<DataSet>),
not raw owning pointers. Callers should drop the manual `delete`.
DataSet's shared_ptr<CoreDataSet> + indices vector make these moves
cheap.
* Factory::create* return std::unique_ptr<T>; no more
manual delete. EnsembleBuilder/ModelEstimator setters take unique_ptr.
Error/Trainer (and concrete subclasses) gain unique_ptr-taking ctors
alongside the existing reference ctors.
* **datatools:** DataSet::coreDataSet(CoreDataSet&) is replaced with
DataSet::coreDataSet(std::shared_ptr<CoreDataSet>). External callers
must wrap their CoreDataSet in std::make_shared<CoreDataSet>().
killCoreData() is removed.

Drive-by fixes ASan caught along the way:
- Parser::selectInserter leaked one byte per parsed value (`new char`
  for strtod's end pointer; should have been a stack `char*`).
- Parser::readDataFile read past rowRange.end() when the requested
  rows were exhausted before EOF.

Adds an NNH_ASAN CMake option (off by default) that builds with
-fsanitize=address,undefined and runs tests with detect_leaks=0
(several Sampler / DataManager APIs still use raw owning pointers
and leak; cleaning those up is a separate PR).

### Features

* **cli:** emit per-member learning curves ([6350ae4](https://github.com/DoktorMike/neuralnethack/commit/6350ae409d0b740f47affeba40379c5035bee653))
* **config:** expose early stopping in TOML config ([5ec68b1](https://github.com/DoktorMike/neuralnethack/commit/5ec68b1c85c4729cb5965bcfa26864a070477593))
* **eval:** add ConfusionMatrix with binary and multi-class metrics ([41ffc07](https://github.com/DoktorMike/neuralnethack/commit/41ffc07bb61faac13102ceea4b5cb630e39ed486))
* **eval:** add regression metrics MAE, MAPE, sMAPE, RMSE, R2 ([e0e8fb9](https://github.com/DoktorMike/neuralnethack/commit/e0e8fb9c933777d4d668a36953a0e0218b470c48))
* **evaltools:** add split conformal prediction ([dc24e59](https://github.com/DoktorMike/neuralnethack/commit/dc24e59a059f4409a35dfe666476ae553049ab58))
* **examples:** aleatoric/epistemic + spiral ([5f193eb](https://github.com/DoktorMike/neuralnethack/commit/5f193ebc60428e29edc03abae6e99fe2fb70498a))
* **examples:** make ensemble size configurable via argv[1] ([3598b9d](https://github.com/DoktorMike/neuralnethack/commit/3598b9d2fbe836700fa87ec76dd147e379f7f707))
* **mlp:** add residual (skip) connections with pre-activation sum merge ([5037df4](https://github.com/DoktorMike/neuralnethack/commit/5037df430edec127bf7a5c7dc78f4df518215ba1))
* **mlp:** softmax output layer ([7bb249f](https://github.com/DoktorMike/neuralnethack/commit/7bb249f81fdf29ef5dd5d5740947ee0a594b8bff))
* **parallel:** EnsembleBuilder runs members in parallel via trainer factory ([29b37c8](https://github.com/DoktorMike/neuralnethack/commit/29b37c800087a5e862aa17a148b8edf1ff326051))
* **parallel:** thread-local RNG + OpenMP ensemble training ([553e08e](https://github.com/DoktorMike/neuralnethack/commit/553e08e678b4776f3e3057a2e3bbc2b332bd8fb3))
* **trainer:** early stopping on validation loss ([f4a0f5f](https://github.com/DoktorMike/neuralnethack/commit/f4a0f5f6190668cc6a73abb3a500a3e38b2642f1))
* **trainer:** learning-curve file with val ([fe4038f](https://github.com/DoktorMike/neuralnethack/commit/fe4038feb6024acfbc0bbfc0bca239a2473c1111))


### Bug Fixes

* **cli:** multiclass eval uses accuracy not auc ([47497ef](https://github.com/DoktorMike/neuralnethack/commit/47497efff93828a87d0720623ffe3a774f5f2e97))
* **ensemblebuilder:** widen snprintf buffer for -Wformat-truncation ([c2de1e1](https://github.com/DoktorMike/neuralnethack/commit/c2de1e10928470b4026b0019b4cebe37803bc8ea))
* **test:** avoid use-after-free in testDataManager ([4ce5766](https://github.com/DoktorMike/neuralnethack/commit/4ce5766567d53e71ffe50b7d0acde0e7e5446a45))


### Documentation

* dedicated README section for residual skip connections ([1a05559](https://github.com/DoktorMike/neuralnethack/commit/1a0555965bea60411ec664cde41a5ee792843e3c))
* **examples:** add Amini cubic ensemble uncertainty demo ([ee10bf7](https://github.com/DoktorMike/neuralnethack/commit/ee10bf7a675e1cd1cfb90be197142f7e4ff589d1))
* **examples:** iris ensemble uncertainty + ggplot ([6616a27](https://github.com/DoktorMike/neuralnethack/commit/6616a276bd5824b0cb180aff1c302d46b48e1113))
* **examples:** residual ensemble on XOR ([e276178](https://github.com/DoktorMike/neuralnethack/commit/e27617872a226b76bdf699a672152aaf17ee616a))
* **examples:** residual ensemble uncertainty demo ([4fce163](https://github.com/DoktorMike/neuralnethack/commit/4fce1633b542cc671e4e58eed1d7bd80685d59d8))
* **examples:** residual vs plain deep MLP regression demo ([1b8816e](https://github.com/DoktorMike/neuralnethack/commit/1b8816ec9121a72c932c614bdda95c7601cfb83e))
* **examples:** spiral progress prints per member ([9360c11](https://github.com/DoktorMike/neuralnethack/commit/9360c11d170dba87c8b4f15d4c2fc271deaf9d32))
* fix README quick start for shared_ptr API + show skipFrom ([1d654d7](https://github.com/DoktorMike/neuralnethack/commit/1d654d7c1393ec4f4e7eb9c8b9119064a58b6fc5))
* move TODO to TODO.md, list ASan-found leaks ([c505f79](https://github.com/DoktorMike/neuralnethack/commit/c505f7968a002f078c3814541e154c9ebbd8029e))
* **plot:** show holdout shading, mean+std band, members on uncertainty plot ([95d2235](https://github.com/DoktorMike/neuralnethack/commit/95d2235d0ebf114ca5924a7878386a70d985dc22))
* **readme:** rewrite in mike voice + sync features ([56497f8](https://github.com/DoktorMike/neuralnethack/commit/56497f8c24b1c3197bad929c1e8cffcc02ebae82))
* refresh TODO.md after Factory unique_ptr cleanup ([7d80e5a](https://github.com/DoktorMike/neuralnethack/commit/7d80e5afb170aca457f8090e3f2037ee3933d611))
* **todo:** drop matrix-template, expand ROC entry ([6a9c5c8](https://github.com/DoktorMike/neuralnethack/commit/6a9c5c85e1646e0d54b6fbc9dc36dff0605c3578))
* **todo:** drop residual step 2 ([cc09cbb](https://github.com/DoktorMike/neuralnethack/commit/cc09cbb1119d2e817d3c325a03795ac15712ea3a))
* **todo:** drop time-series + perf entries ([b84042f](https://github.com/DoktorMike/neuralnethack/commit/b84042f12fafffe0702f1d614f9c511e1a5bd5f2))
* **todo:** expand with concrete next-step ideas ([4b4d4b5](https://github.com/DoktorMike/neuralnethack/commit/4b4d4b5fe51e86b40ef7ee16afbd5d1df1eaae0a))


### Other

* DataManager::split + Sampler::next return by value ([5c3e4e5](https://github.com/DoktorMike/neuralnethack/commit/5c3e4e5651b15bf717810b40e9fe2fcdac758cfc))
* **datatools:** DataSet now owns CoreDataSet via shared_ptr ([319f5eb](https://github.com/DoktorMike/neuralnethack/commit/319f5ebc99cb5f0014d6ad5485f69ca7d7e46205))
* **error:** batched outputError ([262b2de](https://github.com/DoktorMike/neuralnethack/commit/262b2dea815ef63e693bbd9128cc1bfa4a635607))
* **error:** reuse packBatch buffers ([6a0ebe6](https://github.com/DoktorMike/neuralnethack/commit/6a0ebe6e65be591c666dd1f7cfa8d3326fadb925))
* formatting ([db59a79](https://github.com/DoktorMike/neuralnethack/commit/db59a79982c76b0ab1a58566480325fb6c3fde3c))
* formatting ([9a02301](https://github.com/DoktorMike/neuralnethack/commit/9a023012c9bf252425e1fa5b0ea0be04b8f263bf))
* formatting ([a9feaf5](https://github.com/DoktorMike/neuralnethack/commit/a9feaf51e328425c1967ed19c509afec82e07df0))
* **gof:** assert chi2 separates good vs bad fit ([ddc86e4](https://github.com/DoktorMike/neuralnethack/commit/ddc86e4ff3949f965f4a05cad1391d67f5155076))
* own Factory ownership graph via unique_ptr ([012b590](https://github.com/DoktorMike/neuralnethack/commit/012b590daa84b3ac81c30404e5dea923a89c1175))
* update badges [skip ci] ([ec00185](https://github.com/DoktorMike/neuralnethack/commit/ec001850b465b6e9d415f02dda6d7f137c2b7251))
* update badges [skip ci] ([2c075d4](https://github.com/DoktorMike/neuralnethack/commit/2c075d45accd7b4af2c40c7d37e09edeb1ddaebd))
* update badges [skip ci] ([f1142f7](https://github.com/DoktorMike/neuralnethack/commit/f1142f71c02540b5386b5205d5fcded26514226b))
* update badges [skip ci] ([3f30ea3](https://github.com/DoktorMike/neuralnethack/commit/3f30ea3163923cb049a09bbb5c9d4b5c9c5ac2e7))
* update badges [skip ci] ([10fdbef](https://github.com/DoktorMike/neuralnethack/commit/10fdbef9c094f90e6ce79a409976d3d1768454b3))
* update badges [skip ci] ([83520ed](https://github.com/DoktorMike/neuralnethack/commit/83520edfebc5a8e6d9038cf866410c1444f1d81e))
* update badges [skip ci] ([e20d567](https://github.com/DoktorMike/neuralnethack/commit/e20d567ecdec1be78a647a356ad676ee124a217c))
* update badges [skip ci] ([166c13a](https://github.com/DoktorMike/neuralnethack/commit/166c13a5d1b6e4682c131c55b3b8a1ec87b32a69))
* update badges [skip ci] ([5edab28](https://github.com/DoktorMike/neuralnethack/commit/5edab280284dd7d92e62703af0b10664a3107748))
* update badges [skip ci] ([abd962b](https://github.com/DoktorMike/neuralnethack/commit/abd962b4abd3abd7d8fff4874f8d17b20c5f9ffa))
* update badges [skip ci] ([b09d8b3](https://github.com/DoktorMike/neuralnethack/commit/b09d8b3bbbd277f3aec2a3c89cbd9f8cf192434c))
* update badges [skip ci] ([94766dd](https://github.com/DoktorMike/neuralnethack/commit/94766dd86631b1807380dd354513a3d07632339a))
* update badges [skip ci] ([a354eb1](https://github.com/DoktorMike/neuralnethack/commit/a354eb15bb918a3eb99829e7445ecf1bf4249e56))
* update badges [skip ci] ([976bff7](https://github.com/DoktorMike/neuralnethack/commit/976bff77bae2d36f6b8166ced2f6fc7e0252376a))
* update badges [skip ci] ([3a3291a](https://github.com/DoktorMike/neuralnethack/commit/3a3291a6ab627a39efc785c37c7204c373c627d7))
* update badges [skip ci] ([f176b9e](https://github.com/DoktorMike/neuralnethack/commit/f176b9e065aaa0aaf734fdf4db8c71585494f0b3))
* update badges [skip ci] ([42d4285](https://github.com/DoktorMike/neuralnethack/commit/42d428535bcffe3d3d4a16282d1f5820e0187450))
* update badges [skip ci] ([b54914e](https://github.com/DoktorMike/neuralnethack/commit/b54914e60bcfc4aebb9cf689b0abff3097bd2a5e))
* update badges [skip ci] ([464a87a](https://github.com/DoktorMike/neuralnethack/commit/464a87a519b60345cca8e6a2f9155fa31c859d70))

## [3.0.0](https://github.com/DoktorMike/neuralnethack/compare/v2.1.1...v3.0.0) (2026-05-01)


### ⚠ BREAKING CHANGES

* existing `.txt` configs no longer parse. Run
`scripts/migrate-config.py old.txt -o new.toml` to convert them.
The full schema is documented in README.md.

### Features

* switch config format to TOML ([b07a1cf](https://github.com/DoktorMike/neuralnethack/commit/b07a1cf3f3a3e492d8ed1c0f304fc0d269bcd143))


### Documentation

* document neuralnethack CLI binary and outputs ([5cf310f](https://github.com/DoktorMike/neuralnethack/commit/5cf310fdb6e8dd972d90059eb1e0647df5be8a96))

### [2.1.1](https://github.com/DoktorMike/neuralnethack/compare/v2.1.0...v2.1.1) (2026-04-08)


### Other

* formatting ([5e9115f](https://github.com/DoktorMike/neuralnethack/commit/5e9115fa9492c2a228fe96aed31b758cbb0e1abc))
* merge badge jobs into single job to prevent push race ([67c9d60](https://github.com/DoktorMike/neuralnethack/commit/67c9d60094827d0119b2e800d04dd8f332f11843))
* update badges [skip ci] ([ed0e614](https://github.com/DoktorMike/neuralnethack/commit/ed0e614cb7b28cda38a6742407be33241d8d988e))
* update format badge [skip ci] ([2d9d31f](https://github.com/DoktorMike/neuralnethack/commit/2d9d31f0b45f9a5f5682eea2208a6e879531b3e1))
* update format badge [skip ci] ([83bfa2d](https://github.com/DoktorMike/neuralnethack/commit/83bfa2ddc28ec05281576ce481239816f54609bf))


### Documentation

* update README with normalization, make targets, config format, and dropout example ([6bef3c2](https://github.com/DoktorMike/neuralnethack/commit/6bef3c2e0d9bbcb4d358bce7b2562cb679d329eb))

## [2.1.0](https://github.com/DoktorMike/neuralnethack/compare/v2.0.1...v2.1.0) (2026-04-08)


### Features

* add BatchNorm and LayerNorm support ([89215e9](https://github.com/DoktorMike/neuralnethack/commit/89215e93fac3f9e5977cb5674166c7aa4d93e2d1))


### Other

* add 5 new test suites to improve coverage (29.8% -> 47.6%) ([4011d95](https://github.com/DoktorMike/neuralnethack/commit/4011d953ddc508ee1cde92e396fd57c721b49cca))
* add code coverage with gcov and Codecov upload ([8c23097](https://github.com/DoktorMike/neuralnethack/commit/8c23097a9b1617e0a5778ad523b0802681b854f0))
* add format check job with badge, plus C++23 and license badges ([563f8fe](https://github.com/DoktorMike/neuralnethack/commit/563f8fea1bb6979e99f72c2e1eb80d8310cc74af))
* add normalization test (BatchNorm, LayerNorm, baseline on XOR) ([93dba9b](https://github.com/DoktorMike/neuralnethack/commit/93dba9b8bf6c17ea4dd694c332b130e2eb361ceb))
* add self-hosted coverage badge to README ([7e7db31](https://github.com/DoktorMike/neuralnethack/commit/7e7db31c5a79a760583dc5e54b3f8704f92f09ac))
* apply clang-format to all source files ([e761369](https://github.com/DoktorMike/neuralnethack/commit/e76136999c1007a8d26f71bc1681e1add33ee090))
* remove Codecov upload, just print coverage summary ([fcdd189](https://github.com/DoktorMike/neuralnethack/commit/fcdd18930d2e1c073479b9bb72625e9628d94abc))
* update coverage badge [skip ci] ([f9665b4](https://github.com/DoktorMike/neuralnethack/commit/f9665b48cca0d5447fcac3ec69217f1947935494))
* use lcov for coverage reports (local HTML + Codecov upload) ([0c198f3](https://github.com/DoktorMike/neuralnethack/commit/0c198f31c9a557eee0e0fa38c2f82871fe85660a))

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
