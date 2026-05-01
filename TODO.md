# TODO

## Memory safety / lifetime cleanup

These were exposed by the `NNH_ASAN=ON` build (`cmake -B build-asan -DNNH_ASAN=ON`).
Every entry is the same shape: a function returns a raw `T*` that the caller is
expected to `delete`, and not every caller does. Fix is to switch the API to
`std::unique_ptr<T>` (or in some cases return by value once we trust the move
semantics). Mechanical but touches a lot of files.

### Factory returns raw owning pointers
`neuralnethack/Factory.cc` is the worst offender — `createMlp`, `createError`,
`createTrainer`, `createSampler`, `createEnsembleBuilder`, `createModelEstimator`
all return raw owning pointers. Convert each to `std::unique_ptr<T>`. Touches
every test that builds a model via the factory plus the `src/*.cc` binaries.
Be careful: `createError` internally calls `createMlp` and the trainer/error
graph holds references back into each other. Probably needs the trainer/error
ownership rewired (the `Error` should own its `Mlp`, the `Trainer` should own
its `Error`) so the lifetime is a clean tree.

### DataManager::split / splitN return raw owning pointers
`DataTools::DataManager::split` returns `pair<DataSet, DataSet>*` and the K-fold
variant returns `vector<DataSet>*`. Convert to value or `unique_ptr`. Used by
all the samplers — `BootstrapSampler`, `CrossSplitSampler`, `HoldOutSampler`,
`DummySampler` — so the change cascades.

### BootstrapSampler::reset leaks the old DataManager
`BootstrapSampler::reset()` does `theDataManager = new DataManager()` without
freeing the previous one. The other samplers may have similar patterns; audit
`reset()` across all of them when fixing.

### Evaluator owned by Roc as raw `new`
`Roc` allocates a `new Evaluator()` and deletes it in its destructor, which is
fine when `Roc` itself is owned correctly. Several tests construct a `Roc`
that is then leaked by raw-pointer indirection (e.g. via the factories above).
Fixing the factories should fix this transitively. If it doesn't, switch
`Roc::theEval` to `unique_ptr<Evaluator>`.

### ASan setup notes
- `NNH_ASAN=ON` builds with `-fsanitize=address,undefined`.
- Tests run with `ASAN_OPTIONS=detect_leaks=0` because the leaks above would
  otherwise drown the more critical use-after-free / OOB / UB findings. Once
  the factory cleanup lands, flip leak detection back on.
- After every fix, re-run `ctest --test-dir build-asan` with leak detection on
  and confirm the count of remaining direct leaks went down.

## Performance
- Optimize the code to make it run faster.

## Statistics
- Write a gnuplot-able learning curve to a file.
- Calculate a P-value for the ROC curve.

## Other
- Template the Matrix library.
- Softmax in output layer of MLP.
