# TODO

## Open

### DataManager::split returns raw owning pointers
`DataTools::DataManager::split(DataSet&)` returns `pair<DataSet, DataSet>*`,
and `DataManager::split(DataSet&, uint k)` returns `vector<DataSet>*`. Callers
have to remember to `delete`. Convert both to value or `unique_ptr` returns.
Touches the samplers (`BootstrapSampler`, `CrossSplitSampler`, `HoldOutSampler`,
`DummySampler`) and any test that drains a sampler manually.

## Performance
- Optimize the code to make it run faster.

## Statistics
- Write a gnuplot-able learning curve to a file.
- Calculate a P-value for the ROC curve.

## Other
- Template the Matrix library.
- Softmax in output layer of MLP.
