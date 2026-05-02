# TODO

## Open follow-ups

### Time series evaluation (phase 2 of the metrics work)
MASE, Theil's U, Diebold-Mariano, Ljung-Box on residuals. Belongs in its own
`TimeSeriesMetrics` class because most of these need extra context (training
set's seasonal-naive error for MASE; two competing forecasts for DM; etc).
Open question: do we actually want the library to grow into a time-series
direction, or is this a "use libtorch" signal.

### clangd: unused-include sweep
Across the session clangd has flagged a steady stream of unused includes
(`MatrixTools.hh`, `algorithm`, `iterator`, etc) in test files and a few lib
files. None are bugs but they slow rebuilds. Run a single mechanical
include-cleanup pass with clang-tidy `--checks=misc-unused-includes`.

## Performance
- Optimize the code to make it run faster.

## Statistics
- Write a gnuplot-able learning curve to a file.
- Calculate a P-value for the ROC curve.

## Other
- Template the Matrix library.
- Softmax in output layer of MLP.
