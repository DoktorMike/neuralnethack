// Pima benchmark for neuralnethack. Builds an 8-32-1 MLP trained with
// Adam + SummedSquare for `EPOCHS` epochs, batch `BATCH`. Reports
// wall-clock train time, mean per-sample inference latency on the test
// set, and test accuracy. Single-binary, links against the static lib.

#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include "pima_loader.hh"
#include "timing.hh"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

DataSet toDataSet(const bench::Pima& p) {
	auto core = std::make_shared<CoreDataSet>();
	for (std::size_t i = 0; i < p.X.size(); ++i) {
		std::vector<double> in = p.X[i];
		std::vector<double> out = {static_cast<double>(p.y[i])};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

} // namespace

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "test/pima-indians-diabetes";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 100;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 10;

	bench::Pima trn = bench::loadPima(root + "/pima.trn.tab");
	bench::Pima tst = bench::loadPima(root + "/pima.tst.tab");
	bench::zNormalise(trn, tst);

	DataSet trnSet = toDataSet(trn);
	DataSet tstSet = toDataSet(tst);

	std::vector<uint> arch = {8, 32, 1};
	std::vector<std::string> types = {"tansig", "logsig"};

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif
#ifdef USE_BLAS
	const char* blas = "openblas";
#else
	const char* blas = "none";
#endif

	for (int t = 0; t < trials; ++t) {
		const uint seed = 42 + static_cast<uint>(t);
		std::srand(seed);
		nnh::rand::seed(seed);
		Mlp mlp(arch, types, false);
		SummedSquare loss(mlp, trnSet);
		Adam opt(mlp, trnSet, loss, 0.0, batch, 0.01);
		opt.numEpochs(epochs);
		std::ostringstream sink;

		const auto t0 = bench::clk::now();
		opt.train(sink);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		int correct = 0;
		const int reps = 20;
		const auto i0 = bench::clk::now();
		for (int r = 0; r < reps; ++r)
			for (std::size_t i = 0; i < tst.X.size(); ++i) {
				const auto& y = mlp.propagate(tst.X[i]);
				if (r == 0 && (y[0] >= 0.5 ? 1 : 0) == tst.y[i]) ++correct;
			}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / (reps * tst.X.size());
		const double acc = double(correct) / tst.X.size();

		bench::emit("neuralnethack", "pima", "8-32-1", epochs, batch, threads, blas, t + 1,
		            train_s, infer_us, acc);
	}
	return 0;
}
