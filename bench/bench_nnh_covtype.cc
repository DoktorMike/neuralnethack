// Covertype benchmark for neuralnethack. 54-128-7 softmax MLP trained
// with Adam + CrossEntropy. Same trial loop and CSV format as the Pima
// harness so summarise.py can fold the rows together.

#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/Mlp.hh"

#include "covtype_loader.hh"
#include "timing.hh"

#include <cstdlib>
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

constexpr uint K = 7;

DataSet toDataSet(const bench::CovType& p) {
	auto core = std::make_shared<CoreDataSet>();
	for (std::size_t i = 0; i < p.X.size(); ++i) {
		std::vector<double> in = p.X[i];
		std::vector<double> out(K, 0.0);
		out[p.y[i]] = 1.0;
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

uint argmax(const std::vector<double>& v) {
	uint a = 0;
	for (uint i = 1; i < v.size(); ++i)
		if (v[i] > v[a]) a = i;
	return a;
}

} // namespace

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "datasets/covtype";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 5;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 3;

	bench::CovType trn = bench::loadCovType(root + "/covtype.trn.csv");
	bench::CovType tst = bench::loadCovType(root + "/covtype.tst.csv");
	bench::zNormaliseContinuous(trn, tst);

	DataSet trnSet = toDataSet(trn);

	std::vector<uint> arch = {54, 128, K};
	std::vector<std::string> types = {"tansig", "purelin"};

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
		Mlp mlp(arch, types, /*softmax=*/true);
		CrossEntropy loss(mlp, trnSet);
		Adam opt(mlp, trnSet, loss, 0.0, batch, 0.01);
		opt.numEpochs(epochs);
		std::ostringstream sink;

		const auto t0 = bench::clk::now();
		opt.train(sink);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		std::size_t correct = 0;
		const auto i0 = bench::clk::now();
		for (std::size_t i = 0; i < tst.X.size(); ++i) {
			const auto& y = mlp.propagate(tst.X[i]);
			if (static_cast<int>(argmax(y)) == tst.y[i]) ++correct;
		}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / tst.X.size();
		const double acc = static_cast<double>(correct) / tst.X.size();

		bench::emit("neuralnethack", "covtype", "54-128-7", epochs, batch, threads, blas, t + 1,
		            train_s, infer_us, acc);
	}
	return 0;
}
