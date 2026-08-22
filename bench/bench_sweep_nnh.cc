// Width sweep: synthetic regression, arch IN-H-1, Adam + SummedSquare.
// Prints: H,train_s_per_epoch,infer_us
#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
	const uint IN = (argc > 1) ? std::atoi(argv[1]) : 64;
	const uint H = (argc > 2) ? std::atoi(argv[2]) : 128;
	const uint N = (argc > 3) ? std::atoi(argv[3]) : 4096;
	const int epochs = (argc > 4) ? std::atoi(argv[4]) : 5;
	const int batch = (argc > 5) ? std::atoi(argv[5]) : 64;

	nnh::rand::seed(42);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < N; ++i) {
		std::vector<double> x(IN);
		double s = 0;
		for (auto& v : x) {
			v = 2.0 * nnh::rand::uniform() - 1.0;
			s += v;
		}
		std::vector<double> t = {s > 0 ? 1.0 : 0.0};
		core->addPattern(Pattern(std::to_string(i), x, t));
	}
	DataSet ds;
	ds.coreDataSet(core);

	std::vector<uint> arch = {IN, H, 1};
	std::vector<std::string> types = {"tansig", "logsig"};
	Mlp mlp(arch, types, false);
	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, batch, 0.01);
	opt.numEpochs(epochs);
	std::ostringstream sink;

	auto t0 = clk::now();
	opt.train(sink);
	auto t1 = clk::now();
	double per_epoch = std::chrono::duration<double>(t1 - t0).count() / epochs;

	// batch-1 inference
	std::vector<double> x0 = core->pattern(0).input();
	volatile double acc = 0;
	const int reps = 2000;
	auto i0 = clk::now();
	for (int r = 0; r < reps; ++r)
		acc += mlp.propagate(x0)[0];
	auto i1 = clk::now();
	double infer_us = 1e6 * std::chrono::duration<double>(i1 - i0).count() / reps;

	std::printf("nnh,%u,%u,%.5f,%.2f\n", IN, H, per_epoch, infer_us);
	return 0;
}
