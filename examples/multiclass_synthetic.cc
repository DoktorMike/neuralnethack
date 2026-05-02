// Softmax MLP on a synthetic 3-class planar partitioning task.
//
// Sanity demo for the softmax + cross-entropy path: no external data, just
// a deterministic 3-region split of the unit square. Trains a small MLP
// and reports train/test accuracy.
//
// Build:   cmake --build build --target multiclass_synthetic
// Run:     ./build/multiclass_synthetic

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "Random.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/Mlp.hh"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {
constexpr uint K = 3;

uint targetClass(double a, double b) {
	if (a + b > 0.5) return 0;
	if (a - b > 0.0) return 1;
	return 2;
}

DataSet makeData(uint n, uint seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> uni(-1.0, 1.0);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		double a = uni(rng);
		double b = uni(rng);
		std::vector<double> in = {a, b};
		std::vector<double> tgt(K, 0.0);
		tgt[targetClass(a, b)] = 1.0;
		core->addPattern(Pattern(std::to_string(i), in, tgt));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

double accuracy(Mlp& mlp, DataSet& ds) {
	uint correct = 0;
	for (uint i = 0; i < ds.size(); ++i) {
		const auto& p = mlp.propagate(ds.pattern(i).input());
		const auto& t = ds.pattern(i).output();
		uint argP = 0, argT = 0;
		for (uint j = 1; j < K; ++j) {
			if (p[j] > p[argP]) argP = j;
			if (t[j] > t[argT]) argT = j;
		}
		if (argP == argT) ++correct;
	}
	return static_cast<double>(correct) / ds.size();
}
} // namespace

int main() {
	const uint seed = 23;
	srand(seed);
	nnh::rand::seed(seed);

	DataSet trn = makeData(400, seed);
	DataSet tst = makeData(200, seed + 1);

	std::vector<uint> arch = {2, 16, K};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, /*softmax=*/true);

	CrossEntropy loss(mlp, trn);
	Adam opt(mlp, trn, loss, /*te=*/0.0, /*bs=*/32, /*lr=*/0.05);
	opt.numEpochs(2000);
	std::ostringstream sink;
	opt.train(sink);

	std::cout << "synthetic 3-class:\n"
	          << "  train acc = " << accuracy(mlp, trn) << "\n"
	          << "  test acc  = " << accuracy(mlp, tst) << "\n";
	return 0;
}
