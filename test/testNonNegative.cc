#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

// y = +1.0*x0 - 1.0*x1 + 0.5*x2: the true beta of x1 is negative, so a
// non-negativity constraint on all three inputs should clamp x1's weight
// to exactly 0 while x0 and x2 stay positive and close to truth.
DataSet buildData(uint n) {
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		std::vector<double> in = {nnh::rand::uniform(), nnh::rand::uniform(),
		                          nnh::rand::uniform()};
		std::vector<double> out = {1.0 * in[0] - 1.0 * in[1] + 0.5 * in[2] +
		                           0.01 * (2.0 * nnh::rand::uniform() - 1.0)};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

bool checkWeights(Mlp& mlp, const char* opt) {
	const auto& w = mlp.layer(0).weights(); // [1 x 4]: x0 x1 x2 bias
	const bool ok = w[0] > 0.5 && w[0] < 1.5 && w[1] == 0.0 && w[2] > 0.1 && w[2] < 0.9;
	if (!ok)
		std::cerr << "FAIL (" << opt << ": w = " << w[0] << " " << w[1] << " " << w[2] << ")"
		          << std::endl;
	else
		std::cout << "PASS (w0 " << w[0] << ", w1 " << w[1] << ", w2 " << w[2] << ")" << std::endl;
	return ok;
}

bool adamProjects() {
	std::cout << "adam projects constrained weights: ";
	nnh::rand::seed(5);
	DataSet ds = buildData(300);
	Mlp mlp({3, 1}, {"purelin"}, false);
	mlp.nonNegative(0, 0, 2); // constrain the three inputs, not the bias
	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, 32, 0.02);
	opt.numEpochs(300);
	std::ostringstream sink;
	opt.train(sink);
	return checkWeights(mlp, "adam");
}

bool qnProjects() {
	std::cout << "l-bfgs projects constrained weights: ";
	nnh::rand::seed(5);
	DataSet ds = buildData(300);
	Mlp mlp({3, 1}, {"purelin"}, false);
	mlp.nonNegative(0, 0, 2);
	SummedSquare loss(mlp, ds);
	QuasiNewton opt(mlp, ds, loss, 0.0, 300);
	opt.numEpochs(400);
	std::ostringstream sink;
	opt.train(sink);
	return checkWeights(mlp, "l-bfgs");
}

// Same fit without the constraint must recover the negative beta -- the
// constraint is optional, not baked in.
bool unconstrainedUnaffected() {
	std::cout << "unconstrained fit recovers negative beta: ";
	nnh::rand::seed(5);
	DataSet ds = buildData(300);
	Mlp mlp({3, 1}, {"purelin"}, false);
	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, 32, 0.02);
	opt.numEpochs(300);
	std::ostringstream sink;
	opt.train(sink);
	const auto& w = mlp.layer(0).weights();
	if (!(w[1] < -0.5)) {
		std::cerr << "FAIL (w1 = " << w[1] << ", expected ~ -1)" << std::endl;
		return false;
	}
	std::cout << "PASS (w1 " << w[1] << ")" << std::endl;
	return true;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== Non-Negative Weights Test Suite ===" << std::endl << std::endl;

	allPass &= adamProjects();
	allPass &= qnProjects();
	allPass &= unconstrainedUnaffected();

	std::cout << std::endl << (allPass ? "ALL PASS" : "SOME FAILED") << std::endl;
	return allPass ? 0 : 1;
}
