#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

// Tiny XOR DataSet -- just enough to drive forward + backward through
// each activation layer so the gradient/derivative paths are exercised
// (otherwise gcov reports ~6% on Linear/ELU/LeakyReLU layers).
DataSet buildXor() {
	auto core = std::make_shared<CoreDataSet>();
	double in_[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double out_[][1] = {{0}, {1}, {1}, {0}};
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(in_[i], in_[i] + 2);
		std::vector<double> out(out_[i], out_[i] + 1);
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

bool trainBriefly(const std::string& act) {
	srand(7);
	nnh::rand::seed(7);
	DataSet data = buildXor();
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {act, "logsig"};
	Mlp mlp(arch, types, false);
	SummedSquare loss(mlp, data);
	Adam opt(mlp, data, loss, 0.0, 4, 0.05);
	opt.numEpochs(50);
	std::ostringstream sink;
	opt.train(sink);
	for (uint i = 0; i < data.size(); ++i) {
		const auto& y = mlp.propagate(data.pattern(i).input());
		if (!std::isfinite(y[0])) return false;
	}
	return true;
}

} // namespace

int main() {
	const std::vector<std::string> activations = {"logsig", "tansig",    "purelin",
	                                              "relu",   "leakyrelu", "elu"};

	const std::vector<uint> arch = {2, 3, 1};
	const std::vector<double> input = {0.5, 0.5};
	bool pass = true;

	for (const auto& act : activations) {
		std::cout << "Testing activation: " << act << std::endl;

		// Set deterministic weights
		srand(42);
		nnh::rand::seed(42);

		std::vector<std::string> types = {act, "logsig"};
		Mlp mlp(arch, types, false);

		const std::vector<double>& output = mlp.propagate(input);

		// Check output is finite
		if (!std::isfinite(output[0])) {
			std::cerr << "  FAIL: output is not finite for " << act << " (got " << output[0] << ")"
			          << std::endl;
			pass = false;
			continue;
		}

		// Check output is in [0, 1] (sigmoid output layer)
		if (output[0] < 0.0 || output[0] > 1.0) {
			std::cerr << "  FAIL: output out of [0,1] for " << act << " (got " << output[0] << ")"
			          << std::endl;
			pass = false;
			continue;
		}

		std::cout << "  output = " << output[0] << " (OK)" << std::endl;

		// Test clone via copy constructor
		Mlp cloned(mlp);
		const std::vector<double>& cloneOutput = cloned.propagate(input);

		if (cloneOutput[0] != output[0]) {
			std::cerr << "  FAIL: clone output mismatch for " << act << " (original=" << output[0]
			          << ", clone=" << cloneOutput[0] << ")" << std::endl;
			pass = false;
			continue;
		}

		std::cout << "  clone output = " << cloneOutput[0] << " (match)" << std::endl;

		if (!trainBriefly(act)) {
			std::cerr << "  FAIL: training step produced non-finite output for " << act
			          << std::endl;
			pass = false;
			continue;
		}
		std::cout << "  trained 50 epochs (OK)" << std::endl;
	}

	if (pass) {
		std::cout << std::endl << "All activation tests PASSED." << std::endl;
	} else {
		std::cerr << std::endl << "Some activation tests FAILED." << std::endl;
	}

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
