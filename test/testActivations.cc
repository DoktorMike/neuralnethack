#include "mlp/Mlp.hh"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace MultiLayerPerceptron;

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
		srand48(42);

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
	}

	if (pass) {
		std::cout << std::endl << "All activation tests PASSED." << std::endl;
	} else {
		std::cerr << std::endl << "Some activation tests FAILED." << std::endl;
	}

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
