#include "Ensemble.hh"
#include "mlp/Mlp.hh"
#include "mlp/Serialization.hh"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <memory>

using namespace MultiLayerPerceptron;
using namespace NeuralNetHack;

#define CHECK(cond, msg)                                                       \
	do {                                                                       \
		if (!(cond)) {                                                         \
			std::cerr << "FAIL: " << (msg) << std::endl;                       \
			pass = false;                                                       \
		}                                                                      \
	} while (0)

int main() {
	srand(42);
	srand48(42);

	bool pass = true;

	// -- 1. Create two 2-3-1 MLPs with "relu" hidden, "logsig" output --
	std::vector<uint> arch = {2, 3, 1};
	std::vector<std::string> types = {"relu", "logsig"};

	Mlp mlp1(arch, types, false);
	Mlp mlp2(arch, types, false);
	mlp1.regenerateWeights();
	mlp2.regenerateWeights();

	// -- 2. Build an Ensemble and add both MLPs via addMlp(mlp_ref) --
	Ensemble ensemble;
	ensemble.addMlp(mlp1);
	ensemble.addMlp(mlp2);

	// -- 3. Verify ensemble.size() == 2 --
	CHECK(ensemble.size() == 2, "ensemble size should be 2 after adding two MLPs");

	// -- 4. Propagate input {0.5, 0.5}, verify output is finite and in [0,1] --
	std::vector<double> input = {0.5, 0.5};
	std::vector<double> output = ensemble.propagate(input);

	CHECK(output.size() == 1, "output should have 1 element for a 2-3-1 network");
	CHECK(std::isfinite(output[0]), "output should be finite");
	CHECK(output[0] >= 0.0 && output[0] <= 1.0,
	      "output should be in [0,1] for logsig output");

	std::cout << "Ensemble propagate output: " << output[0] << std::endl;

	// -- 5. Test delMlp: delete one MLP, verify size == 1, propagate still works --
	ensemble.delMlp(0);
	CHECK(ensemble.size() == 1, "ensemble size should be 1 after deleting one MLP");

	std::vector<double> output_after_del = ensemble.propagate(input);
	CHECK(output_after_del.size() == 1, "output should still have 1 element after delMlp");
	CHECK(std::isfinite(output_after_del[0]),
	      "output should be finite after delMlp");
	CHECK(output_after_del[0] >= 0.0 && output_after_del[0] <= 1.0,
	      "output should be in [0,1] after delMlp");

	std::cout << "After delMlp propagate output: " << output_after_del[0] << std::endl;

	// -- 6. Test addMlp with unique_ptr --
	auto ptr = std::make_unique<Mlp>(arch, types, false);
	ptr->regenerateWeights();
	ensemble.addMlp(std::move(ptr), 0.5);

	CHECK(ensemble.size() == 2,
	      "ensemble size should be 2 after adding via unique_ptr");

	std::vector<double> output_ptr = ensemble.propagate(input);
	CHECK(output_ptr.size() == 1, "output should have 1 element after unique_ptr add");
	CHECK(std::isfinite(output_ptr[0]),
	      "output should be finite after unique_ptr add");

	std::cout << "After addMlp(unique_ptr) propagate output: " << output_ptr[0]
	          << std::endl;

	// -- 7. Test copy constructor --
	Ensemble copy(ensemble);
	CHECK(copy.size() == ensemble.size(),
	      "copy should have the same size as original");

	std::vector<double> out_orig = ensemble.propagate(input);
	std::vector<double> out_copy = copy.propagate(input);

	CHECK(out_orig.size() == out_copy.size(),
	      "copy output should have the same size as original output");
	for (size_t i = 0; i < out_orig.size(); ++i) {
		CHECK(out_orig[i] == out_copy[i],
		      "copy output should match original output exactly");
	}

	std::cout << "Copy constructor output matches: " << out_copy[0] << std::endl;

	// -- 8. Test save/load roundtrip --
	const std::string path = "test_ensemble.nne";
	saveEnsembleBinary(ensemble, path);

	auto loaded = loadEnsembleBinary(path);
	CHECK(loaded->size() == ensemble.size(),
	      "loaded ensemble should have the same size as original");

	std::vector<double> out_loaded = loaded->propagate(input);
	CHECK(out_loaded.size() == out_orig.size(),
	      "loaded output should have the same size as original output");
	for (size_t i = 0; i < out_orig.size(); ++i) {
		CHECK(out_orig[i] == out_loaded[i],
		      "loaded output should match original output exactly");
	}

	std::cout << "Save/load roundtrip output matches: " << out_loaded[0]
	          << std::endl;

	// Clean up the file
	std::remove(path.c_str());

	if (pass) {
		std::cout << "All ensemble tests PASSED." << std::endl;
	} else {
		std::cerr << "Some ensemble tests FAILED." << std::endl;
	}

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
