#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace MultiLayerPerceptron;
using namespace DataTools;

// Build XOR dataset
DataSet makeXorData(CoreDataSet& core) {
	double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double xor_out[][1] = {{0}, {1}, {1}, {0}};
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(xor_in[i], xor_in[i] + 2);
		std::vector<double> out(xor_out[i], xor_out[i] + 1);
		core.addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);
	return data;
}

// Evaluate: returns true if all 4 XOR outputs are within tolerance
bool evaluate(Mlp& mlp, DataSet& data, const std::string& label) {
	double xor_expected[] = {0, 1, 1, 0};
	bool pass = true;
	int tp = 0, tn = 0, fp = 0, fn = 0;

	std::cout << "\n--- " << label << " ---" << std::endl;
	for (int i = 0; i < 4; ++i) {
		const auto& out = mlp.propagate(data.pattern(i).input());
		double expected = xor_expected[i];
		double got = out[0];
		int predicted = (got >= 0.5) ? 1 : 0;
		int actual = (int)expected;

		std::cout << "  pattern " << i << ": got=" << got << " expected=" << expected << std::endl;

		if (actual == 1 && predicted == 1) tp++;
		else if (actual == 0 && predicted == 0) tn++;
		else if (actual == 0 && predicted == 1) fp++;
		else fn++;

		if (predicted != actual) pass = false;
	}

	double accuracy = (double)(tp + tn) / 4.0;
	double sensitivity = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
	double specificity = (tn + fp > 0) ? (double)tn / (tn + fp) : 0.0;
	std::cout << "  Accuracy: " << accuracy << "  Sens: " << sensitivity
	          << "  Spec: " << specificity << std::endl;

	return pass;
}

// Train and evaluate with a given NormType
bool testWithNorm(NormType nt, const std::string& label) {
	srand(42);
	srand48(42);

	CoreDataSet core;
	DataSet data = makeXorData(core);

	std::vector<uint> arch = {2, 8, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	mlp.normType(nt);

	SummedSquare error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(3000);
	trainer.train(std::cout);

	return evaluate(mlp, data, label);
}

int main() {
	bool pass = true;

	if (!testWithNorm(NormType::BatchNorm, "BatchNorm")) {
		std::cerr << "FAIL: BatchNorm did not learn XOR" << std::endl;
		pass = false;
	}

	if (!testWithNorm(NormType::LayerNorm, "LayerNorm")) {
		std::cerr << "FAIL: LayerNorm did not learn XOR" << std::endl;
		pass = false;
	}

	if (!testWithNorm(NormType::None, "No normalization (baseline)")) {
		std::cerr << "FAIL: Baseline did not learn XOR" << std::endl;
		pass = false;
	}

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
