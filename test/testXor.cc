#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/Serialization.hh"
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

int main() {
	srand(42);
	srand48(42);

	// -- Build the XOR dataset --
	CoreDataSet core;
	double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double xor_out[][1] = {{0}, {1}, {1}, {0}};
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(xor_in[i], xor_in[i] + 2);
		std::vector<double> out(xor_out[i], xor_out[i] + 1);
		core.addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);

	// -- Create a 2-4-1 network (ReLU hidden, sigmoid output) --
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);

	// -- Train with Adam for 2000 epochs --
	SummedSquare error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(2000);
	trainer.train(std::cout);

	// -- Evaluate --
	bool pass = true;
	int tp = 0, tn = 0, fp = 0, fn = 0;
	double threshold = 0.5;

	std::cout << std::endl;
	for (int i = 0; i < 4; ++i) {
		const auto& out = mlp.propagate(data.pattern(i).input());
		double expected = xor_out[i][0];
		double got = out[0];
		int predicted = (got >= threshold) ? 1 : 0;
		int actual = (int)expected;

		std::cout << xor_in[i][0] << " XOR " << xor_in[i][1] << " = " << got << " (predicted "
		          << predicted << ", expected " << actual << ")" << std::endl;

		if (actual == 1 && predicted == 1)
			tp++;
		else if (actual == 0 && predicted == 0)
			tn++;
		else if (actual == 0 && predicted == 1)
			fp++;
		else
			fn++;

		if (fabs(got - expected) > 0.2) {
			std::cerr << "  FAIL: error too large" << std::endl;
			pass = false;
		}
	}

	double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
	double sensitivity = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
	double specificity = (tn + fp > 0) ? (double)tn / (tn + fp) : 0.0;
	double balanced_accuracy = (sensitivity + specificity) / 2.0;

	std::cout << std::endl
	          << "Accuracy:          " << accuracy << std::endl
	          << "Balanced Accuracy: " << balanced_accuracy << std::endl
	          << "Sensitivity:       " << sensitivity << std::endl
	          << "Specificity:       " << specificity << std::endl;

	// -- Test save/load roundtrip --
	saveMlpBinary(mlp, "xor_test.nnh");
	auto loaded = loadMlpBinary("xor_test.nnh");

	for (int i = 0; i < 4; ++i) {
		const auto& orig = mlp.propagate(data.pattern(i).input());
		const auto& reload = loaded->propagate(data.pattern(i).input());
		if (orig[0] != reload[0]) {
			std::cerr << "FAIL: serialization roundtrip mismatch at pattern " << i << std::endl;
			pass = false;
		}
	}

	// Clean up
	std::remove("xor_test.nnh");

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
