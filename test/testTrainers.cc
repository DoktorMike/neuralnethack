#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace MultiLayerPerceptron;
using namespace DataTools;

static double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static double xor_out[][1] = {{0}, {1}, {1}, {0}};

static void buildXorDataSet(CoreDataSet& core, DataSet& data) {
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(xor_in[i], xor_in[i] + 2);
		std::vector<double> out(xor_out[i], xor_out[i] + 1);
		core.addPattern(Pattern(std::to_string(i), in, out));
	}
	data.coreDataSet(core);
}

static double evaluateAccuracy(Mlp& mlp, DataSet& data) {
	int correct = 0;
	for (int i = 0; i < 4; ++i) {
		const auto& out = mlp.propagate(data.pattern(i).input());
		int predicted = (out[0] >= 0.5) ? 1 : 0;
		int actual = (int)xor_out[i][0];
		if (predicted == actual) correct++;
	}
	return (double)correct / 4.0;
}

static bool testGDSSE() {
	srand(42); srand48(42);
	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	SummedSquare error(mlp, data);
	GradientDescent trainer(mlp, data, error, 0.001, 4, 0.1, 0.99, 0.9);
	trainer.numEpochs(3000);
	std::ostringstream sink;
	trainer.train(sink);
	double acc = evaluateAccuracy(mlp, data);
	std::cout << "GD + SSE:     " << acc * 100 << "%" << std::endl;
	return acc == 1.0;
}

static bool testGDCE() {
	srand(42); srand48(42);
	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	CrossEntropy error(mlp, data);
	GradientDescent trainer(mlp, data, error, 0.001, 4, 0.1, 0.99, 0.9);
	trainer.numEpochs(3000);
	std::ostringstream sink;
	trainer.train(sink);
	double acc = evaluateAccuracy(mlp, data);
	std::cout << "GD + CE:      " << acc * 100 << "%" << std::endl;
	return acc == 1.0;
}

static bool testAdamSSE() {
	srand(42); srand48(42);
	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	SummedSquare error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(2000);
	std::ostringstream sink;
	trainer.train(sink);
	double acc = evaluateAccuracy(mlp, data);
	std::cout << "Adam + SSE:   " << acc * 100 << "%" << std::endl;
	return acc == 1.0;
}

static bool testAdamCE() {
	srand(42); srand48(42);
	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	CrossEntropy error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(2000);
	std::ostringstream sink;
	trainer.train(sink);
	double acc = evaluateAccuracy(mlp, data);
	std::cout << "Adam + CE:    " << acc * 100 << "%" << std::endl;
	return acc == 1.0;
}

static bool testQNSSE() {
	srand(42); srand48(42);
	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);
	SummedSquare error(mlp, data);
	QuasiNewton trainer(mlp, data, error, 0.001, 4);
	trainer.numEpochs(500);
	std::ostringstream sink;
	trainer.train(sink);
	double acc = evaluateAccuracy(mlp, data);
	std::cout << "QN + SSE:     " << acc * 100 << "%" << std::endl;
	return acc == 1.0;
}

int main() {
	bool pass = true;
	std::cout << "Testing optimizer/loss combinations on XOR:" << std::endl;
	if (!testGDSSE()) { std::cerr << "  FAIL" << std::endl; pass = false; }
	if (!testGDCE()) { std::cerr << "  FAIL" << std::endl; pass = false; }
	if (!testAdamSSE()) { std::cerr << "  FAIL" << std::endl; pass = false; }
	if (!testAdamCE()) { std::cerr << "  FAIL" << std::endl; pass = false; }
	if (!testQNSSE()) { std::cerr << "  FAIL" << std::endl; pass = false; }
	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
