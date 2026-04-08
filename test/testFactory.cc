#include "Factory.hh"
#include "Config.hh"
#include "mlp/Weights.hh"
#include "mlp/Mlp.hh"
#include "mlp/Trainer.hh"
#include "mlp/Error.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace NeuralNetHack;

#define CHECK(cond, msg)                                                                           \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::cerr << "FAIL: " << (msg) << std::endl;                                          \
			return false;                                                                          \
		}                                                                                          \
	} while (0)

// ---------------------------------------------------------------------------
// Part 1 - Weights
// ---------------------------------------------------------------------------

static bool testWeightsSize() {
	std::vector<uint> arch = {2, 3, 1};
	Weights w(arch);
	// Expected: (2+1)*3 + (3+1)*1 = 9 + 4 = 13
	CHECK(w.size() == 13, "Weights size should be 13, got " + std::to_string(w.size()));
	std::cout << "  PASS: Weights size is 13" << std::endl;
	return true;
}

static bool testWeightsUpdate() {
	std::vector<uint> arch = {2, 3, 1};
	Weights w(arch);
	double newVal = 3.14;
	w.update(0, newVal);
	CHECK(w.weights()[0] == newVal,
	      "Weights update: expected " + std::to_string(newVal) +
	          ", got " + std::to_string(w.weights()[0]));
	std::cout << "  PASS: Weights update works" << std::endl;
	return true;
}

static bool testWeightsKill() {
	std::vector<uint> arch = {2, 3, 1};
	Weights w(arch);
	w.update(5, 99.0);
	w.kill(5);
	CHECK(w.weights()[5] == 0.0,
	      "Weights kill: expected 0, got " + std::to_string(w.weights()[5]));
	std::cout << "  PASS: Weights kill works" << std::endl;
	return true;
}

static bool testWeightsCopyConstructor() {
	std::vector<uint> arch = {2, 3, 1};
	Weights w1(arch);
	w1.update(0, 1.23);
	w1.update(12, 4.56);

	Weights w2(w1);
	CHECK(w2.size() == w1.size(),
	      "Copy ctor: size mismatch");
	CHECK(w2.weights()[0] == 1.23,
	      "Copy ctor: weight[0] mismatch");
	CHECK(w2.weights()[12] == 4.56,
	      "Copy ctor: weight[12] mismatch");

	// Verify independence: mutating the copy does not affect the original
	w2.update(0, 9.99);
	CHECK(w1.weights()[0] == 1.23,
	      "Copy ctor: modifying copy affected original");
	std::cout << "  PASS: Weights copy constructor works" << std::endl;
	return true;
}

static bool testWeightsAssignment() {
	std::vector<uint> arch = {2, 3, 1};
	Weights w1(arch);
	w1.update(0, 7.77);
	w1.update(6, 8.88);

	std::vector<uint> arch2 = {1, 1};
	Weights w2(arch2);
	w2 = w1;

	CHECK(w2.size() == w1.size(),
	      "Assignment: size mismatch");
	CHECK(w2.weights()[0] == 7.77,
	      "Assignment: weight[0] mismatch");
	CHECK(w2.weights()[6] == 8.88,
	      "Assignment: weight[6] mismatch");

	// Verify independence
	w2.update(0, 0.01);
	CHECK(w1.weights()[0] == 7.77,
	      "Assignment: modifying assignee affected original");
	std::cout << "  PASS: Weights assignment operator works" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Part 2 - Factory
// ---------------------------------------------------------------------------

static DataSet buildDummyDataSet() {
	CoreDataSet core;
	std::vector<double> in = {0.5, 0.8};
	std::vector<double> out = {1.0};
	core.addPattern(Pattern("p0", in, out));
	DataSet data;
	data.coreDataSet(core);
	return data;
}

static bool testFactoryCreateMlp() {
	Config config;
	config.architecture({2, 3, 1});
	config.actFcn({"relu", "logsig"});
	config.errFcn("sumsqr");
	config.minMethod("gd");
	config.maxEpochs(100);
	config.batchSize(1);
	config.learningRate(0.1);
	config.decLearningRate(0.99);
	config.momentum(0.9);
	config.weightElimOn(false);

	Mlp* mlp = Factory::createMlp(config);
	CHECK(mlp != 0, "createMlp returned null");

	// Verify propagation works with a 2-input pattern
	std::vector<double> input = {0.5, 0.8};
	const std::vector<double>& output = mlp->propagate(input);
	CHECK(output.size() == 1, "Mlp output size should be 1, got " + std::to_string(output.size()));
	std::cout << "  PASS: Factory::createMlp works" << std::endl;

	delete mlp;
	return true;
}

static bool testFactoryCreateTrainerGD() {
	Config config;
	config.architecture({2, 3, 1});
	config.actFcn({"relu", "logsig"});
	config.errFcn("sumsqr");
	config.minMethod("gd");
	config.maxEpochs(100);
	config.batchSize(1);
	config.learningRate(0.1);
	config.decLearningRate(0.99);
	config.momentum(0.9);
	config.weightElimOn(false);

	DataSet data = buildDummyDataSet();
	Trainer* trainer = Factory::createTrainer(config, data);
	CHECK(trainer != 0, "createTrainer(gd) returned null");
	std::cout << "  PASS: Factory::createTrainer (gd) works" << std::endl;

	delete trainer->error();
	delete trainer;
	return true;
}

static bool testFactoryCreateTrainerAdam() {
	Config config;
	config.architecture({2, 3, 1});
	config.actFcn({"relu", "logsig"});
	config.errFcn("sumsqr");
	config.minMethod("adam");
	config.maxEpochs(100);
	config.batchSize(1);
	config.adamLearningRate(0.001);
	config.adamBeta1(0.9);
	config.adamBeta2(0.999);
	config.adamEpsilon(1e-8);
	config.adamWeightDecay(0.0);
	config.weightElimOn(false);

	DataSet data = buildDummyDataSet();
	Trainer* trainer = Factory::createTrainer(config, data);
	CHECK(trainer != 0, "createTrainer(adam) returned null");
	std::cout << "  PASS: Factory::createTrainer (adam) works" << std::endl;

	delete trainer->error();
	delete trainer;
	return true;
}

static bool testFactoryCreateTrainerQN() {
	Config config;
	config.architecture({2, 3, 1});
	config.actFcn({"relu", "logsig"});
	config.errFcn("sumsqr");
	config.minMethod("qn");
	config.maxEpochs(100);
	config.batchSize(1);
	config.weightElimOn(false);

	DataSet data = buildDummyDataSet();
	Trainer* trainer = Factory::createTrainer(config, data);
	CHECK(trainer != 0, "createTrainer(qn) returned null");
	std::cout << "  PASS: Factory::createTrainer (qn) works" << std::endl;

	delete trainer->error();
	delete trainer;
	return true;
}

static bool testFactoryCreateErrorKullback() {
	Config config;
	config.architecture({2, 3, 1});
	config.actFcn({"relu", "logsig"});
	config.errFcn("kullback");
	config.weightElimOn(false);

	DataSet data = buildDummyDataSet();
	Error* error = Factory::createError(config, data);
	CHECK(error != 0, "createError(kullback) returned null");
	std::cout << "  PASS: Factory::createError (kullback) works" << std::endl;

	delete error;
	return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
	srand48(42);

	bool allPassed = true;

	std::cout << "Part 1 - Weights tests:" << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	if (!testWeightsSize()) allPassed = false;
	if (!testWeightsUpdate()) allPassed = false;
	if (!testWeightsKill()) allPassed = false;
	if (!testWeightsCopyConstructor()) allPassed = false;
	if (!testWeightsAssignment()) allPassed = false;

	std::cout << std::endl;
	std::cout << "Part 2 - Factory tests:" << std::endl;
	std::cout << std::string(50, '-') << std::endl;

	if (!testFactoryCreateMlp()) allPassed = false;
	if (!testFactoryCreateTrainerGD()) allPassed = false;
	if (!testFactoryCreateTrainerAdam()) allPassed = false;
	if (!testFactoryCreateTrainerQN()) allPassed = false;
	if (!testFactoryCreateErrorKullback()) allPassed = false;

	std::cout << std::endl;
	std::cout << std::string(50, '-') << std::endl;
	if (allPassed) {
		std::cout << "All tests passed." << std::endl;
		return EXIT_SUCCESS;
	} else {
		std::cerr << "Some tests FAILED." << std::endl;
		return EXIT_FAILURE;
	}
}
