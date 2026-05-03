// Tests for validation-loss early stopping.
//
// Synthetic noisy 1D regression with a deliberately-overprovisioned MLP:
// without early stopping the model overfits, so val loss eventually rises.
// Tests verify that:
//   1. earlyStopping(0) is a no-op (full training runs).
//   2. earlyStopping(p) with a val set triggers a stop and sets the
//      earlyStopped() flag.
//   3. Weights are restored to the best-val snapshot: final val loss
//      equals the per-epoch minimum recorded in the learning-curve file.

#include "Config.hh"
#include "Factory.hh"
#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"
#include "parser/Parser.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

double standardNormal() {
	double u1 = nnh::rand::uniform();
	double u2 = nnh::rand::uniform();
	if (u1 < 1e-12) u1 = 1e-12;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

DataSet buildNoisyRegression(uint n, double sigma, uint seed) {
	nnh::rand::seed(seed);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		double x = -3.0 + 6.0 * nnh::rand::uniform();
		double y = std::sin(x) + sigma * standardNormal();
		std::vector<double> in = {x};
		std::vector<double> out = {y};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

// Parse "# epoch trainErr [valErr]" learning-curve file. Returns the
// minimum valErr observed (any column index 2). Returns +inf on failure.
double minValFromCurve(const std::string& path) {
	std::ifstream in(path);
	if (!in) return std::numeric_limits<double>::infinity();
	double best = std::numeric_limits<double>::infinity();
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream iss(line);
		double epoch, trainErr, valErr;
		if (iss >> epoch >> trainErr >> valErr) {
			if (valErr < best) best = valErr;
		}
	}
	return best;
}

// Match SummedSquare::outputError exactly: sum_sq / N (no 0.5 factor).
double valLoss(Mlp& mlp, DataSet& val) {
	double sum = 0;
	for (uint i = 0; i < val.size(); ++i) {
		auto& pat = val.pattern(i);
		const auto& y = mlp.propagate(pat.input());
		const auto& t = pat.output();
		double diff = y[0] - t[0];
		sum += diff * diff;
	}
	return sum / val.size();
}

bool testNoEarlyStop() {
	std::cout << "Test: patience=0 disables early stopping ... ";
	std::srand(11);
	nnh::rand::seed(11);
	DataSet trn = buildNoisyRegression(40, 0.4, 21);
	DataSet val = buildNoisyRegression(40, 0.4, 22);

	std::vector<uint> arch = {1, 32, 1};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, false);
	SummedSquare loss(mlp, trn);
	Adam opt(mlp, trn, loss, 0.0, 16, 0.01);
	opt.numEpochs(200);
	opt.validationData(&val);
	// no earlyStopping() call → patience=0 by default

	std::ostringstream sink;
	opt.train(sink);
	if (opt.earlyStopped()) {
		std::cerr << "FAIL (earlyStopped() should be false)\n";
		return false;
	}
	std::cout << "PASS\n";
	return true;
}

bool testEarlyStopTriggers() {
	std::cout << "Test: patience triggers on overfitting model ... ";
	std::srand(7);
	nnh::rand::seed(7);
	DataSet trn = buildNoisyRegression(20, 0.5, 31);
	DataSet val = buildNoisyRegression(40, 0.5, 32);

	std::vector<uint> arch = {1, 64, 64, 1};
	std::vector<std::string> types = {"tansig", "tansig", "purelin"};
	Mlp mlp(arch, types, false);
	SummedSquare lossFn(mlp, trn);
	Adam opt(mlp, trn, lossFn, 0.0, 8, 0.02);
	opt.numEpochs(5000);
	opt.validationData(&val);
	opt.earlyStopping(/*patience=*/15, /*minDelta=*/1e-5);

	std::ostringstream sink;
	opt.train(sink);
	if (!opt.earlyStopped()) {
		std::cerr << "FAIL (earlyStopped() should be true on this overfit case)\n";
		return false;
	}
	std::cout << "PASS\n";
	return true;
}

bool testBestWeightsRestored() {
	std::cout << "Test: weights restored to best-val snapshot ... ";
	std::srand(91);
	nnh::rand::seed(91);
	DataSet trn = buildNoisyRegression(20, 0.5, 41);
	DataSet val = buildNoisyRegression(40, 0.5, 42);

	std::vector<uint> arch = {1, 64, 64, 1};
	std::vector<std::string> types = {"tansig", "tansig", "purelin"};
	Mlp mlp(arch, types, false);
	SummedSquare lossFn(mlp, trn);
	Adam opt(mlp, trn, lossFn, 0.0, 8, 0.02);
	opt.numEpochs(5000);
	opt.validationData(&val);
	opt.earlyStopping(15, 1e-5);
	const std::string curvePath = "_es_curve.dat";
	opt.learningCurveFile(curvePath);

	std::ostringstream sink;
	opt.train(sink);

	const double bestValSeen = minValFromCurve(curvePath);
	const double finalVal = valLoss(mlp, val);
	std::remove(curvePath.c_str());

	// After restoration, the model's val loss should match the best
	// observed in the learning curve up to numerical noise (the loss
	// function and the learning-curve writer use the same outputError
	// path so they should agree closely).
	const double tol = 1e-6 * std::max(1.0, bestValSeen);
	if (std::fabs(finalVal - bestValSeen) > tol) {
		std::cerr << "FAIL (final val=" << finalVal << ", best seen=" << bestValSeen
		          << ", diff=" << std::fabs(finalVal - bestValSeen) << ")\n";
		return false;
	}
	std::cout << "PASS (val=" << finalVal << ", best seen=" << bestValSeen << ")\n";
	return true;
}

bool testConfigWiring() {
	std::cout << "Test: TOML training.early_stopping plumbs into Config + trainer ... ";
	const char* toml =
	    "suffix = \"x\"\n"
	    "seed = 1\n"
	    "normalization = \"no\"\n"
	    "problem_type = \"regr\"\n"
	    "[data.train]\n"
	    "file = \"x\"\nid_col = 0\nin_cols = \"1\"\nout_cols = \"2\"\nrow_range = \"0\"\n"
	    "[data.test]\n"
	    "file = \"x\"\nid_col = 0\nin_cols = \"1\"\nout_cols = \"2\"\nrow_range = \"0\"\n"
	    "[network]\n"
	    "size = [1, 4, 1]\n"
	    "activations = [\"tansig\", \"purelin\"]\n"
	    "error_fcn = \"sumsqr\"\n"
	    "[training]\n"
	    "method = \"adam\"\n"
	    "max_epochs = 10\n"
	    "[training.gd]\nbatch_size = 4\nlearning_rate = 0.1\nlr_decay = 0.99\nmomentum = 0.0\n"
	    "[training.early_stopping]\n"
	    "patience = 7\n"
	    "min_delta = 0.0025\n"
	    "[ensemble]\nmethod = \"none\"\nruns = 1\nparts = 1\nsplit = \"ser\"\nvary_weights = "
	    "false\n"
	    "[model_selection]\nmethod = \"none\"\nruns = 1\nparts = 1\nsplit = \"ser\"\nfraction = "
	    "0.5\n"
	    "[output]\nsave_session = false\nsave_output_list = false\n";

	std::istringstream in(toml);
	NeuralNetHack::Config cfg;
	NeuralNetHack::Parser::readConfigurationFile(in, cfg);

	if (cfg.earlyStopPatience() != 7) {
		std::cerr << "FAIL (patience=" << cfg.earlyStopPatience() << ", expected 7)\n";
		return false;
	}
	if (std::fabs(cfg.earlyStopMinDelta() - 0.0025) > 1e-12) {
		std::cerr << "FAIL (min_delta=" << cfg.earlyStopMinDelta() << ", expected 0.0025)\n";
		return false;
	}

	// Tiny dummy dataset just to satisfy createTrainer's signature.
	auto core = std::make_shared<CoreDataSet>();
	std::vector<double> ix = {0.0};
	std::vector<double> iy = {0.0};
	core->addPattern(Pattern("0", ix, iy));
	DataSet ds;
	ds.coreDataSet(core);
	auto trainer = NeuralNetHack::Factory::createTrainer(cfg, ds);

	if (trainer->earlyStoppingPatience() != 7) {
		std::cerr << "FAIL (Factory did not wire patience: trainer reports "
		          << trainer->earlyStoppingPatience() << ")\n";
		return false;
	}
	if (std::fabs(trainer->earlyStoppingMinDelta() - 0.0025) > 1e-12) {
		std::cerr << "FAIL (trainer min_delta=" << trainer->earlyStoppingMinDelta() << ")\n";
		return false;
	}
	std::cout << "PASS\n";
	return true;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== Early Stopping Test Suite ===\n\n";
	allPass &= testNoEarlyStop();
	allPass &= testEarlyStopTriggers();
	allPass &= testBestWeightsRestored();
	allPass &= testConfigWiring();
	std::cout << "\n";
	if (allPass) {
		std::cout << "All tests PASSED.\n";
		return EXIT_SUCCESS;
	}
	std::cout << "Some tests FAILED.\n";
	return EXIT_FAILURE;
}
