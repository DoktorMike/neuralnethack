#include "Random.hh"
#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

// Imbalanced binary problem: many negatives near (0,0), few positives near
// (1,1). Returns a DataSet view over a freshly built CoreDataSet.
static DataSet buildImbalanced(uint nNeg, uint nPos) {
	auto core = std::make_shared<CoreDataSet>();
	uint id = 0;
	for (uint i = 0; i < nNeg; ++i) {
		double x = 0.05 * (i % 5);
		std::vector<double> in = {x, x};
		std::vector<double> out = {0.0};
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	for (uint i = 0; i < nPos; ++i) {
		double x = 1.0 - 0.05 * (i % 5);
		std::vector<double> in = {x, x};
		std::vector<double> out = {1.0};
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	DataSet data;
	data.coreDataSet(core);
	return data;
}

// Overlapping variant: in addition to a clean negative cluster, place nNoise
// NEGATIVE patterns on top of the positive cluster at (1,1). The positive
// region is now ambiguous, so an unweighted loss (dominated by the negatives
// there) underpredicts the minority unless its class is upweighted.
static DataSet buildOverlap(uint nNeg, uint nPos, uint nNoise) {
	auto core = std::make_shared<CoreDataSet>();
	uint id = 0;
	for (uint i = 0; i < nNeg; ++i) {
		double x = 0.05 * (i % 5);
		std::vector<double> in = {x, x}, out = {0.0};
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	for (uint i = 0; i < nPos; ++i) {
		std::vector<double> in = {1.0, 1.0}, out = {1.0};
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	for (uint i = 0; i < nNoise; ++i) {
		std::vector<double> in = {1.0, 1.0}, out = {0.0};
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	DataSet data;
	data.coreDataSet(core);
	return data;
}

static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
	double m = 0;
	for (uint i = 0; i < a.size(); ++i)
		m = std::max(m, std::fabs(a[i] - b[i]));
	return m;
}

// ---------------------------------------------------------------------------
// Test 1: uniform class weights {1,1} == no weights (exact backward compat)
// ---------------------------------------------------------------------------
static bool testUniformWeightsIdentity() {
	std::cout << "Test: classWeights({1,1}) == unweighted ... ";
	nnh::rand::seed(11);
	DataSet data = buildImbalanced(40, 8);
	Mlp mlp({2, 4, 1}, {"tansig", "logsig"}, false);

	CrossEntropy err(mlp, data);
	err.gradient();
	std::vector<double> g0 = mlp.gradients();

	err.classWeights({1.0, 1.0});
	err.gradient();
	std::vector<double> g1 = mlp.gradients();

	double d = maxAbsDiff(g0, g1);
	if (d > 1e-12) {
		std::cerr << "FAIL (gradient differs by " << d << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (maxdiff=" << d << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 2: a uniform scale {c,c} leaves the weighted-mean gradient unchanged
// ---------------------------------------------------------------------------
static bool testUniformScaleInvariant() {
	std::cout << "Test: classWeights({3,3}) == unweighted (weighted mean) ... ";
	nnh::rand::seed(11);
	DataSet data = buildImbalanced(40, 8);
	Mlp mlp({2, 4, 1}, {"tansig", "logsig"}, false);

	CrossEntropy err(mlp, data);
	err.gradient();
	std::vector<double> g0 = mlp.gradients();

	err.classWeights({3.0, 3.0});
	err.gradient();
	std::vector<double> g3 = mlp.gradients();

	double d = maxAbsDiff(g0, g3);
	if (d > 1e-9) {
		std::cerr << "FAIL (gradient differs by " << d << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (maxdiff=" << d << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 3: asymmetric weights actually change the gradient
// ---------------------------------------------------------------------------
static bool testAsymmetricChangesGradient() {
	std::cout << "Test: classWeights({1,5}) changes the gradient ... ";
	nnh::rand::seed(11);
	DataSet data = buildImbalanced(40, 8);
	Mlp mlp({2, 4, 1}, {"tansig", "logsig"}, false);

	CrossEntropy err(mlp, data);
	err.gradient();
	std::vector<double> g0 = mlp.gradients();

	err.classWeights({1.0, 5.0});
	err.gradient();
	std::vector<double> g4 = mlp.gradients();

	double d = maxAbsDiff(g0, g4);
	if (d < 1e-6) {
		std::cerr << "FAIL (gradient did not change, maxdiff=" << d << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (maxdiff=" << d << ")" << std::endl;
	return true;
}

// Train a model on the imbalanced set, optionally upweighting positives, and
// return the mean predicted probability over the positive patterns.
static double meanPosOutput(double posWeight) {
	nnh::rand::seed(99);
	srand(99);
	DataSet data = buildOverlap(40, 8, 8);
	Mlp mlp({2, 4, 1}, {"tansig", "logsig"}, false);

	CrossEntropy err(mlp, data);
	if (posWeight != 1.0) err.classWeights({1.0, posWeight});
	Adam trainer(mlp, data, err, 0.0, 16, 0.02);
	trainer.numEpochs(1500);
	std::ostringstream devnull;
	trainer.train(devnull);

	double sum = 0;
	uint nPos = 0;
	for (uint i = 0; i < data.size(); ++i) {
		Pattern& p = data.pattern(i);
		if (p.output()[0] > 0.5) {
			sum += mlp.propagate(p.input())[0];
			nPos++;
		}
	}
	return sum / nPos;
}

// ---------------------------------------------------------------------------
// Test 4: upweighting the minority raises its predicted probability
// ---------------------------------------------------------------------------
static bool testUpweightMinorityHelps() {
	std::cout << "Test: upweighting minority raises its predictions ... ";
	double plain = meanPosOutput(1.0);
	double weighted = meanPosOutput(8.0);

	if (!(weighted > plain)) {
		std::cerr << "FAIL (weighted mean pos output " << weighted << " not above plain " << plain
		          << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (plain=" << plain << ", weighted=" << weighted << ")" << std::endl;
	return true;
}

int main() {
	bool allPass = true;
	std::cout << "=== Class Weights Test Suite ===" << std::endl << std::endl;

	allPass &= testUniformWeightsIdentity();
	allPass &= testUniformScaleInvariant();
	allPass &= testAsymmetricChangesGradient();
	allPass &= testUpweightMinorityHelps();

	std::cout << std::endl;
	if (allPass) {
		std::cout << "All tests PASSED." << std::endl;
		return EXIT_SUCCESS;
	}
	std::cout << "Some tests FAILED." << std::endl;
	return EXIT_FAILURE;
}
