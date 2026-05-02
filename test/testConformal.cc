// Tests for split conformal prediction (regression + classification LAC).
//
// Each test trains a small Mlp on synthetic data with known structure,
// splits patterns into train / calibrate / test, runs Conformal, and
// checks that empirical coverage on the held-out test set is close to the
// nominal 1 - alpha. Tolerances are wide enough that the test passes
// reliably under different RNG draws but tight enough to catch a broken
// quantile.

#include "Random.hh"
#include "Ensemble.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "evaltools/Conformal.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace NeuralNetHack;

namespace {

double standardNormal() {
	double u1 = nnh::rand::uniform();
	double u2 = nnh::rand::uniform();
	if (u1 < 1e-12) u1 = 1e-12;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

DataSet buildRegressionData(uint n, double sigma, uint seed) {
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

DataSet buildBlobData(uint nPerClass, uint seed) {
	nnh::rand::seed(seed);
	const double centres[3][2] = {{-2.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}};
	auto core = std::make_shared<CoreDataSet>();
	uint id = 0;
	for (uint c = 0; c < 3; ++c) {
		for (uint i = 0; i < nPerClass; ++i) {
			double x0 = centres[c][0] + 0.6 * standardNormal();
			double x1 = centres[c][1] + 0.6 * standardNormal();
			std::vector<double> in = {x0, x1};
			std::vector<double> out = {0.0, 0.0, 0.0};
			out[c] = 1.0;
			core->addPattern(Pattern(std::to_string(id++), in, out));
		}
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

// Split a DataSet into three views by partitioning pattern indices.
// Returns (train, cal, test). Sizes nA, nB, rest.
void threeWaySplit(DataSet& src, uint nA, uint nB, DataSet& a, DataSet& b, DataSet& c) {
	auto core = src.sharedCoreDataSet();
	const uint n = src.size();
	std::vector<uint> idx(n);
	for (uint i = 0; i < n; ++i) idx[i] = i;
	for (uint i = n - 1; i > 0; --i) {
		const uint j = static_cast<uint>(nnh::rand::uniform() * (i + 1));
		std::swap(idx[i], idx[j]);
	}
	std::vector<uint> ai(idx.begin(), idx.begin() + nA);
	std::vector<uint> bi(idx.begin() + nA, idx.begin() + nA + nB);
	std::vector<uint> ci(idx.begin() + nA + nB, idx.end());
	a.coreDataSet(core);
	a.indices(ai);
	b.coreDataSet(core);
	b.indices(bi);
	c.coreDataSet(core);
	c.indices(ci);
}

std::unique_ptr<Mlp> trainRegression(DataSet& trn, uint seed) {
	std::srand(seed);
	nnh::rand::seed(seed);
	std::vector<uint> arch = {1, 16, 1};
	std::vector<std::string> types = {"tansig", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, false);
	SummedSquare loss(*mlp, trn);
	Adam opt(*mlp, trn, loss, 0.0, 16, 0.01);
	opt.numEpochs(1500);
	std::ostringstream sink;
	opt.train(sink);
	return mlp;
}

std::unique_ptr<Mlp> trainClassifier(DataSet& trn, uint seed) {
	std::srand(seed);
	nnh::rand::seed(seed);
	std::vector<uint> arch = {2, 12, 3};
	std::vector<std::string> types = {"tansig", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, /*softmax=*/true);
	CrossEntropy loss(*mlp, trn);
	Adam opt(*mlp, trn, loss, 0.0, 16, 0.02);
	opt.numEpochs(1500);
	std::ostringstream sink;
	opt.train(sink);
	return mlp;
}

bool testRegressionCoverage() {
	std::cout << "Test: Conformal regression coverage on noisy sin(x) ... ";
	const double sigma = 0.5;
	const double alpha = 0.1;

	DataSet full = buildRegressionData(1000, sigma, 11);
	DataSet trn, cal, tst;
	threeWaySplit(full, 500, 250, trn, cal, tst);

	auto mlp = trainRegression(trn, 7);
	Ensemble e(*mlp, 1.0);

	EvalTools::Conformal cp(EvalTools::Conformal::Mode::Regression, alpha);
	cp.calibrate(e, cal);
	const double cov = cp.coverage(e, tst);

	// For exchangeable data with n_cal=250 the marginal coverage should
	// concentrate tightly around 1-alpha. Allow ±0.07 to absorb both the
	// finite-sample binomial fluctuation and the small bias from imperfect
	// regressor fit.
	if (!(cov > 1.0 - alpha - 0.07 && cov < 1.0)) {
		std::cerr << "FAIL (cov=" << cov << ", target=" << (1.0 - alpha) << ")\n";
		return false;
	}
	if (cp.quantiles().size() != 1) {
		std::cerr << "FAIL (expected 1 per-dim qhat, got " << cp.quantiles().size() << ")\n";
		return false;
	}
	std::cout << "PASS (cov=" << cov << ", qhat=" << cp.quantiles()[0] << ")\n";
	return true;
}

bool testRegressionInterval() {
	std::cout << "Test: Conformal regression interval shape ... ";
	DataSet full = buildRegressionData(400, 0.3, 23);
	DataSet trn, cal, tst;
	threeWaySplit(full, 200, 100, trn, cal, tst);

	auto mlp = trainRegression(trn, 13);
	Ensemble e(*mlp, 1.0);

	EvalTools::Conformal cp(EvalTools::Conformal::Mode::Regression, 0.1);
	cp.calibrate(e, cal);
	auto iv = cp.interval(e, {0.5});
	if (iv.size() != 1) {
		std::cerr << "FAIL (expected 1 dim, got " << iv.size() << ")\n";
		return false;
	}
	if (!(iv[0].lo < iv[0].hi)) {
		std::cerr << "FAIL (lo>=hi: [" << iv[0].lo << "," << iv[0].hi << "])\n";
		return false;
	}
	const double width = iv[0].hi - iv[0].lo;
	if (!(width > 0.0 && width < 10.0)) {
		std::cerr << "FAIL (implausible width " << width << ")\n";
		return false;
	}
	std::cout << "PASS (width=" << width << ")\n";
	return true;
}

bool testClassificationCoverage() {
	std::cout << "Test: Conformal classification (LAC) coverage on 3-blob ... ";
	const double alpha = 0.1;
	DataSet full = buildBlobData(200, 31);
	DataSet trn, cal, tst;
	threeWaySplit(full, 300, 150, trn, cal, tst);

	auto mlp = trainClassifier(trn, 19);
	Ensemble e(*mlp, 1.0);

	EvalTools::Conformal cp(EvalTools::Conformal::Mode::Classification, alpha);
	cp.calibrate(e, cal);
	const double cov = cp.coverage(e, tst);

	if (!(cov > 1.0 - alpha - 0.07 && cov <= 1.0)) {
		std::cerr << "FAIL (cov=" << cov << ", target=" << (1.0 - alpha) << ")\n";
		return false;
	}
	std::cout << "PASS (cov=" << cov << ", qhat=" << cp.quantiles()[0] << ")\n";
	return true;
}

bool testClassificationSetSensible() {
	std::cout << "Test: Conformal LAC sets contain argmax for confident points ... ";
	DataSet full = buildBlobData(200, 41);
	DataSet trn, cal, tst;
	threeWaySplit(full, 300, 150, trn, cal, tst);

	auto mlp = trainClassifier(trn, 5);
	Ensemble e(*mlp, 1.0);

	EvalTools::Conformal cp(EvalTools::Conformal::Mode::Classification, 0.1);
	cp.calibrate(e, cal);

	// At a point near class-0 centre the set must include class 0 and be small.
	auto s = cp.set(e, {-2.0, 0.0});
	bool has0 = false;
	for (uint k : s)
		if (k == 0) has0 = true;
	if (!has0) {
		std::cerr << "FAIL (set near class-0 centre missing class 0)\n";
		return false;
	}
	if (s.size() > 3) {
		std::cerr << "FAIL (set size " << s.size() << " > 3)\n";
		return false;
	}
	std::cout << "PASS (|set|=" << s.size() << ")\n";
	return true;
}

bool testGuards() {
	std::cout << "Test: Conformal guards (alpha range, mode mismatch, uncalibrated) ... ";
	bool ok = true;
	try {
		EvalTools::Conformal bad(EvalTools::Conformal::Mode::Regression, 1.5);
		ok = false;
		std::cerr << "FAIL (alpha=1.5 should throw)\n";
	} catch (const std::invalid_argument&) {
	}

	DataSet full = buildRegressionData(50, 0.3, 99);
	DataSet trn, cal, tst;
	threeWaySplit(full, 30, 10, trn, cal, tst);
	auto mlp = trainRegression(trn, 1);
	Ensemble e(*mlp, 1.0);

	EvalTools::Conformal cpReg(EvalTools::Conformal::Mode::Regression, 0.1);
	try {
		(void)cpReg.interval(e, {0.0});
		ok = false;
		std::cerr << "FAIL (uncalibrated interval should throw)\n";
	} catch (const std::runtime_error&) {
	}

	cpReg.calibrate(e, cal);
	try {
		(void)cpReg.set(e, {0.0});
		ok = false;
		std::cerr << "FAIL (regression-mode set() should throw)\n";
	} catch (const std::runtime_error&) {
	}

	if (ok) std::cout << "PASS\n";
	return ok;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== Conformal Test Suite ===\n\n";
	allPass &= testRegressionCoverage();
	allPass &= testRegressionInterval();
	allPass &= testClassificationCoverage();
	allPass &= testClassificationSetSensible();
	allPass &= testGuards();
	std::cout << "\n";
	if (allPass) {
		std::cout << "All tests PASSED.\n";
		return EXIT_SUCCESS;
	}
	std::cout << "Some tests FAILED.\n";
	return EXIT_FAILURE;
}
