#include "Ensemble.hh"
#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "evaltools/Uncertainty.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Adstock.hh"
#include "mlp/Mlp.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/Serialization.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

// Random dataset with C channels x L lags + P covariates in, 1 output.
DataSet buildRandom(uint n, uint C, uint L, uint P) {
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		std::vector<double> in(C * L + P);
		for (auto& v : in)
			v = nnh::rand::uniform();
		std::vector<double> out = {nnh::rand::uniform()};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

// Central finite differences on the adstock params through the full
// Mlp + SummedSquare error vs the analytic gradients from gradient().
bool gradientCheck(Adstock::Kernel kernel, const char* name) {
	std::cout << "gradient check (" << name << "): ";
	nnh::rand::seed(7);
	const uint C = 2, L = 6, P = 1;
	DataSet ds = buildRandom(24, C, L, P);

	std::vector<uint> arch = {C + P, 4, 1};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, false);
	Adstock ads(C, L, P, kernel);
	// Nudge params off their symmetric defaults
	for (uint j = 0; j < ads.nParams(); ++j)
		ads.params()[j] += 0.1 * (j + 1);
	mlp.adstock(ads);

	SummedSquare loss(mlp, ds);
	loss.gradient(mlp, ds);
	std::vector<double> analytic = mlp.adstock()->gradients();

	const double h = 1e-6;
	double maxdiff = 0.0;
	for (uint j = 0; j < mlp.adstock()->nParams(); ++j) {
		double& p = mlp.adstock()->params()[j];
		const double p0 = p;
		p = p0 + h;
		const double ep = loss.outputError(mlp, ds);
		p = p0 - h;
		const double em = loss.outputError(mlp, ds);
		p = p0;
		// SummedSquare uses delta = (t - o) without the factor 2, so the
		// library's gradients are (1/2) dE/dparam. The same halving holds
		// for ordinary layer weights; verify the adstock params match it.
		const double numeric = (ep - em) / (2.0 * h) / 2.0;
		maxdiff = std::max(maxdiff, std::abs(numeric - analytic[j]));
	}
	if (maxdiff > 1e-6) {
		std::cerr << "FAIL (max |numeric - analytic| = " << maxdiff << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (maxdiff " << maxdiff << ")" << std::endl;
	return true;
}

// Generate data from a known geometric kernel and check training
// recovers lambda.
bool recovery() {
	std::cout << "geometric lambda recovery: ";
	nnh::rand::seed(11);
	const uint C = 1, L = 12, P = 0;
	const double trueLambda = 0.7;

	// True kernel
	std::vector<double> w(L);
	double S = 0;
	for (uint l = 0; l < L; ++l) {
		w[l] = std::pow(trueLambda, l);
		S += w[l];
	}
	for (auto& v : w)
		v /= S;

	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < 400; ++i) {
		std::vector<double> in(L);
		for (auto& v : in)
			v = nnh::rand::uniform();
		double a = 0;
		for (uint l = 0; l < L; ++l)
			a += w[l] * in[l];
		std::vector<double> out = {a};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);

	std::vector<uint> arch = {C, 1};
	std::vector<std::string> types = {"purelin"};
	Mlp mlp(arch, types, false);
	mlp.adstock(Adstock(C, L, P, Adstock::Kernel::Geometric));

	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, 64, 0.05);
	opt.numEpochs(400);
	std::ostringstream sink;
	opt.train(sink);

	const double rho = mlp.adstock()->params()[0];
	const double lambda = 1.0 / (1.0 + std::exp(-rho));
	if (std::abs(lambda - trueLambda) > 0.05) {
		std::cerr << "FAIL (recovered lambda " << lambda << ", true " << trueLambda << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS (lambda " << lambda << ")" << std::endl;
	return true;
}

// Weibull must recover a DELAYED peak (k > 1 regime). Ground truth is a
// gamma-shaped bump peaking at lag 4 -- deliberately not a Weibull, so
// this also checks graceful behavior under kernel misspecification --
// behind a saturating response.
bool delayedPeakRecovery() {
	std::cout << "weibull delayed-peak recovery: ";
	nnh::rand::seed(1);
	const uint L = 14;

	std::vector<double> w(L);
	double S = 0;
	for (uint l = 0; l < L; ++l) {
		const double t = l;
		w[l] = t * t * std::exp(-t / 2.0); // peak at lag 4
		S += w[l];
	}
	for (auto& v : w)
		v /= S;

	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < 600; ++i) {
		std::vector<double> in(L);
		for (auto& v : in)
			v = nnh::rand::uniform();
		double a = 0;
		for (uint l = 0; l < L; ++l)
			a += w[l] * in[l];
		std::vector<double> out = {a / (a + 0.5)}; // saturation
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);

	Mlp mlp({1, 4, 1}, {"tansig", "purelin"}, false);
	mlp.adstock(Adstock(1, L, 0, Adstock::Kernel::Weibull));
	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, 32, 0.02);
	opt.numEpochs(800);
	std::ostringstream sink;
	opt.train(sink);

	const auto r = mlp.adstock()->kernelWeights(0);
	uint truePeak = 0, recPeak = 0;
	for (uint l = 1; l < L; ++l) {
		if (w[l] > w[truePeak]) truePeak = l;
		if (r[l] > r[recPeak]) recPeak = l;
	}
	const int diff = static_cast<int>(recPeak) - static_cast<int>(truePeak);
	if (recPeak == 0 || std::abs(diff) > 1) {
		std::cerr << "FAIL (true peak lag " << truePeak << ", recovered lag " << recPeak << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS (peak lag " << recPeak << ", true " << truePeak << ")" << std::endl;
	return true;
}

// L-BFGS sees adstock params through the flat weight vector.
bool quasiNewtonTrains() {
	std::cout << "L-BFGS trains adstock params: ";
	nnh::rand::seed(3);
	const uint C = 2, L = 5, P = 0;
	DataSet ds = buildRandom(64, C, L, P);
	std::vector<uint> arch = {C, 3, 1};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, false);
	mlp.adstock(Adstock(C, L, P, Adstock::Kernel::Weibull));
	const std::vector<double> before = mlp.adstock()->params();

	SummedSquare loss(mlp, ds);
	QuasiNewton opt(mlp, ds, loss, 0.0, 64);
	opt.numEpochs(20);
	std::ostringstream sink;
	opt.train(sink);

	double moved = 0;
	for (uint j = 0; j < before.size(); ++j)
		moved = std::max(moved, std::abs(mlp.adstock()->params()[j] - before[j]));
	if (moved < 1e-8) {
		std::cerr << "FAIL (params did not move)" << std::endl;
		return false;
	}
	std::cout << "PASS (max param move " << moved << ")" << std::endl;
	return true;
}

// Bootstrap-ensemble kernel-parameter uncertainty: the percentile band
// over member lambdas should cover the true lambda and have nonzero
// width; the kernel-weight bands should cover the true kernel.
bool ensembleUncertainty() {
	std::cout << "ensemble kernel uncertainty: ";
	nnh::rand::seed(17);
	const uint C = 1, L = 10, P = 0;
	const double trueLambda = 0.6;

	std::vector<double> w(L);
	double S = 0;
	for (uint l = 0; l < L; ++l) {
		w[l] = std::pow(trueLambda, l);
		S += w[l];
	}
	for (auto& v : w)
		v /= S;

	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < 300; ++i) {
		std::vector<double> in(L);
		for (auto& v : in)
			v = nnh::rand::uniform();
		double a = 0;
		for (uint l = 0; l < L; ++l)
			a += w[l] * in[l];
		// mild noise so resamples actually disagree
		std::vector<double> out = {a + 0.02 * (2.0 * nnh::rand::uniform() - 1.0)};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet full;
	full.coreDataSet(core);

	Mlp proto({C, 1}, {"purelin"}, false);
	proto.adstock(Adstock(C, L, P, Adstock::Kernel::Geometric));
	SummedSquare loss(proto, full);
	Adam opt(proto, full, loss, 0.0, 32, 0.05);
	opt.numEpochs(250);

	NeuralNetHack::Ensemble ens;
	const uint M = 8;
	std::ostringstream sink;
	for (uint m = 0; m < M; ++m) {
		std::vector<uint> idx(core->size());
		for (auto& v : idx)
			v = static_cast<uint>(nnh::rand::uniform() * core->size()) % core->size();
		DataSet boot;
		boot.coreDataSet(core);
		boot.indices(idx);
		ens.addMlp(opt.trainNew(boot, sink), 1.0);
	}

	auto s = EvalTools::Uncertainty::summarizeAdstock(ens, 0.1);
	const double lo = s.paramLower[0], hi = s.paramUpper[0];
	bool ok = lo < trueLambda && trueLambda < hi && hi - lo > 1e-4 && hi - lo < 0.3;
	for (uint l = 0; l < L && ok; ++l)
		ok = s.weightLower[0][l] <= w[l] + 0.02 && w[l] - 0.02 <= s.weightUpper[0][l];
	if (!ok) {
		std::cerr << "FAIL (lambda band [" << lo << ", " << hi << "], true " << trueLambda << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS (lambda 90% band [" << lo << ", " << hi << "])" << std::endl;
	return true;
}

// Save/load round-trip preserves adstock meta, params, and predictions.
bool serializationRoundTrip() {
	std::cout << "NNH2 serialization round-trip: ";
	nnh::rand::seed(5);
	const uint C = 2, L = 4, P = 1;
	std::vector<uint> arch = {C + P, 3, 1};
	std::vector<std::string> types = {"tansig", "logsig"};
	Mlp mlp(arch, types, false);
	Adstock ads(C, L, P, Adstock::Kernel::Geometric);
	ads.params()[0] = 0.4;
	ads.params()[1] = -0.9;
	mlp.adstock(ads);

	std::stringstream buf;
	saveMlpBinary(mlp, buf);
	auto loaded = loadMlpBinary(buf);

	std::vector<double> x(C * L + P);
	for (auto& v : x)
		v = nnh::rand::uniform();
	const double y0 = mlp.propagate(x)[0];
	const double y1 = loaded->propagate(x)[0];
	if (!loaded->adstock() || loaded->adstock()->kernel() != Adstock::Kernel::Geometric ||
	    loaded->adstock()->nLags() != L || y0 != y1) {
		std::cerr << "FAIL (y0 " << y0 << " y1 " << y1 << ")" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== Adstock Test Suite ===" << std::endl << std::endl;

	allPass &= gradientCheck(Adstock::Kernel::Geometric, "geometric");
	allPass &= gradientCheck(Adstock::Kernel::Weibull, "weibull");
	allPass &= recovery();
	allPass &= delayedPeakRecovery();
	allPass &= quasiNewtonTrains();
	allPass &= ensembleUncertainty();
	allPass &= serializationRoundTrip();

	std::cout << std::endl << (allPass ? "ALL PASS" : "SOME FAILED") << std::endl;
	return allPass ? 0 : 1;
}
