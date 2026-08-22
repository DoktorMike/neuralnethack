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

#include <algorithm>
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

// --- Boxed mode (doc/spec-boxed-adstock.md) -----------------------------

// Finite differences over ALL boxed params (box kernels, hill, logits).
bool gradientCheckBoxed(Adstock::Kernel kernel, bool hillOn, const char* name) {
	std::cout << "boxed gradient check (" << name << "): ";
	nnh::rand::seed(23);
	const uint C = 4, L = 6, P = 1, K = 3;
	DataSet ds = buildRandom(24, C, L, P);

	Mlp mlp({C + P, 4, 1}, {"tansig", "purelin"}, false);
	Adstock ads(C, L, P, kernel, K,
	            hillOn ? Adstock::Saturation::Hill : Adstock::Saturation::None);
	// Nudge every param off symmetric/default values
	for (uint j = 0; j < ads.nParams(); ++j)
		ads.params()[j] += 0.05 * (static_cast<int>(j % 7) - 3);
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
		const double numeric = (ep - em) / (2.0 * h) / 2.0; // 1/2: SSE convention
		const double d = std::abs(numeric - analytic[j]);
		if (!std::isfinite(d)) {
			std::cerr << "FAIL (non-finite gradient at param " << j << ")" << std::endl;
			return false;
		}
		maxdiff = std::max(maxdiff, d);
	}
	if (maxdiff > 1e-6) {
		std::cerr << "FAIL (max |numeric - analytic| = " << maxdiff << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (maxdiff " << maxdiff << ")" << std::endl;
	return true;
}

// 12 channels drawn from 3 known geometric boxes; the model must route
// channels to the right box and recover the box lambdas. Also asserts
// the entropy penalty hardens the routing (spec tests 2 and 4).
bool boxedRecovery() {
	std::cout << "boxed routing recovery: ";
	nnh::rand::seed(31);
	const uint C = 12, L = 10, K = 3;
	const double boxLambda[3] = {0.2, 0.5, 0.8};

	std::vector<std::vector<double>> w(K, std::vector<double>(L));
	for (uint k = 0; k < K; ++k) {
		double S = 0;
		for (uint l = 0; l < L; ++l) {
			w[k][l] = std::pow(boxLambda[k], l);
			S += w[k][l];
		}
		for (auto& v : w[k])
			v /= S;
	}

	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < 500; ++i) {
		std::vector<double> in(C * L);
		for (auto& v : in)
			v = nnh::rand::uniform();
		double y = 0;
		for (uint c = 0; c < C; ++c) {
			const uint b = c % K; // true box of channel c
			double a = 0;
			for (uint l = 0; l < L; ++l)
				a += w[b][l] * in[c * L + l];
			y += a / C;
		}
		std::vector<double> out = {y + 0.01 * (2.0 * nnh::rand::uniform() - 1.0)};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);

	// Two-phase schedule: explore with the penalty off (turning it on from
	// step one hardens the routing before the boxes separate), then turn
	// it on to push the routing to one-hot.
	Mlp mlp({C, 1}, {"purelin"}, false);
	mlp.adstock(Adstock(C, L, 0, Adstock::Kernel::Geometric, K));
	SummedSquare loss(mlp, ds);
	std::ostringstream sink;
	{
		Adam opt(mlp, ds, loss, 0.0, 32, 0.05);
		opt.numEpochs(400);
		opt.train(sink);
	}
	mlp.adstock()->entropyPenalty(0.01);
	{
		Adam opt(mlp, ds, loss, 0.0, 32, 0.02);
		opt.numEpochs(100);
		opt.train(sink);
	}

	// Canonicalize recovered boxes by lambda, then check assignments
	Adstock* a = mlp.adstock();
	std::vector<double> nat = a->naturalParams(); // K lambdas
	std::vector<uint> order(K);
	for (uint k = 0; k < K; ++k)
		order[k] = k;
	std::sort(order.begin(), order.end(), [&](uint i, uint j) { return nat[i] < nat[j]; });

	double maxLambdaErr = 0;
	for (uint k = 0; k < K; ++k)
		maxLambdaErr = std::max(maxLambdaErr, std::abs(nat[order[k]] - boxLambda[k]));

	std::vector<uint> canonOf(K);
	for (uint k = 0; k < K; ++k)
		canonOf[order[k]] = k;
	uint correct = 0;
	double minMaxPi = 1.0;
	const std::vector<uint> assign = a->boxAssignments();
	for (uint c = 0; c < C; ++c) {
		if (canonOf[assign[c]] == c % K) ++correct;
		const std::vector<double> pi = a->routingProbs(c);
		minMaxPi = std::min(minMaxPi, *std::max_element(pi.begin(), pi.end()));
	}
	if (correct < 10 || maxLambdaErr > 0.05 || minMaxPi < 0.9) {
		std::cerr << "FAIL (correct " << correct << "/12, lambda err " << maxLambdaErr
		          << ", min max-pi " << minMaxPi << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (" << correct << "/12 routed, lambda err " << maxLambdaErr
	          << ", min max-pi " << minMaxPi << ")" << std::endl;
	return true;
}

// Two boxes with distinct half-saturations; recovered ordering must hold
// (spec test 3).
bool saturationRecovery() {
	std::cout << "boxed saturation recovery: ";
	nnh::rand::seed(41);
	const uint C = 6, L = 8, K = 2;
	const double boxLambda[2] = {0.3, 0.7};
	const double boxHalf[2] = {0.3, 1.5};

	std::vector<std::vector<double>> w(K, std::vector<double>(L));
	for (uint k = 0; k < K; ++k) {
		double S = 0;
		for (uint l = 0; l < L; ++l) {
			w[k][l] = std::pow(boxLambda[k], l);
			S += w[k][l];
		}
		for (auto& v : w[k])
			v /= S;
	}
	auto hillFn = [](double a, double s) { return a / (a + s); };

	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < 600; ++i) {
		std::vector<double> in(C * L);
		for (auto& v : in)
			v = 2.0 * nnh::rand::uniform();
		double y = 0;
		for (uint c = 0; c < C; ++c) {
			const uint b = c % K;
			double a = 0;
			for (uint l = 0; l < L; ++l)
				a += w[b][l] * in[c * L + l];
			y += hillFn(a, boxHalf[b]) / C;
		}
		std::vector<double> out = {y + 0.005 * (2.0 * nnh::rand::uniform() - 1.0)};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);

	Mlp mlp({C, 1}, {"purelin"}, false);
	Adstock ads(C, L, 0, Adstock::Kernel::Geometric, K, Adstock::Saturation::Hill);
	ads.entropyPenalty(0.001);
	mlp.adstock(ads);
	SummedSquare loss(mlp, ds);
	Adam opt(mlp, ds, loss, 0.0, 32, 0.05);
	opt.numEpochs(800);
	std::ostringstream sink;
	opt.train(sink);

	// Order boxes by recovered lambda; half-saturations must order the
	// same way as the truth (fast box has the smaller half-saturation).
	Adstock* a = mlp.adstock();
	std::vector<double> nat = a->naturalParams(); // [K lambdas | K sat | K exp]
	const uint fast = nat[0] < nat[1] ? 0u : 1u;
	const double sFast = nat[K + fast], sSlow = nat[K + (1 - fast)];
	if (!(sFast < sSlow)) {
		std::cerr << "FAIL (half-saturations: fast " << sFast << ", slow " << sSlow << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS (half-sat fast " << sFast << " < slow " << sSlow << ")" << std::endl;
	return true;
}

// Label-switching canonicalization: an ensemble of one member plus a
// box-permuted copy must summarize with zero-width bands and stability 1.
bool labelSwitching() {
	std::cout << "boxed label-switching canonicalization: ";
	const uint C = 5, L = 8, K = 3;
	const uint ppk = 1; // geometric

	Mlp m1({C, 1}, {"purelin"}, false);
	Adstock a1(C, L, 0, Adstock::Kernel::Geometric, K, Adstock::Saturation::Hill);
	// Distinct params everywhere
	for (uint j = 0; j < a1.nParams(); ++j)
		a1.params()[j] += 0.13 * (static_cast<int>(j % 5) - 2);
	m1.adstock(a1);

	// Member 2: permute the boxes (rotation perm[k] = (k+1) % K)
	Mlp m2(m1);
	Adstock* a2 = m2.adstock();
	const std::vector<double> p1 = m1.adstock()->params();
	std::vector<double>& p2 = a2->params();
	for (uint k = 0; k < K; ++k) {
		const uint dst = (k + 1) % K;
		for (uint p = 0; p < ppk; ++p)
			p2[dst * ppk + p] = p1[k * ppk + p];
		p2[K * ppk + dst] = p1[K * ppk + k];         // sigma block
		p2[K * ppk + K + dst] = p1[K * ppk + K + k]; // nu block
	}
	const uint lOff = K * ppk + 2 * K;
	for (uint c = 0; c < C; ++c)
		for (uint k = 0; k < K; ++k)
			p2[lOff + c * K + (k + 1) % K] = p1[lOff + c * K + k];

	NeuralNetHack::Ensemble ens;
	ens.addMlp(std::make_unique<Mlp>(m1), 1.0);
	ens.addMlp(std::make_unique<Mlp>(m2), 1.0);

	auto s = EvalTools::Uncertainty::summarizeBoxedAdstock(ens, 0.1);
	double maxWidth = 0;
	for (uint k = 0; k < K; ++k)
		for (uint l = 0; l < L; ++l)
			maxWidth = std::max(maxWidth, s.kernelUpper[k][l] - s.kernelLower[k][l]);
	for (uint j = 0; j < s.paramMean.size(); ++j)
		maxWidth = std::max(maxWidth, s.paramUpper[j] - s.paramLower[j]);
	double minStab = 1.0;
	for (uint c = 0; c < C; ++c)
		minStab = std::min(minStab, s.stability[c]);
	if (maxWidth > 1e-12 || minStab < 1.0) {
		std::cerr << "FAIL (max band width " << maxWidth << ", min stability " << minStab << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

// NNH3 round-trip preserves boxed meta, params, temperature, predictions.
bool boxedSerialization() {
	std::cout << "NNH3 serialization round-trip: ";
	nnh::rand::seed(9);
	const uint C = 3, L = 5, P = 1, K = 2;
	Mlp mlp({C + P, 3, 1}, {"tansig", "logsig"}, false);
	Adstock ads(C, L, P, Adstock::Kernel::Weibull, K, Adstock::Saturation::Hill);
	for (uint j = 0; j < ads.nParams(); ++j)
		ads.params()[j] += 0.07 * (static_cast<int>(j % 4) - 1);
	ads.temperature(0.7);
	mlp.adstock(ads);

	std::stringstream buf;
	saveMlpBinary(mlp, buf);
	auto loaded = loadMlpBinary(buf);

	std::vector<double> x(C * L + P);
	for (auto& v : x)
		v = nnh::rand::uniform();
	const double y0 = mlp.propagate(x)[0];
	const double y1 = loaded->propagate(x)[0];
	const Adstock* la = loaded->adstock();
	if (!la || !la->boxed() || la->nBoxes() != K ||
	    la->saturation() != Adstock::Saturation::Hill || la->temperature() != 0.7 || y0 != y1) {
		std::cerr << "FAIL (y0 " << y0 << " y1 " << y1 << ")" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
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

	std::cout << std::endl;
	allPass &= gradientCheckBoxed(Adstock::Kernel::Geometric, false, "geometric");
	allPass &= gradientCheckBoxed(Adstock::Kernel::Geometric, true, "geometric+hill");
	allPass &= gradientCheckBoxed(Adstock::Kernel::Weibull, true, "weibull+hill");
	allPass &= boxedRecovery();
	allPass &= saturationRecovery();
	allPass &= labelSwitching();
	allPass &= boxedSerialization();

	std::cout << std::endl << (allPass ? "ALL PASS" : "SOME FAILED") << std::endl;
	return allPass ? 0 : 1;
}
