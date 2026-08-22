// Marketing-mix-style example: recover per-channel carryover (adstock)
// kernels jointly with a saturating response network.
//
// Synthetic ground truth: 3 media channels with different geometric
// carryover (fast/medium/slow decay), a saturating response per channel,
// plus a weekly-seasonality covariate. The model sees only the raw lag
// window per channel and must learn the lag structure itself -- with
// 1 trained kernel parameter per channel instead of L free weights per
// channel per neuron.
//
// Prints true vs recovered kernels and holdout fit.

#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Adstock.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

constexpr uint C = 3;   // media channels
constexpr uint L = 14;  // lag window (days)
constexpr uint P = 1;   // passthrough covariates (weekly seasonality)
constexpr uint T = 900; // days simulated

const double trueLambda[C] = {0.2, 0.5, 0.8}; // fast / medium / slow decay
const double channelBeta[C] = {1.0, 0.8, 1.2};
const double channelSat[C] = {1.5, 2.0, 2.5}; // saturation half-points

std::vector<double> geometricKernel(double lambda) {
	std::vector<double> w(L);
	double S = 0;
	for (uint l = 0; l < L; ++l) {
		w[l] = std::pow(lambda, l);
		S += w[l];
	}
	for (auto& v : w)
		v /= S;
	return w;
}

double saturate(double a, double half) { // Hill-style diminishing returns
	return a / (a + half);
}

} // namespace

int main() {
	nnh::rand::seed(2026);

	// Simulate daily spend per channel (bursty: campaigns on/off)
	std::vector<std::vector<double>> spend(C, std::vector<double>(T));
	for (uint c = 0; c < C; ++c) {
		bool on = false;
		for (uint t = 0; t < T; ++t) {
			if (nnh::rand::uniform() < 0.05) on = !on;
			spend[c][t] = on ? 2.0 + 3.0 * nnh::rand::uniform() : 0.2 * nnh::rand::uniform();
		}
	}

	// Ground-truth response: adstock -> saturation -> sum + seasonality + noise
	std::vector<std::vector<double>> trueW(C);
	for (uint c = 0; c < C; ++c)
		trueW[c] = geometricKernel(trueLambda[c]);

	auto core = std::make_shared<CoreDataSet>();
	for (uint t = L - 1; t < T; ++t) {
		std::vector<double> in;
		in.reserve(C * L + P);
		double sales = 0.5; // base
		for (uint c = 0; c < C; ++c) {
			double a = 0;
			for (uint l = 0; l < L; ++l) {
				const double x = spend[c][t - l];
				in.push_back(x); // lag 0 = today, older lags after
				a += trueW[c][l] * x;
			}
			sales += channelBeta[c] * saturate(a, channelSat[c]);
		}
		const double season = std::sin(2.0 * M_PI * (t % 7) / 7.0);
		in.push_back(season);
		sales += 0.3 * season + 0.05 * (2.0 * nnh::rand::uniform() - 1.0);
		std::vector<double> out = {sales};
		core->addPattern(Pattern(std::to_string(t), in, out));
	}

	// Train / holdout split (chronological)
	const uint n = core->size();
	const uint nTrn = (n * 4) / 5;
	std::vector<uint> trnIdx(nTrn), tstIdx(n - nTrn);
	for (uint i = 0; i < nTrn; ++i)
		trnIdx[i] = i;
	for (uint i = nTrn; i < n; ++i)
		tstIdx[i - nTrn] = i;
	DataSet trn, tst;
	trn.coreDataSet(core);
	trn.indices(trnIdx);
	tst.coreDataSet(core);
	tst.indices(tstIdx);

	// Model: adstock stage (geometric, trained) + saturating MLP
	std::vector<uint> arch = {C + P, 8, 1};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, false);
	mlp.adstock(Adstock(C, L, P, Adstock::Kernel::Geometric));

	SummedSquare loss(mlp, trn);
	Adam opt(mlp, trn, loss, 0.0, 32, 0.01);
	opt.numEpochs(600);
	std::ostringstream sink;
	opt.train(sink);

	// Report recovered kernels
	std::printf("Per-channel carryover, true vs recovered (%u trained params total):\n\n",
	            mlp.adstock()->nParams());
	for (uint c = 0; c < C; ++c) {
		const double rho = mlp.adstock()->params()[c];
		const double lambda = 1.0 / (1.0 + std::exp(-rho));
		std::printf("channel %u: true lambda %.2f  recovered %.3f\n", c, trueLambda[c], lambda);
		auto w = mlp.adstock()->kernelWeights(c);
		std::printf("  true kernel:      ");
		for (uint l = 0; l < 7; ++l)
			std::printf("%.3f ", trueW[c][l]);
		std::printf("...\n  recovered kernel: ");
		for (uint l = 0; l < 7; ++l)
			std::printf("%.3f ", w[l]);
		std::printf("...\n");
	}

	// Holdout fit
	double ssRes = 0, ssTot = 0, mean = 0;
	for (uint i = 0; i < tst.size(); ++i)
		mean += tst.pattern(i).output()[0];
	mean /= tst.size();
	for (uint i = 0; i < tst.size(); ++i) {
		Pattern& p = tst.pattern(i);
		const double yhat = mlp.propagate(p.input())[0];
		const double y = p.output()[0];
		ssRes += (y - yhat) * (y - yhat);
		ssTot += (y - mean) * (y - mean);
	}
	std::printf("\nHoldout R^2 (last %u days): %.4f\n", n - nTrn, 1.0 - ssRes / ssTot);
	return 0;
}
