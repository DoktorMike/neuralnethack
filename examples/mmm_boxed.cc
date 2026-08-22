// Realistic boxed-adstock MMM example.
//
// 50 insertion types = 10 media channels x 5 messages, weekly data over
// 3 years (156 obs). Carryover regime is a property of the MEDIA (search
// decays in days, brand TV lingers for months, sponsorship peaks weeks
// after the flight); the message only scales effectiveness. Ground
// truth uses 5 regimes -- four decaying at different speeds plus one
// delayed-peak -- each with its own saturation, one of them S-shaped.
// Covariates: annual seasonality, unemployment (slow random walk,
// negative effect), linear trend.
//
// The model sees only raw lag windows and covariates. A boxed adstock
// stage (K=5, Weibull kernels, Hill saturation) must discover the
// regimes, route the 50 insertion types into them, and recover the
// carryover shapes -- 50 insertion types collapse to 5 effective ones.
//
// Estimating 50 independent lambdas/(k,s)/saturations from 156 rows
// would be hopeless; the boxes are what make it identifiable.
//
// Expected results (honest reading): the slow and delayed kernels are
// recovered nearly exactly -- including the S-shaped Hill exponent
// (~2.1 vs true 2.0) and the week-5 peak -- fast media route cleanly,
// and holdout R^2 lands near 0.85. The MIDDLE regimes blur into their
// neighbors: exact-box routing is ~24/50 but within-one-regime is
// ~47/50, because separating lambda 0.30 from 0.60 with ~20 flights of
// evidence per channel is close to the information limit of 156 weekly
// rows. The ensemble stability readout says exactly that: only ~16/50
// channels are stable across bootstrap members, which is the honest
// signal a real engagement should report instead of a confident
// point-routing.

#include "Ensemble.hh"
#include "Random.hh"
#include "evaltools/Uncertainty.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Adstock.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

constexpr uint MEDIA = 10, MSGS = 5;
constexpr uint C = MEDIA * MSGS; // 50 insertion types
constexpr uint L = 13;           // one quarter of weekly lags
constexpr uint P = 3;            // season, unemployment, trend
constexpr uint K = 5;            // boxes (architectural choice)
constexpr uint T = 156 + L - 1;  // simulate enough weeks for 156 usable rows

// Regime of media m: two media per regime.
uint regimeOf(uint media) { return media / 2; }

// True carryover kernels, normalized over the window. Regimes 0-3 decay
// geometrically at increasing lambda; regime 4 peaks at week 5.
std::vector<double> trueKernel(uint r) {
	std::vector<double> w(L);
	if (r < 4) {
		const double lambda[4] = {0.05, 0.30, 0.60, 0.85};
		for (uint l = 0; l < L; ++l)
			w[l] = std::pow(lambda[r], l);
	} else {
		for (uint l = 0; l < L; ++l)
			w[l] = std::pow(l + 1.0, 2.0) * std::exp(-(l + 1.0) / 3.0); // peak ~ week 5
	}
	double S = 0;
	for (auto v : w)
		S += v;
	for (auto& v : w)
		v /= S;
	return w;
}

// Per-regime saturation: hill(a; s, n). Regime 3 (long brand) is
// S-shaped (n = 2), the rest plain diminishing returns.
const double trueHalf[K] = {0.6, 0.8, 1.0, 1.2, 0.9};
const double trueExp[K] = {1.0, 1.0, 1.0, 2.0, 1.0};

double hillFn(double a, double s, double n) {
	const double an = std::pow(a, n), sn = std::pow(s, n);
	return an / (an + sn);
}

} // namespace

int main() {
	nnh::rand::seed(4711);

	// Weekly spend per insertion type: flighted campaigns.
	std::vector<std::vector<double>> spend(C, std::vector<double>(T, 0.0));
	for (uint c = 0; c < C; ++c) {
		bool on = false;
		for (uint t = 0; t < T; ++t) {
			if (nnh::rand::uniform() < 0.18) on = !on; // flighted, frequent transitions
			spend[c][t] = on ? 1.0 + 2.0 * nnh::rand::uniform() : 0.0;
		}
	}

	// Covariates.
	std::vector<double> unemployment(T);
	double u = 0.0;
	for (uint t = 0; t < T; ++t) {
		u = 0.98 * u + 0.1 * (2.0 * nnh::rand::uniform() - 1.0);
		unemployment[t] = u;
	}

	// Message effectiveness: creative quality scales the media effect.
	const double msgBeta[MSGS] = {0.6, 0.8, 1.0, 1.2, 1.4};

	std::vector<std::vector<double>> tk(K);
	for (uint r = 0; r < K; ++r)
		tk[r] = trueKernel(r);

	auto core = std::make_shared<CoreDataSet>();
	for (uint t = L - 1; t < T; ++t) {
		std::vector<double> in;
		in.reserve(C * L + P);
		double sales = 2.0; // base
		for (uint c = 0; c < C; ++c) {
			const uint media = c / MSGS, msg = c % MSGS, r = regimeOf(media);
			double a = 0;
			for (uint l = 0; l < L; ++l) {
				const double x = spend[c][t - l];
				in.push_back(x);
				a += tk[r][l] * x;
			}
			sales += 0.8 * msgBeta[msg] * hillFn(a, trueHalf[r], trueExp[r]);
		}
		const double season = std::sin(2.0 * M_PI * (t % 52) / 52.0);
		const double trend = static_cast<double>(t) / T;
		in.push_back(season);
		in.push_back(unemployment[t]);
		in.push_back(trend);
		sales += 0.5 * season - 0.8 * unemployment[t] + 1.0 * trend;
		sales += 0.03 * (2.0 * nnh::rand::uniform() - 1.0);
		std::vector<double> out = {sales};
		core->addPattern(Pattern(std::to_string(t), in, out));
	}

	// Chronological 80/20 split.
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

	std::printf("50 insertion types (10 media x 5 messages), %u weekly obs "
	            "(%u train / %u holdout), K=%u boxes\n\n",
	            n, nTrn, n - nTrn, K);

	// Model: linear head (156 obs cannot feed a wide dense layer), boxed
	// Weibull adstock with Hill saturation.
	Mlp mlp({C + P, 1}, {"purelin"}, false);
	mlp.adstock(Adstock(C, L, P, Adstock::Kernel::Weibull, K, Adstock::Saturation::Hill));

	// Media effects are non-negative by construction: constrain the head's
	// media columns (projected gradient; covariates and bias stay free).
	// Beyond realism this keeps early routing gradients pointing the right
	// way -- a channel whose randomly-initialized beta starts negative
	// would otherwise drift toward the wrong box.
	mlp.nonNegative(0, 0, C - 1);

	SummedSquare loss(mlp, trn);
	std::ostringstream sink;
	// Phase 1: explore with the entropy penalty OFF (see the warmup
	// warning in Adstock.hh), phase 2: harden the routing.
	{
		Adam opt(mlp, trn, loss, 0.0, 32, 0.02);
		opt.numEpochs(800);
		opt.train(sink);
	}
	mlp.adstock()->entropyPenalty(0.01);
	{
		Adam opt(mlp, trn, loss, 0.0, 32, 0.005);
		opt.numEpochs(200);
		opt.train(sink);
	}

	Adstock* a = mlp.adstock();

	// Canonicalize: recovered boxes and true regimes both sorted by mean
	// carryover lag, then compare.
	auto meanLag = [&](const std::vector<double>& w) {
		double m = 0;
		for (uint l = 0; l < L; ++l)
			m += l * w[l];
		return m;
	};
	std::vector<uint> recOrder(K), trueOrder(K);
	for (uint k = 0; k < K; ++k)
		recOrder[k] = trueOrder[k] = k;
	std::sort(recOrder.begin(), recOrder.end(),
	          [&](uint i, uint j) { return meanLag(a->boxKernel(i)) < meanLag(a->boxKernel(j)); });
	std::sort(trueOrder.begin(), trueOrder.end(),
	          [&](uint i, uint j) { return meanLag(tk[i]) < meanLag(tk[j]); });
	std::vector<uint> canonOfRec(K), canonOfTrue(K);
	for (uint k = 0; k < K; ++k) {
		canonOfRec[recOrder[k]] = k;
		canonOfTrue[trueOrder[k]] = k;
	}

	std::printf("Recovered boxes (sorted fast -> slow, first 8 lags):\n");
	for (uint k = 0; k < K; ++k) {
		const uint rb = recOrder[k], tb = trueOrder[k];
		const auto rw = a->boxKernel(rb);
		std::printf("box %u  true:      ", k);
		for (uint l = 0; l < 8; ++l)
			std::printf("%.3f ", tk[tb][l]);
		std::printf(" half-sat %.2f exp %.2f\n", trueHalf[tb], trueExp[tb]);
		std::printf("       recovered: ");
		for (uint l = 0; l < 8; ++l)
			std::printf("%.3f ", rw[l]);
		std::printf(" half-sat %.2f exp %.2f\n", a->boxSaturation(rb), a->boxHillExponent(rb));
	}

	// Routing table: media x message -> canonical box; count correct.
	const std::vector<uint> assign = a->boxAssignments();
	uint correct = 0, within1 = 0;
	// One row per media; the 5 cells are that media's messages (so each
	// cell is one of the 50 insertion types). A cell shows the box the
	// model routed that insertion type into, boxes numbered 0..K-1 from
	// fastest to slowest carryover. Carryover is a property of the media,
	// so every cell in a row should equal the row's expected box.
	std::printf("\nRouting: cell = box the model chose for that media x message "
	            "(0 = fastest carryover ... %u = slowest, '.' = matches expected)\n",
	            K - 1);
	std::printf("%-8s %-12s %s\n", "media", "expected box", "message 0..4");
	for (uint media = 0; media < MEDIA; ++media) {
		const uint want = canonOfTrue[regimeOf(media)];
		std::printf("%-8u %-12u ", media, want);
		for (uint msg = 0; msg < MSGS; ++msg) {
			const uint c = media * MSGS + msg;
			const uint got = canonOfRec[assign[c]];
			if (got == want)
				std::printf(".  ");
			else
				std::printf("%u  ", got);
			if (got == want) ++correct;
			if (got == want || got + 1 == want || want + 1 == got) ++within1;
		}
		std::printf("\n");
	}
	std::printf("exact box: %u / %u, within one carryover regime: %u / %u\n", correct, C,
	            within1, C);

	// Holdout fit.
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
	std::printf("holdout R^2 (last %u weeks): %.4f\n", n - nTrn, 1.0 - ssRes / ssTot);

	// Ensemble: bootstrap members, then label-switching-safe stability.
	// Assignment stability is the client-facing readout: a channel whose
	// routing flips across members should not be presented as "known".
	const uint M = 8;
	NeuralNetHack::Ensemble ens;
	for (uint m = 0; m < M; ++m) {
		Mlp member(mlp);
		member.regenerateWeights();
		std::vector<uint> idx(nTrn);
		for (auto& v : idx)
			v = trnIdx[static_cast<uint>(nnh::rand::uniform() * nTrn) % nTrn];
		DataSet boot;
		boot.coreDataSet(core);
		boot.indices(idx);
		SummedSquare bloss(member, boot);
		{
			Adam opt(member, boot, bloss, 0.0, 32, 0.02);
			opt.numEpochs(800);
			opt.train(sink);
		}
		member.adstock()->entropyPenalty(0.01);
		{
			Adam opt(member, boot, bloss, 0.0, 32, 0.005);
			opt.numEpochs(200);
			opt.train(sink);
		}
		ens.addMlp(std::make_unique<Mlp>(member), 1.0);
	}
	const auto bsum = EvalTools::Uncertainty::summarizeBoxedAdstock(ens, 0.1);
	uint modalCorrect = 0, stable = 0;
	double meanStab = 0;
	for (uint c = 0; c < C; ++c) {
		if (bsum.modalBox[c] == canonOfTrue[regimeOf(c / MSGS)]) ++modalCorrect;
		meanStab += bsum.stability[c] / C;
		if (bsum.stability[c] >= 0.75) ++stable;
	}
	std::printf("\nEnsemble (%u members): modal-box routing %u / %u correct, "
	            "mean stability %.2f, %u / %u channels stable (>= 6/8 members agree)\n",
	            M, modalCorrect, C, meanStab, stable, C);
	return 0;
}
