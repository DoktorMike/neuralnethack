#include "Uncertainty.hh"
#include "../mlp/Mlp.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace NeuralNetHack;
using std::vector;

namespace EvalTools {
namespace Uncertainty {

double predictiveEntropy(const vector<double>& p) {
	double h = 0;
	for (double q : p)
		if (q > 1e-12) h -= q * std::log(q);
	return h;
}

EntropyDecomposition decomposeEntropy(const vector<vector<double>>& memberProbs) {
	assert(!memberProbs.empty());
	const std::size_t M = memberProbs.size();
	const std::size_t K = memberProbs[0].size();

	vector<double> mean(K, 0.0);
	double aleatoric = 0.0;
	for (const auto& p : memberProbs) {
		assert(p.size() == K);
		for (std::size_t k = 0; k < K; ++k)
			mean[k] += p[k];
		aleatoric += predictiveEntropy(p);
	}
	const double inv = 1.0 / (double)M;
	for (double& v : mean)
		v *= inv;
	aleatoric *= inv;

	EntropyDecomposition d;
	d.total = predictiveEntropy(mean);
	d.aleatoric = aleatoric;
	d.epistemic = std::max(0.0, d.total - d.aleatoric);
	return d;
}

EntropyDecomposition decomposeEntropy(Ensemble& ensemble, const vector<double>& input) {
	vector<vector<double>> probs;
	probs.reserve(ensemble.size());
	for (uint i = 0; i < ensemble.size(); ++i) {
		vector<double> p = ensemble.mlp(i).propagate(input); // copy: propagate returns a ref
		if (p.size() == 1) {
			const double v = p[0];
			p = {1.0 - v, v};
		}
		probs.push_back(std::move(p));
	}
	return decomposeEntropy(probs);
}

namespace {
// Linear-interpolated percentile of a sorted vector (Roc CI convention).
double pct(const vector<double>& sorted, double q) {
	const double pos = q * (sorted.size() - 1);
	const uint lo = (uint)std::floor(pos);
	const uint hi = (uint)std::ceil(pos);
	const double frac = pos - lo;
	return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}
} // namespace

AdstockSummary summarizeAdstock(Ensemble& ensemble, double alpha) {
	const uint M = ensemble.size();
	if (M == 0) throw std::invalid_argument("summarizeAdstock: empty ensemble");

	const MultiLayerPerceptron::Adstock* first = ensemble.mlp(0).adstock();
	if (!first) throw std::invalid_argument("summarizeAdstock: member 0 has no adstock stage");
	const uint C = first->nChannels();
	const uint L = first->nLags();
	const uint ppc = first->nParamsPerChannel();
	const auto family = first->kernel();

	// Gather per-member kernels and natural params
	vector<vector<vector<double>>> w(M); // [M][C][L]
	vector<vector<double>> nat(M);       // [M][C*ppc]
	for (uint m = 0; m < M; ++m) {
		const MultiLayerPerceptron::Adstock* a = ensemble.mlp(m).adstock();
		if (!a || a->nChannels() != C || a->nLags() != L || a->kernel() != family)
			throw std::invalid_argument("summarizeAdstock: member " + std::to_string(m) +
			                            " has a missing or mismatched adstock stage");
		w[m].resize(C);
		for (uint c = 0; c < C; ++c)
			w[m][c] = a->kernelWeights(c);
		nat[m] = a->naturalParams();
	}

	AdstockSummary s;
	s.channels = C;
	s.lags = L;
	s.paramsPerChannel = ppc;
	s.weightMean.assign(C, vector<double>(L, 0.0));
	s.weightLower.assign(C, vector<double>(L, 0.0));
	s.weightUpper.assign(C, vector<double>(L, 0.0));
	s.paramMean.assign(C * ppc, 0.0);
	s.paramLower.assign(C * ppc, 0.0);
	s.paramUpper.assign(C * ppc, 0.0);

	vector<double> vals(M);
	for (uint c = 0; c < C; ++c)
		for (uint l = 0; l < L; ++l) {
			double mean = 0;
			for (uint m = 0; m < M; ++m) {
				vals[m] = w[m][c][l];
				mean += vals[m];
			}
			std::sort(vals.begin(), vals.end());
			s.weightMean[c][l] = mean / M;
			s.weightLower[c][l] = pct(vals, alpha / 2.0);
			s.weightUpper[c][l] = pct(vals, 1.0 - alpha / 2.0);
		}
	for (uint j = 0; j < C * ppc; ++j) {
		double mean = 0;
		for (uint m = 0; m < M; ++m) {
			vals[m] = nat[m][j];
			mean += vals[m];
		}
		std::sort(vals.begin(), vals.end());
		s.paramMean[j] = mean / M;
		s.paramLower[j] = pct(vals, alpha / 2.0);
		s.paramUpper[j] = pct(vals, 1.0 - alpha / 2.0);
	}
	return s;
}

BoxedAdstockSummary summarizeBoxedAdstock(Ensemble& ensemble, double alpha) {
	using MultiLayerPerceptron::Adstock;
	const uint M = ensemble.size();
	if (M == 0) throw std::invalid_argument("summarizeBoxedAdstock: empty ensemble");

	const Adstock* first = ensemble.mlp(0).adstock();
	if (!first || !first->boxed())
		throw std::invalid_argument("summarizeBoxedAdstock: member 0 has no boxed adstock stage");
	const uint C = first->nChannels();
	const uint L = first->nLags();
	const uint K = first->nBoxes();
	const uint ppb = first->nParamsPerChannel();
	const bool hill = first->saturation() == Adstock::Saturation::Hill;
	const auto family = first->kernel();

	// Per member: canonical box order (by mean carryover lag), then
	// remapped kernels, params, and routing.
	vector<vector<vector<double>>> kern(M); // [M][K][L] canonical
	vector<vector<double>> par(M);          // [M][K*ppb] natural, canonical
	vector<vector<double>> sat(M), expo(M); // [M][K] canonical (hill)
	vector<vector<vector<double>>> rout(M); // [M][C][K] canonical

	for (uint m = 0; m < M; ++m) {
		const Adstock* a = ensemble.mlp(m).adstock();
		if (!a || !a->boxed() || a->nChannels() != C || a->nLags() != L || a->nBoxes() != K ||
		    a->kernel() != family || (a->saturation() == Adstock::Saturation::Hill) != hill)
			throw std::invalid_argument("summarizeBoxedAdstock: member " + std::to_string(m) +
			                            " has a missing or mismatched boxed adstock stage");

		// canonical order: ascending mean lag
		vector<std::pair<double, uint>> order(K);
		vector<vector<double>> w(K);
		for (uint k = 0; k < K; ++k) {
			w[k] = a->boxKernel(k);
			double meanLag = 0;
			for (uint l = 0; l < L; ++l)
				meanLag += l * w[k][l];
			order[k] = {meanLag, k};
		}
		std::sort(order.begin(), order.end());

		const vector<double> nat = a->naturalParams(); // [K*ppb | K sat | K exp]
		kern[m].resize(K);
		par[m].assign(K * ppb, 0.0);
		if (hill) {
			sat[m].assign(K, 0.0);
			expo[m].assign(K, 0.0);
		}
		for (uint k = 0; k < K; ++k) {
			const uint src = order[k].second;
			kern[m][k] = w[src];
			for (uint p = 0; p < ppb; ++p)
				par[m][k * ppb + p] = nat[src * ppb + p];
			if (hill) {
				sat[m][k] = nat[K * ppb + src];
				expo[m][k] = nat[K * ppb + K + src];
			}
		}
		rout[m].resize(C);
		for (uint c = 0; c < C; ++c) {
			const vector<double> pi = a->routingProbs(c);
			rout[m][c].assign(K, 0.0);
			for (uint k = 0; k < K; ++k)
				rout[m][c][k] = pi[order[k].second];
		}
	}

	BoxedAdstockSummary s;
	s.channels = C;
	s.lags = L;
	s.boxes = K;
	s.paramsPerBox = ppb;
	s.hill = hill;

	vector<double> vals(M);
	auto band = [&](auto getter, double& mean, double& lo, double& hi) {
		double acc = 0;
		for (uint m = 0; m < M; ++m) {
			vals[m] = getter(m);
			acc += vals[m];
		}
		std::sort(vals.begin(), vals.end());
		mean = acc / M;
		lo = pct(vals, alpha / 2.0);
		hi = pct(vals, 1.0 - alpha / 2.0);
	};

	s.kernelMean.assign(K, vector<double>(L, 0.0));
	s.kernelLower.assign(K, vector<double>(L, 0.0));
	s.kernelUpper.assign(K, vector<double>(L, 0.0));
	for (uint k = 0; k < K; ++k)
		for (uint l = 0; l < L; ++l)
			band([&](uint m) { return kern[m][k][l]; }, s.kernelMean[k][l], s.kernelLower[k][l],
			     s.kernelUpper[k][l]);

	s.paramMean.assign(K * ppb, 0.0);
	s.paramLower.assign(K * ppb, 0.0);
	s.paramUpper.assign(K * ppb, 0.0);
	for (uint j = 0; j < K * ppb; ++j)
		band([&](uint m) { return par[m][j]; }, s.paramMean[j], s.paramLower[j], s.paramUpper[j]);

	if (hill) {
		s.satMean.assign(K, 0.0);
		s.satLower.assign(K, 0.0);
		s.satUpper.assign(K, 0.0);
		s.expMean.assign(K, 0.0);
		s.expLower.assign(K, 0.0);
		s.expUpper.assign(K, 0.0);
		for (uint k = 0; k < K; ++k) {
			band([&](uint m) { return sat[m][k]; }, s.satMean[k], s.satLower[k], s.satUpper[k]);
			band([&](uint m) { return expo[m][k]; }, s.expMean[k], s.expLower[k], s.expUpper[k]);
		}
	}

	// Routing: mean pi, modal box, and assignment stability
	s.meanRouting.assign(C, vector<double>(K, 0.0));
	s.modalBox.assign(C, 0);
	s.stability.assign(C, 0.0);
	for (uint c = 0; c < C; ++c) {
		vector<uint> votes(K, 0);
		for (uint m = 0; m < M; ++m) {
			uint best = 0;
			for (uint k = 0; k < K; ++k) {
				s.meanRouting[c][k] += rout[m][c][k] / M;
				if (rout[m][c][k] > rout[m][c][best]) best = k;
			}
			++votes[best];
		}
		uint modal = 0;
		for (uint k = 1; k < K; ++k)
			if (votes[k] > votes[modal]) modal = k;
		s.modalBox[c] = modal;
		s.stability[c] = static_cast<double>(votes[modal]) / M;
	}
	return s;
}

} // namespace Uncertainty
} // namespace EvalTools
