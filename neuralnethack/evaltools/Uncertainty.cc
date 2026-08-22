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
	vector<vector<vector<double>>> w(M);   // [M][C][L]
	vector<vector<double>> nat(M);         // [M][C*ppc]
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

} // namespace Uncertainty
} // namespace EvalTools
