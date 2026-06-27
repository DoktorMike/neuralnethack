#include "Uncertainty.hh"
#include "../mlp/Mlp.hh"

#include <algorithm>
#include <cassert>
#include <cmath>

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

} // namespace Uncertainty
} // namespace EvalTools
