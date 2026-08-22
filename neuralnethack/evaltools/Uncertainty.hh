#ifndef __Uncertainty_hh__
#define __Uncertainty_hh__

#include "../Ensemble.hh"

#include <vector>

namespace EvalTools {

/**Predictive-uncertainty utilities for ensembles of classifiers.
 *
 * The information-theoretic decomposition follows Depeweg et al. (2018):
 * given an ensemble whose members each output a categorical distribution
 * \f$p_i\f$ over the same K classes, the entropy of the ensemble-mean
 * prediction splits into an aleatoric and an epistemic part,
 * \f[ \underbrace{H(\bar p)}_{\text{total}}
 *     = \underbrace{\tfrac1M\sum_i H(p_i)}_{\text{aleatoric}}
 *     + \underbrace{I(y; i)}_{\text{epistemic}}, \f]
 * where the epistemic term is the mutual information between the class and
 * the member identity (a.k.a. BALD). Aleatoric uncertainty reflects genuine
 * class overlap (irreducible noise); epistemic uncertainty reflects member
 * disagreement and grows out of distribution.
 */
namespace Uncertainty {

/**Result of decomposing an ensemble's predictive entropy. */
struct EntropyDecomposition {
	double total;     ///< H(mean prediction)
	double aleatoric; ///< mean of per-member entropies
	double epistemic; ///< total - aleatoric (clamped at 0); mutual information
};

/**Shannon entropy (natural log) of a probability vector. Entries <= 0 are
 * skipped, so an unnormalised or sparse vector is handled gracefully.
 * \param p a categorical distribution.
 * \return the entropy in nats.
 */
double predictiveEntropy(const std::vector<double>& p);

/**Decompose the predictive entropy over a set of per-member probability
 * vectors. Members are weighted uniformly (as in the cited literature).
 * \param memberProbs one categorical distribution per member, all of equal
 * length K.
 * \return the total / aleatoric / epistemic decomposition.
 */
EntropyDecomposition decomposeEntropy(const std::vector<std::vector<double>>& memberProbs);

/**Convenience overload: propagate the input through every ensemble member,
 * collect each member's predictive distribution, and decompose. A
 * single-output (sigmoid) member with value p is treated as the categorical
 * {1 - p, p}. Members are weighted uniformly; ensemble scales are ignored,
 * matching the uncertainty-decomposition literature.
 * \param ensemble the ensemble to query.
 * \param input the input pattern.
 * \return the total / aleatoric / epistemic decomposition.
 */
EntropyDecomposition decomposeEntropy(NeuralNetHack::Ensemble& ensemble,
                                      const std::vector<double>& input);

/**Ensemble summary of fitted adstock (carryover) kernels.
 *
 * Every ensemble member trained on a resample carries its own fitted
 * kernel parameters, so the spread across members is a bootstrap-style
 * uncertainty estimate for the carryover structure itself -- the part of
 * a marketing-mix readout people actually argue about.
 */
struct AdstockSummary {
	uint channels;         ///< media channels C
	uint lags;             ///< window length L
	uint paramsPerChannel; ///< 1 (geometric) or 2 (Weibull)

	/**Per-channel per-lag kernel weights across members: mean and the
	 * alpha/2, 1-alpha/2 percentile band. Each is [C][L]. */
	std::vector<std::vector<double>> weightMean, weightLower, weightUpper;

	/**Natural-scale kernel parameters (lambda, or k and s), channel-major
	 * [C * paramsPerChannel]: member mean and percentile band. */
	std::vector<double> paramMean, paramLower, paramUpper;
};

/**Summarize the adstock kernels across an ensemble's members with
 * percentile intervals (linear-interpolated quantiles, matching the ROC
 * bootstrap CI convention). Members are weighted uniformly.
 * \param ensemble ensemble whose members all carry an adstock stage of
 * identical shape and kernel family (throws std::invalid_argument
 * otherwise, or when the ensemble is empty).
 * \param alpha miscoverage: the band is [alpha/2, 1-alpha/2] (default 0.1
 * = an 90% interval).
 * \return per-channel kernel-weight bands and natural-parameter bands.
 */
AdstockSummary summarizeAdstock(NeuralNetHack::Ensemble& ensemble, double alpha = 0.1);

} // namespace Uncertainty
} // namespace EvalTools

#endif
