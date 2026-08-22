#ifndef __Adstock_hh__
#define __Adstock_hh__

#include <string>
#include <vector>

namespace MultiLayerPerceptron {

using uint = unsigned int;

/**A differentiable, parametric lag-structure input stage (adstock).
 *
 * Motivated by marketing-mix models: an investment today affects the
 * response over many future days. Feeding L raw lags per channel into a
 * dense layer creates L free weights per channel per neuron that must be
 * regularized or pruned; instead this stage collapses each channel's lag
 * window through a normalized parametric kernel with 1-2 free parameters
 * per channel, trained jointly with the network by the existing
 * optimizers.
 *
 * Input layout per pattern (channel-major):
 *   [c0 lag0..lagL-1, c1 lag0..lagL-1, ..., passthrough covariates]
 * where lag0 is "today". Output layout:
 *   [c0 adstocked, c1 adstocked, ..., passthrough copied through]
 *
 * Kernels (weights normalized to sum to 1 over the window; overall scale
 * belongs to the dense layers behind this stage):
 *   Geometric: w_l ~ lambda^l, lambda = sigmoid(rho). 1 param/channel.
 *              Monotone decay from today.
 *   Weibull:   w_l ~ z^(k-1) exp(-z^k), z = (l+1)/s, k = exp(kappa),
 *              s = exp(sigma). 2 params/channel. Allows a delayed peak.
 * All free parameters are unconstrained reals; the positivity/interval
 * constraints live in the transform, so any gradient trainer works.
 */
class Adstock {
  public:
	enum class Kernel { Geometric, Weibull };

	/**\param channels number of media channels (lagged inputs).
	 * \param lags window length L per channel (lag 0 = today).
	 * \param passthrough trailing covariates copied through untouched.
	 * \param k kernel family, shared by all channels (params per channel).
	 */
	Adstock(uint channels, uint lags, uint passthrough, Kernel k = Kernel::Geometric);

	uint nChannels() const { return theChannels; }
	uint nLags() const { return theLags; }
	uint nPassthrough() const { return thePassthrough; }
	Kernel kernel() const { return theKernel; }

	/**Expected raw input dimension: channels*lags + passthrough. */
	uint inputDim() const { return theChannels * theLags + thePassthrough; }
	/**Produced output dimension: channels + passthrough. */
	uint outputDim() const { return theChannels + thePassthrough; }

	uint nParamsPerChannel() const { return theKernel == Kernel::Geometric ? 1u : 2u; }
	uint nParams() const { return theChannels * nParamsPerChannel(); }

	/**Unconstrained trainable parameters, channel-major. */
	std::vector<double>& params() { return theParams; }
	const std::vector<double>& params() const { return theParams; }
	/**Accumulated gradients, same layout as params(). */
	std::vector<double>& gradients() { return theGradients; }
	const std::vector<double>& gradients() const { return theGradients; }

	/**Momentum buffer for SGD-style trainers (mirrors Layer's
	 * weightUpdates). */
	std::vector<double>& paramUpdates() { return theUpdates; }

	void killGradients();
	/**Reset parameters to family defaults (geometric lambda=0.5;
	 * Weibull k=2, scale=L/3). */
	void initParams();

	/**The normalized kernel weights for channel c at the current
	 * parameters (length L). For inspection/reporting. */
	std::vector<double> kernelWeights(uint c) const;

	/**Kernel parameters on their natural scale, channel-major:
	 * geometric decay lambda in (0,1) per channel; Weibull shape k and
	 * scale s (both > 0) per channel. For reporting. */
	std::vector<double> naturalParams() const;

	/**Transform a batch [B x inputDim] row-major; returns pointer to the
	 * internal output buffer [B x outputDim]. Also refreshes the cached
	 * kernels used by accumulateGradients. */
	const double* transformBatch(const double* in, uint B);

	/**The output buffer filled by the last transformBatch call. */
	const std::vector<double>& outputs() const { return theOutputs; }

	/**Transform a single pattern into out (size outputDim). */
	void transform(const double* in, double* out) const;

	/**Accumulate parameter gradients given the raw batch input passed to
	 * transformBatch and the error deltas w.r.t. this stage's outputs
	 * [B x outputDim]. Uses the kernels cached by transformBatch. */
	void accumulateGradients(const double* rawIn, const double* outDelta, uint B);

  private:
	/**Recompute theW [C x L] and theDw [C x ppc x L] from theParams. */
	void computeKernels() const;
	void channelKernel(uint c, double* w, double* dw) const;

	uint theChannels, theLags, thePassthrough;
	Kernel theKernel;
	std::vector<double> theParams;
	std::vector<double> theGradients;
	std::vector<double> theUpdates;
	std::vector<double> theOutputs;
	mutable std::vector<double> theW;  ///< cached kernel weights [C x L]
	mutable std::vector<double> theDw; ///< cached dw/dparam [C x ppc x L]
	mutable std::vector<double> theCachedParams; ///< params the cache was built from
	mutable bool theKernelsFresh = false;
};

/**Round-trip helpers for serialization / config. */
std::string kernelToTag(Adstock::Kernel k);
Adstock::Kernel kernelFromTag(const std::string& tag);

} // namespace MultiLayerPerceptron

#endif
