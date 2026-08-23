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
 * window through a normalized parametric kernel with few free
 * parameters, trained jointly with the network by the existing
 * optimizers.
 *
 * Input layout per pattern (channel-major):
 *   [c0 lag0..lagL-1, c1 lag0..lagL-1, ..., passthrough covariates]
 * where lag0 is "today". Output layout:
 *   [c0 transformed, c1 transformed, ..., passthrough copied through]
 *
 * Two modes:
 *
 * **Per-channel** (nBoxes == 0): every channel has its own kernel
 * parameters (1-2 per channel). Right when C is small.
 *
 * **Boxed** (nBoxes == K > 0): K shared kernel "boxes" (carryover
 * regimes: short / medium / long) plus per-channel routing
 * pi_c = softmax(logits_c / tau) that mixes the boxes:
 *   kernel_c = sum_k pi_ck w(theta_k)
 *   a_c      = kernel_c . x_c
 *   out_c    = sum_k pi_ck hill(a_c; s_k, n_k)   (Saturation::Hill)
 *            = a_c                               (Saturation::None)
 * One routing gates both carryover and saturation: a channel is one
 * insertion type end to end. C media channels collapse to K effective
 * ones; each box's parameters pool ~C/K channels' worth of data. An
 * optional entropy penalty pushes the routing toward one-hot. See
 * doc/spec-boxed-adstock.md.
 *
 * Kernels (weights normalized to sum to 1 over the window; overall scale
 * belongs to the dense layers behind this stage):
 *   Geometric: w_l ~ lambda^l, lambda = sigmoid(rho). 1 param.
 *              Monotone decay from today.
 *   Weibull:   w_l ~ z^(k-1) exp(-z^k), z = (l+1)/s, k = exp(kappa),
 *              s = exp(sigma). 2 params. Allows a delayed peak.
 * Saturation (boxed mode only): hill(a; s, n) = a^n / (a^n + s^n) with
 * s = exp(sigma), n bounded to [0.5, 3] via sigmoid; n starts at 1 (plain diminishing
 * returns), n > 1 learns an S-shaped response.
 * All free parameters are unconstrained reals; the positivity/interval
 * constraints live in the transform, so any gradient trainer works.
 */
class Adstock {
  public:
	enum class Kernel { Geometric, Weibull };
	enum class Saturation { None, Hill };

	/**Per-channel mode: one kernel per channel.
	 * \param channels number of media channels (lagged inputs).
	 * \param lags window length L per channel (lag 0 = today).
	 * \param passthrough trailing covariates copied through untouched.
	 * \param k kernel family, shared by all channels (params per channel).
	 */
	Adstock(uint channels, uint lags, uint passthrough, Kernel k = Kernel::Geometric);

	/**Boxed mode: K shared kernel boxes with learned per-channel routing,
	 * optionally gating a per-box Hill saturation with the same routing.
	 * \param channels number of media channels.
	 * \param lags window length L per channel.
	 * \param passthrough trailing covariates copied through untouched.
	 * \param k kernel family shared by all boxes.
	 * \param nBoxes number of boxes K (>= 1).
	 * \param sat per-box saturation applied after the adstock dot.
	 */
	Adstock(uint channels, uint lags, uint passthrough, Kernel k, uint nBoxes,
	        Saturation sat = Saturation::None);

	uint nChannels() const { return theChannels; }
	uint nLags() const { return theLags; }
	uint nPassthrough() const { return thePassthrough; }
	Kernel kernel() const { return theKernel; }
	uint nBoxes() const { return theNBoxes; }
	bool boxed() const { return theNBoxes > 0; }
	Saturation saturation() const { return theSaturation; }

	/**Softmax temperature for the routing (boxed mode; default 1).
	 * Lower it during training to harden assignments. */
	double temperature() const { return theTau; }
	void temperature(double tau) { theTau = tau; }

	/**Entropy-penalty coefficient beta (boxed mode; default 0 = off).
	 * The loss gains beta * sum_c H(pi_c), pushing routing toward
	 * one-hot. Applied by the Error classes alongside weight
	 * elimination.
	 *
	 * Schedule it: train with beta = 0 until the routing stabilizes,
	 * THEN enable the penalty to harden the assignments. Enabling it
	 * from the first epoch hardens the routing toward whichever vertex
	 * is nearest before the boxes have separated, locking in
	 * chance-level assignments. */
	double entropyPenalty() const { return theEntropyBeta; }
	void entropyPenalty(double beta) { theEntropyBeta = beta; }

	/**Expected raw input dimension: channels*lags + passthrough. */
	uint inputDim() const { return theChannels * theLags + thePassthrough; }
	/**Produced output dimension: channels + passthrough. */
	uint outputDim() const { return theChannels + thePassthrough; }

	/**Kernel-family parameters per kernel: 1 (geometric) or 2 (Weibull). */
	uint nParamsPerChannel() const { return theKernel == Kernel::Geometric ? 1u : 2u; }
	uint nParams() const;

	/**Unconstrained trainable parameters. Per-channel mode:
	 * channel-major kernel params. Boxed mode layout:
	 * [box kernel params (K*ppk) | hill sigmas (K) | hill nus (K) |
	 *  routing logits (C*K, channel-major)] (hill blocks only with
	 * Saturation::Hill). */
	std::vector<double>& params() { return theParams; }
	const std::vector<double>& params() const { return theParams; }
	/**Accumulated gradients, same layout as params(). */
	std::vector<double>& gradients() { return theGradients; }
	const std::vector<double>& gradients() const { return theGradients; }

	/**Momentum buffer for SGD-style trainers (mirrors Layer's
	 * weightUpdates). */
	std::vector<double>& paramUpdates() { return theUpdates; }

	void killGradients();
	/**Reset parameters to family defaults. Per-channel: geometric
	 * lambda=0.5, Weibull k=2/scale=L/3. Boxed: box kernels staggered
	 * from fast to slow decay (symmetry breaking), hill s=1/n=1,
	 * logits 0 (uniform routing). */
	void initParams();

	/**The normalized kernel weights for channel c at the current
	 * parameters (length L); in boxed mode the routing-mixed kernel.
	 * For inspection/reporting. */
	std::vector<double> kernelWeights(uint c) const;

	/**Kernel parameters on their natural scale. Per-channel mode:
	 * channel-major (lambda, or k and s). Boxed mode: box-major kernel
	 * params, then hill half-saturations (K), then hill exponents (K)
	 * when saturation is on. Routing logits are not included. */
	std::vector<double> naturalParams() const;

	// Boxed-mode reporting ---------------------------------------------------
	/**Routing probabilities pi_c for channel c (length K). */
	std::vector<double> routingProbs(uint c) const;
	/**argmax_k pi_ck per channel. */
	std::vector<uint> boxAssignments() const;
	/**Normalized kernel of box k (length L). */
	std::vector<double> boxKernel(uint k) const;
	/**Hill half-saturation of box k, natural scale (Saturation::Hill). */
	double boxSaturation(uint k) const;
	/**Hill exponent of box k, natural scale (Saturation::Hill). */
	double boxHillExponent(uint k) const;

	/**Transform a batch [B x inputDim] row-major; returns pointer to the
	 * internal output buffer [B x outputDim]. Also refreshes the caches
	 * used by accumulateGradients. */
	const double* transformBatch(const double* in, uint B);

	/**The output buffer filled by the last transformBatch call. */
	const std::vector<double>& outputs() const { return theOutputs; }

	/**Transform a single pattern into out (size outputDim). */
	void transform(const double* in, double* out) const;

	/**Accumulate parameter gradients given the raw batch input passed to
	 * transformBatch and the error deltas w.r.t. this stage's outputs
	 * [B x outputDim]. Uses the caches from transformBatch. */
	void accumulateGradients(const double* rawIn, const double* outDelta, uint B);

	/**Add the entropy-penalty gradient beta * d(sum_c H(pi_c))/dlogits to
	 * the routing-logit gradients. Called by the Error classes after the
	 * batch normalisation divide (weight-elimination precedent). No-op
	 * unless boxed and beta > 0. */
	void applyEntropyPenaltyGradient();

  private:
	/**Recompute cached kernels (and routing in boxed mode) from
	 * theParams. */
	void computeKernels() const;
	/**Normalized kernel + d/dparam from one param set p (length ppk). */
	void kernelFromParams(const double* p, double* w, double* dw) const;
	void softmaxRouting(uint c, double* pi) const;

	/**Hill exponent bounds (natural scale): the trainable exponent is
	 * n = HILL_EXP_MIN + (HILL_EXP_MAX - HILL_EXP_MIN) * sigmoid(nu).
	 * Bounded so the response cannot degenerate to a step function on
	 * flighted data (see Adstock.cc). */
	static constexpr double HILL_EXP_MIN = 0.5;
	static constexpr double HILL_EXP_MAX = 3.0;
	static double hillExpFromRaw(double nu);
	static double hillExpGradFactor(double nu);
	static double hillExpToRaw(double n);

	// Hill helpers (natural-scale s, n); value and partials at a >= 0.
	static double hill(double a, double s, double n);
	static void hillPartials(double a, double s, double n, double& dha, double& dhs, double& dhn);

	uint theChannels, theLags, thePassthrough;
	Kernel theKernel;
	uint theNBoxes;            ///< 0 = per-channel mode
	Saturation theSaturation;  ///< boxed mode only
	double theTau = 1.0;       ///< routing temperature
	double theEntropyBeta = 0; ///< entropy-penalty coefficient

	std::vector<double> theParams;
	std::vector<double> theGradients;
	std::vector<double> theUpdates;
	std::vector<double> theOutputs;

	// caches rebuilt when params change
	mutable std::vector<double> theW;  ///< kernels: per-channel [C x L] or boxed [K x L]
	mutable std::vector<double> theDw; ///< d kernel / d param, matching theW blocks
	mutable std::vector<double> thePi; ///< boxed: routing [C x K]
	mutable std::vector<double> theCachedParams;
	mutable bool theKernelsFresh = false;

	// batch caches from transformBatch, used by accumulateGradients
	std::vector<double> theA;   ///< boxed: adstocked dot a_c [B x C]
	std::vector<double> theDot; ///< boxed: per-box dots w_k.x_c [B x C x K]

	// param-layout offsets (boxed mode)
	uint kernOff() const { return 0; }
	uint sigOff() const { return theNBoxes * nParamsPerChannel(); }
	uint nuOff() const { return sigOff() + theNBoxes; }
	uint logitOff() const {
		return theNBoxes * nParamsPerChannel() +
		       (theSaturation == Saturation::Hill ? 2 * theNBoxes : 0);
	}
};

/**Round-trip helpers for serialization / config. */
std::string kernelToTag(Adstock::Kernel k);
Adstock::Kernel kernelFromTag(const std::string& tag);

} // namespace MultiLayerPerceptron

#endif
