#ifndef __Conformal_hh__
#define __Conformal_hh__

#include "Ensemble.hh"
#include "datatools/DataSet.hh"

#include <vector>

namespace EvalTools {

/**Split conformal prediction on top of a NeuralNetHack::Ensemble.
 *
 * Two modes:
 *   - Regression: per-dim residual score \f$|t_d - \hat y_d|\f$. Each
 *     output dim gets its own quantile \f$\hat q_d\f$. Marginal coverage
 *     \f$1-\alpha\f$ per dim under exchangeability.
 *   - Classification (LAC, "least ambiguous classifier"): non-conformity
 *     score \f$1 - \hat p_{y}\f$ where \f$y\f$ is the true class. Single
 *     \f$\hat q\f$. Prediction set \f$\{k : \hat p_k \ge 1 - \hat q\}\f$.
 *     Marginal coverage \f$1-\alpha\f$.
 *
 * Caller owns the train/calibration split; pass the held-out calibration
 * DataSet to calibrate(). Use DataTools::HoldOutSampler if you want the
 * standard random split.
 *
 * Classification handles both single-output (sigmoid, treated as 2-class
 * with probs \f$[1-y, y]\f$) and multi-output softmax (one-hot target).
 */
class Conformal {
  public:
	enum class Mode { Regression, Classification };

	struct Interval {
		double lo;
		double hi;
	};

	/**Construct an uncalibrated conformal predictor.
	 * \param mode regression or classification.
	 * \param alpha miscoverage rate in (0,1). 0.1 → 90% target coverage.
	 */
	Conformal(Mode mode, double alpha);

	/**Compute non-conformity scores on the calibration set and store the
	 * \f$1-\alpha\f$ finite-sample-corrected quantile. Must be called
	 * before any predict / coverage call. Sets the output dimension from
	 * the ensemble's first prediction.
	 */
	void calibrate(NeuralNetHack::Ensemble& e, DataTools::DataSet& cal);

	/**Regression: per-dim conformal interval around the ensemble mean
	 * prediction, length nOutput. Throws on mode mismatch or if not
	 * calibrated.
	 */
	std::vector<Interval> interval(NeuralNetHack::Ensemble& e, const std::vector<double>& x) const;

	/**Classification: prediction set as sorted class indices. Throws on
	 * mode mismatch or if not calibrated. May be empty in pathological
	 * cases (model is overconfident on every wrong class for this point).
	 */
	std::vector<uint> set(NeuralNetHack::Ensemble& e, const std::vector<double>& x) const;

	/**Marginal empirical coverage on a held-out test DataSet.
	 * Regression: fraction over (pattern, dim) pairs.
	 * Classification: fraction of patterns whose true class is in the set.
	 * Sanity check; returns NaN on empty input.
	 */
	double coverage(NeuralNetHack::Ensemble& e, DataTools::DataSet& tst) const;

	/**Per-dim regression quantiles (length nOutput) or single-element
	 * vector with the LAC quantile. Empty before calibrate().
	 */
	const std::vector<double>& quantiles() const { return theQ; }

	Mode mode() const { return theMode; }
	double alpha() const { return theAlpha; }
	bool calibrated() const { return theCalibrated; }

  private:
	Mode theMode;
	double theAlpha;
	uint theNOutput = 0;
	std::vector<double> theQ;
	bool theCalibrated = false;
};

} // namespace EvalTools
#endif
