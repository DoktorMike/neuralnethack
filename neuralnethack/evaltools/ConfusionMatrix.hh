#ifndef __ConfusionMatrix_hh__
#define __ConfusionMatrix_hh__

#include "datatools/DataSet.hh"
#include "Ensemble.hh"

#include <ostream>
#include <vector>

namespace EvalTools {

/**A confusion matrix indexed as `m(actual, predicted)`.
 *
 * Works for binary and multi-class problems. For binary, the convention is
 * class 0 = negative, class 1 = positive (matching the existing Evaluator).
 * Build one from raw vectors or directly from an Ensemble.
 */
class ConfusionMatrix {
  public:
	explicit ConfusionMatrix(uint nClasses);

	void add(uint actual, uint predicted);
	void reset();

	uint nClasses() const { return n; }
	uint count(uint actual, uint predicted) const { return m[actual][predicted]; }
	uint total() const;
	uint correct() const;                // diagonal sum
	uint actualTotal(uint cls) const;    // row sum
	uint predictedTotal(uint cls) const; // column sum

	// Binary convenience (asserts nClasses() == 2).
	uint tp() const;
	uint fp() const;
	uint fn() const;
	uint tn() const;

	void print(std::ostream& os) const;

	// ---- Builders ----

	/**Threshold predictions at `cut` (default 0.5) and tally a 2x2 matrix.
	 * `output` is the model's probability for class 1 per pattern.
	 */
	static ConfusionMatrix fromBinary(const std::vector<double>& output,
	                                  const std::vector<uint>& target, double cut = 0.5);

	/**Argmax both predictions and one-hot targets to tally an NxN matrix.
	 * `output[i]` and `target[i]` must each have length == nClasses.
	 */
	static ConfusionMatrix fromMulticlass(const std::vector<std::vector<double>>& output,
	                                      const std::vector<std::vector<double>>& target);

	/**Build directly from an Ensemble + DataSet. Auto-detects binary vs
	 * multi-class from the first prediction's output dimension. For the
	 * binary case, classifies at `cut`.
	 */
	static ConfusionMatrix fromEnsemble(NeuralNetHack::Ensemble& committee,
	                                    DataTools::DataSet& data, double cut = 0.5);

  private:
	uint n;
	std::vector<std::vector<uint>> m; // m[actual][predicted]
};

} // namespace EvalTools

#endif
