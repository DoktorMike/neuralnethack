#ifndef __ModelSelector_hh__
#define __ModelSelector_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"

#include <vector>
#include <utility>

namespace NeuralNetHack {
/**Selecting the optimal model by performing a grid search over the chosen
 * parameters.
 */
class ModelSelector {
  public:
	ModelSelector();
	virtual ~ModelSelector();
	std::pair<Config, double> findBestModel(DataTools::DataSet& trnData, Config& config);

  private:
	/** Method for constructing a full sequence from a vector seq.
	 * \param seq vector containing the start stop and increment value.
	 */
	std::vector<double> sequence(const std::vector<double>& seq);

	std::pair<double, double>* trainAndValidateModel(DataTools::DataSet& trnData,
	                                                 const Config& config);

	/** Calculate the .632+ rule.
	 * \param gamma The no-information error rate. For AUC it's 0.5.
	 * \param meanTrn The mean training performance over the bootstrap samples.
	 * \param meanTst The mean testing performance over the bootstrap samples.
	 */
	double Auc632PlusRule(double meanTrn, double meanTst);
};
} // namespace NeuralNetHack
#endif
