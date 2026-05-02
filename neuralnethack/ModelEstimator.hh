#ifndef __ModelEstimator_hh__
#define __ModelEstimator_hh__

#include "EnsembleBuilder.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace NeuralNetHack {
/**A base class representing the different model estimation methods available.
 * \sa CrossValidator, Bootstrapper.
 */
class ModelEstimator {
  public:
	ModelEstimator();
	ModelEstimator(const ModelEstimator&) = delete;
	ModelEstimator& operator=(const ModelEstimator&) = delete;
	ModelEstimator(ModelEstimator&&) noexcept = default;
	ModelEstimator& operator=(ModelEstimator&&) noexcept = default;
	virtual ~ModelEstimator();

	std::pair<double, double>* estimateModel(double (*errorFunc)(Ensemble& committee,
	                                                             DataTools::DataSet& data));

	std::pair<double, double>* runAndEstimateModel(double (*errorFunc)(Ensemble& committee,
	                                                                   DataTools::DataSet& data));

	/**Accessor for the EnsembleBuilder (read-only). */
	EnsembleBuilder* ensembleBuilder();

	/**Take ownership of an EnsembleBuilder. */
	void ensembleBuilder(std::unique_ptr<EnsembleBuilder> eb);

	/**Accessor for the Sampler (read-only). */
	DataTools::Sampler* sampler();

	/**Take ownership of a Sampler. */
	void sampler(std::unique_ptr<DataTools::Sampler> s);

	/**Accessor for the Session vector. */
	std::vector<Session>& sessions();

	/**Mutator for the Session vector. */
	void sessions(std::vector<Session>& s);

  private:
	std::unique_ptr<EnsembleBuilder> theEnsembleBuilder;
	std::unique_ptr<DataTools::Sampler> theSampler;
	std::vector<Session> theSessions;
};
} // namespace NeuralNetHack
#endif
