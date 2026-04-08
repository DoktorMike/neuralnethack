#ifndef __ModelEstimator_hh__
#define __ModelEstimator_hh__

#include "EnsembleBuilder.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"

#include <vector>
#include <utility>
#include <iostream>

namespace NeuralNetHack {
/**A base class representing the different model estimation methods available.
 * \sa CrossValidator, Bootstrapper.
 */
class ModelEstimator {
  public:
	/**Basic constructor. */
	ModelEstimator();

	/**Constructor taking the mandatory arguments for this class to
	 * function properly. This is probably what you want to use.
	 * \param eb the EnsembleBuilder to use to build the Ensemble.
	 * \param s the Sampler to use to sample the DataSet.
	 */
	ModelEstimator(EnsembleBuilder& eb, DataTools::Sampler& s);

	/**Basic destructor. */
	virtual ~ModelEstimator();

	/**The pointer to the pair of estimation values. The first element
	 * of the pair is the estimation value for the training run and
	 * the second is for the validation.
	 */
	std::pair<double, double>* estimateModel(double (*errorFunc)(Ensemble& committee,
	                                                             DataTools::DataSet& data));

	/**Virtual method that will estimate the model given the DataSet.
	 * \return the estimation value of the training and the validation.
	 */
	std::pair<double, double>* runAndEstimateModel(double (*errorFunc)(Ensemble& committee,
	                                                                   DataTools::DataSet& data));

	/**Accessor for the EnsembleBuilder pointer.
	 * \return the pointer to the EnsembleBuilder.
	 */
	EnsembleBuilder* ensembleBuilder();

	/**Mutator for the EnsembleBuilder pointer.
	 * \param eb the pointer to the EnsembleBuilder to set.
	 */
	void ensembleBuilder(EnsembleBuilder* eb);

	/**Accessor for the Sampler pointer.
	 * \return the pointer to the Sampler.
	 */
	DataTools::Sampler* sampler();

	/**Mutator for the Data pointer.
	 * \param s the pointer to the Data to set.
	 */
	void sampler(DataTools::Sampler* s);

	/**Accessor for the Session vector.
	 * \return the Session vector.
	 */
	std::vector<Session>& sessions();

	/**Mutator for the Session vector.
	 * \param s the Session vector to set.
	 */
	void sessions(std::vector<Session>& s);

  private:
	/**Pointer to the EnsembleBuilder we will use for estimating. */
	EnsembleBuilder* theEnsembleBuilder;

	/**Pointer to the Sampler we will use to estimate the model with. */
	DataTools::Sampler* theSampler;

	/**A vector of Estimations. */
	std::vector<Session> theSessions;
};
} // namespace NeuralNetHack
#endif
