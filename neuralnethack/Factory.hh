#ifndef __Factory_hh__
#define __Factory_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Mlp.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "ModelEstimator.hh"

namespace NeuralNetHack {
/**This namespace encloses a number of functions for creating various
 * objects needed in the NeuralNetHack project. Most of these functions
 * need the configuration class Config. Note that the destruction of the
 * created objects are left to the user.
 * \sa Config.
 */
namespace Factory {
/**Creates an Mlp object.
 * \param config the config options.
 * \return the created Mlp.
 */
MultiLayerPerceptron::Mlp* createMlp(const Config& config);

/**Creates an Error object.
 * \param config the config options.
 * \param data the DataSet to give the Error.
 * \return the created Error.
 */
MultiLayerPerceptron::Error* createError(const Config& config, DataTools::DataSet& data);

/**Creates a Trainer object.
 * \param config the config options.
 * \param data the DataSet to give the Trainer.
 * \return the created Trainer.
 */
MultiLayerPerceptron::Trainer* createTrainer(const Config& config, DataTools::DataSet& data);

/**Creates a Sampler object.
 * \param config the config options.
 * \param data the DataSet to give the Sampler.
 * \return the created Sampler.
 */
DataTools::Sampler* createSampler(const Config& config, DataTools::DataSet& data);

/**Creates an EnsembleBuilder object.
 * \param config the config options.
 * \param data the DataSet to give the EnsembleBuilder.
 * \return the created EnsembleBuilder.
 */
EnsembleBuilder* createEnsembleBuilder(const Config& config, DataTools::DataSet& data);

/**Creates a ModelEstimator object.
 * \param config the config options.
 * \param data the DataSet to give the ModelEstimator.
 * \return the created ModelEstimator.
 */
ModelEstimator* createModelEstimator(const Config& config, DataTools::DataSet& data);
} // namespace Factory
} // namespace NeuralNetHack

#endif
