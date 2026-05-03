#ifndef __Factory_hh__
#define __Factory_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/Mlp.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "ModelEstimator.hh"

#include <memory>

namespace NeuralNetHack {
/**Factory functions for the standard NeuralNetHack object graph. Every
 * factory returns a `unique_ptr` so the caller's lifetime expectations are
 * explicit; the trainer owns its error, the error owns its mlp.
 */
namespace Factory {

std::unique_ptr<MultiLayerPerceptron::Mlp> createMlp(const Config& config);

std::unique_ptr<MultiLayerPerceptron::Error> createError(const Config& config,
                                                         DataTools::DataSet& data);

std::unique_ptr<MultiLayerPerceptron::Trainer> createTrainer(const Config& config,
                                                             DataTools::DataSet& data);

std::unique_ptr<DataTools::Sampler> createSampler(const Config& config, DataTools::DataSet& data);

std::unique_ptr<EnsembleBuilder> createEnsembleBuilder(const Config& config,
                                                       DataTools::DataSet& data);

std::unique_ptr<ModelEstimator> createModelEstimator(const Config& config,
                                                     DataTools::DataSet& data);

} // namespace Factory
} // namespace NeuralNetHack

#endif
