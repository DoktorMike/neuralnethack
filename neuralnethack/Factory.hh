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

/**Column-to-group mapping for grouped max-abs normalisation
 * (Normaliser::calcAndNormaliseMaxAbs) when an adstock stage is
 * configured: all lag columns of one media channel share one group (a
 * lag window must be scaled by a single factor or the kernel warps near
 * the series edges), each passthrough covariate and each output gets
 * its own group. Inputs only — max-abs leaves the target unscaled.
 * Returns an empty vector when adstock is disabled (plain per-column
 * scaling). Length: channels*lags + passthrough.
 */
std::vector<uint> adstockColumnGroups(const Config& config);

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
