#ifndef __Factory_hh__
#define __Factory_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Mlp.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "CrossValidator.hh"
#include "Bootstrapper.hh"
#include "CrossSplitter.hh"
#include "Bagger.hh"

namespace NeuralNetHack
{
	namespace Factory
	{
		MultiLayerPerceptron::Mlp* createMlp(Config& config, DataTools::DataSet& data);
		MultiLayerPerceptron::Error* createError(Config& config, DataTools::DataSet& data);
		MultiLayerPerceptron::Trainer* createTrainer(Config& config, DataTools::DataSet& data);
		EnsembleBuilder* createEnsembleBuilder(Config& config, DataTools::DataSet& data);
		ModelEstimator* createModelEstimator(Config& config, DataTools::DataSet& data);
	}
}

#endif
