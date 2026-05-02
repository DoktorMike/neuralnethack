#include "Factory.hh"
#include "datatools/BootstrapSampler.hh"
#include "datatools/CrossSplitSampler.hh"
#include "datatools/DummySampler.hh"
#include "datatools/HoldOutSampler.hh"

#include <memory>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using DataTools::DataSet;
using DataTools::Sampler;
using std::make_unique;
using std::unique_ptr;

namespace {

unique_ptr<Sampler> buildEnsembleSampler(const Config& config, DataSet& data) {
	const std::string& method = config.ensParamDataSelection();
	unique_ptr<Sampler> sampler;
	if (method == "cs")
		sampler = make_unique<DataTools::CrossSplitSampler>(data, config.ensParamN(),
		                                                   config.ensParamK());
	else if (method == "bagg")
		sampler = make_unique<DataTools::BootstrapSampler>(data, config.ensParamN());
	else if (method == "hold")
		sampler = make_unique<DataTools::HoldOutSampler>(data, 1.0, config.ensParamN());
	else if (method == "none" || method == "dummy")
		sampler = make_unique<DataTools::DummySampler>(data, config.ensParamN());
	if (sampler) sampler->randomSampling(config.ensParamSplitMode());
	return sampler;
}

unique_ptr<Sampler> buildModelSelectionSampler(const Config& config, DataSet& data) {
	const std::string& method = config.msParamDataSelection();
	unique_ptr<Sampler> sampler;
	if (method == "cv")
		sampler = make_unique<DataTools::CrossSplitSampler>(data, config.msParamN(),
		                                                   config.msParamK());
	else if (method == "boot")
		sampler = make_unique<DataTools::BootstrapSampler>(data, config.msParamN());
	else if (method == "hold")
		sampler = make_unique<DataTools::HoldOutSampler>(data, config.msParamNumTrainingData(),
		                                                config.msParamN());
	else if (method == "none" || method == "dummy")
		sampler = make_unique<DataTools::DummySampler>(data, config.msParamN());
	if (sampler) sampler->randomSampling(config.msParamSplitMode());
	return sampler;
}

} // namespace

unique_ptr<Mlp> Factory::createMlp(const Config& config) {
	auto mlp = make_unique<Mlp>(config.architecture(), config.actFcn(), config.softmax());
	for (const auto& sc : config.skipConnections())
		mlp->skipFrom(static_cast<uint>(sc.first), sc.second);
	return mlp;
}

unique_ptr<Error> Factory::createError(const Config& config, DataSet& data) {
	auto mlp = createMlp(config);
	unique_ptr<Error> error;
	if (config.errFcn() == SSE)
		error = make_unique<SummedSquare>(std::move(mlp), data);
	else if (config.errFcn() == CEE)
		error = make_unique<CrossEntropy>(std::move(mlp), data);
	if (error) {
		error->weightElimOn(config.weightElimOn());
		error->weightElimAlpha(config.weightElimAlpha());
		error->weightElimW0(config.weightElimW0());
	}
	return error;
}

unique_ptr<Trainer> Factory::createTrainer(const Config& config, DataSet& data) {
	auto error = createError(config, data);
	unique_ptr<Trainer> trainer;
	if (config.minMethod() == GD) {
		trainer = make_unique<GradientDescent>(std::move(error), data, MAX_ERROR,
		                                       config.batchSize(), config.learningRate(),
		                                       config.decLearningRate(), config.momentum());
	} else if (config.minMethod() == ADAM) {
		trainer = make_unique<Adam>(std::move(error), data, MAX_ERROR, config.batchSize(),
		                            config.adamLearningRate(), config.adamBeta1(),
		                            config.adamBeta2(), config.adamEpsilon(),
		                            config.adamWeightDecay());
	} else {
		trainer = make_unique<QuasiNewton>(std::move(error), data, MAX_ERROR, config.batchSize());
	}
	trainer->numEpochs(config.maxEpochs());
	return trainer;
}

unique_ptr<Sampler> Factory::createSampler(const Config& config, DataSet& data) {
	return buildEnsembleSampler(config, data);
}

unique_ptr<EnsembleBuilder> Factory::createEnsembleBuilder(const Config& config, DataSet& data) {
	auto eb = make_unique<EnsembleBuilder>();
	eb->trainer(createTrainer(config, data));
	eb->sampler(buildEnsembleSampler(config, data));
	// Closure: each parallel worker builds its own trainer (and the
	// transitive Error / Mlp) from the same Config. Captures Config by
	// value so the lambda is safe regardless of the caller's lifetime.
	eb->trainerFactory([config](DataSet& d) { return Factory::createTrainer(config, d); });
	eb->baseSeed(config.seed() == 0 ? 1 : static_cast<uint64_t>(config.seed()));
	return eb;
}

unique_ptr<ModelEstimator> Factory::createModelEstimator(const Config& config, DataSet& data) {
	auto me = make_unique<ModelEstimator>();
	me->sampler(buildModelSelectionSampler(config, data));
	me->ensembleBuilder(createEnsembleBuilder(config, data));
	return me;
}
