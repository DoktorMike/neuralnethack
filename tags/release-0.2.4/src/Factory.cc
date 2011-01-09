#include "Factory.hh"

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;

/**\todo Fix so that softmax gets forwarded ok. */
Mlp* Factory::createMlp(Config& config, DataSet& data)
{
	return new Mlp(config.arch, config.actFcn, false);
}

Error* Factory::createError(Config& config, DataSet& data)
{
	Error* error = 0;
	if(config.errFcn == SSE)
		error = new SummedSquare();
	else if(config.errFcn == CEE)
		error = new CrossEntropy();
	error->mlp(createMlp(config, data));
	error->weightElimOn(config.weightElimOn);
	error->weightElimAlpha(config.weightElimAlpha);
	error->weightElimW0(config.weightElimW0);
	return error;
}

Trainer* Factory::createTrainer(Config& config, DataSet& data)
{
	Trainer* trainer = 0;
	Error* error = createError(config, data);
	if(config.minMethod == GD){
		trainer = new GradientDescent(
				*(error->mlp()), data, *error,
				MAX_ERROR, config.batchSize,
				config.learningRate, config.decLearningRate, config.momentum);
	}else{
		trainer = new QuasiNewton(*(error->mlp()), data, *error, MAX_ERROR, config.batchSize);
	}
	trainer->numEpochs(config.maxEpochs);

	return trainer;
}

EnsembleBuilder* Factory::createEnsembleBuilder(Config& config, DataSet& data)
{
	EnsembleBuilder* eb = 0;
	if(config.ensParamDataSelection == "cs"){
		CrossSplitter* cs = new CrossSplitter();
		cs->numRuns(config.ensParamN);
		cs->numParts(config.ensParamK);
		eb = cs;
	}else if(config.ensParamDataSelection == "bagg"){
		Bagger* bagger = new Bagger();
		bagger->numRuns(config.ensParamN);
		eb = bagger;
	}
	eb->trainer(createTrainer(config, data));
	eb->randomSampling(config.ensParamSplitMode);
	eb->data(&data);
	return eb;
}

ModelEstimator* Factory::createModelEstimator(Config& config, DataSet& data)
{
	ModelEstimator* me = 0;
	if(config.msParamDataSelection == "cv"){
		CrossValidator* cv = new CrossValidator();
		cv->numRuns(config.msParamN);
		cv->numParts(config.msParamK);
		me = cv;
	}else if(config.msParamDataSelection == "boot"){
		Bootstrapper* bs = new Bootstrapper();
		bs->numRuns(config.msParamN);
		me = bs;
	}
	me->ensembleBuilder(createEnsembleBuilder(config, data));
	//me->randomSampling(config.msParamSplitMode);
	me->data(&data);
	return me;
}

