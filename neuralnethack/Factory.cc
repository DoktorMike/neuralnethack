/*$Id: Factory.cc 1604 2007-01-24 18:26:14Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#include "Factory.hh"
#include "datatools/BootstrapSampler.hh"
#include "datatools/CrossSplitSampler.hh"
#include "datatools/HoldOutSampler.hh"
#include "datatools/DummySampler.hh"

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using DataTools::DataSet;

/**\todo Fix so that softmax gets forwarded ok. */
Mlp* Factory::createMlp(const Config& config)
{
	return new Mlp(config.architecture(), config.actFcn(), false);
}

Error* Factory::createError(const Config& config, DataSet& data)
{
	Error* error = 0;
	if(config.errFcn() == SSE)
		error = new SummedSquare();
	else if(config.errFcn() == CEE)
		error = new CrossEntropy();
	error->mlp(createMlp(config));
	error->weightElimOn(config.weightElimOn());
	error->weightElimAlpha(config.weightElimAlpha());
	error->weightElimW0(config.weightElimW0());
	return error;
}

Trainer* Factory::createTrainer(const Config& config, DataSet& data)
{
	Trainer* trainer = 0;
	Error* error = createError(config, data);
	if(config.minMethod() == GD){
		trainer = new GradientDescent(
				*(error->mlp()), data, *error,
				MAX_ERROR, config.batchSize(),
				config.learningRate(), config.decLearningRate(), config.momentum());
	}else{
		trainer = new QuasiNewton(*(error->mlp()), data, *error, MAX_ERROR, config.batchSize());
	}
	trainer->numEpochs(config.maxEpochs());

	return trainer;
}

DataTools::Sampler* Factory::createSampler(const Config& config, DataSet& data)
{
	return 0;
}

EnsembleBuilder* Factory::createEnsembleBuilder(const Config& config, DataSet& data)
{
	DataTools::Sampler* sampler = 0;
 	if(config.ensParamDataSelection() == "cs"){
		sampler = new DataTools::CrossSplitSampler(data, config.ensParamN(), config.ensParamK());
	}else if(config.ensParamDataSelection() == "bagg"){
		sampler = new DataTools::BootstrapSampler(data, config.ensParamN());
	}else if(config.ensParamDataSelection() == "none" || config.ensParamDataSelection() == "dummy"){
		sampler = new DataTools::DummySampler(data, config.ensParamN());
	}else if(config.ensParamDataSelection() == "hold"){
		sampler = new DataTools::HoldOutSampler(data, 1.0, config.ensParamN());
	}
	sampler->randomSampling(config.ensParamSplitMode());
	EnsembleBuilder* eb = new EnsembleBuilder();
	eb->trainer(createTrainer(config, data));
	eb->sampler(sampler);
	return eb;
}

ModelEstimator* Factory::createModelEstimator(const Config& config, DataSet& data)
{
	DataTools::Sampler* sampler = 0;
 	if(config.msParamDataSelection() == "cv"){
		sampler = new DataTools::CrossSplitSampler(data, config.msParamN(), config.msParamK());
	}else if(config.msParamDataSelection() == "boot"){
		sampler = new DataTools::BootstrapSampler(data, config.msParamN());
	}else if(config.msParamDataSelection() == "none" || config.msParamDataSelection() == "dummy"){
		sampler = new DataTools::DummySampler(data, config.msParamN());
	}else if(config.msParamDataSelection() == "hold"){
		sampler = new DataTools::HoldOutSampler(data, config.msParamNumTrainingData(), config.msParamN());
	}
	sampler->randomSampling(config.msParamSplitMode());
	ModelEstimator* me = new ModelEstimator();
	me->sampler(sampler);
	me->ensembleBuilder(createEnsembleBuilder(config, data));
	return me;
}

