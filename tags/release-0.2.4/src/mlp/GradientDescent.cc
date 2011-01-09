#include "GradientDescent.hh"
#include "Error.hh"
#include "matrixtools/MatrixTools.hh"

#include <ostream>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;
using namespace std;

GradientDescent::GradientDescent(Mlp& mlp, DataSet& data, Error& error, double te, uint bs, double lr, double dlr, double m):Trainer(mlp, data, error, te, bs), theLearningRate(lr), theDecLearningRate(dlr), theMomentum(m)
{}

GradientDescent::~GradientDescent(){}

double GradientDescent::learningRate(){return theLearningRate;}

void GradientDescent::learningRate(double lr){theLearningRate=lr;}

double GradientDescent::decLearningRate(){return theDecLearningRate;}

void GradientDescent::decLearningRate(double dlr){theDecLearningRate=dlr;}

double GradientDescent::momentum(){return theMomentum;}

void GradientDescent::momentum(double m){theMomentum=m;}

void GradientDescent::train()
{
	assert(theBatchSize <= theData->size());

	double err = 10;
	double prevErr = 100;
	uint cntr = theNumEpochs; 

	while(cntr-- && !hasConverged(err, prevErr)){//err > theTrainingError)
		prevErr = err;
		err = train(*theData);
		updateLearningRate(err, prevErr);
		if(cntr % 100 == 0)
			cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr<<endl;
		//cout<<"Learning rate: "<<theLearningRate<<endl;
	}
	cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr-1<<endl;
}

//PRIVATE--------------------------------------------------------------------//

GradientDescent::GradientDescent(const GradientDescent& gd):Trainer(gd)
{*this=gd;}

GradientDescent& GradientDescent::operator=(const GradientDescent& gd)
{
	if(this!=&gd){
		Trainer::operator=(gd);
		theLearningRate=gd.theLearningRate;
		theDecLearningRate=gd.theDecLearningRate;
		theMomentum=gd.theMomentum;
	}
	return *this;
}

double GradientDescent::train(DataSet& dset)
{
	theError->mlp(theMlp);
	theError->dset(&dset);
	double err = theError->gradient();

	for(uint i=0; i<theMlp->nLayers(); ++i){
		Layer& l = theMlp->layer(i);
		for(uint i=0; i<l.nWeights(); ++i){
			double upd = l.gradients(i);
			upd *= -theLearningRate;
			upd += theMomentum * l.weightUpdates(i);
			l.weightUpdates(i) = upd; //Store it in weightUpdates
			l.weights(i) += upd; //Update weights.
		}
	}
	return err;
}

void GradientDescent::updateLearningRate(double err, double prevErr)
{
	if(err>prevErr)
		theLearningRate *= theDecLearningRate;
	else{
		double scale = 1.0+(1.0-theDecLearningRate)/10.0;
		theLearningRate *= scale;
	}
}
