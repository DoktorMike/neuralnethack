#include "GradientDescent.hh"
#include "Error.hh"
#include "MatrixTools.hh"

//#include <ios>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace MatrixTools;

GradientDescent::GradientDescent(
		double te,
		uint bs, 
		double lr, 
		double dlr, 
		double m,
		bool we,
		double alpha, 
		double w0)
:Trainer(te, bs, we, alpha, w0), 
	theLearningRate(lr), theDecLearningRate(dlr), theMomentum(m){}

GradientDescent::~GradientDescent(){}

double GradientDescent::learningRate(){return theLearningRate;}

void GradientDescent::learningRate(double lr){theLearningRate=lr;}

double GradientDescent::decLearningRate(){return theDecLearningRate;}

void GradientDescent::decLearningRate(double dlr){theDecLearningRate=dlr;}

double GradientDescent::momentum(){return theMomentum;}

void GradientDescent::momentum(double m){theMomentum=m;}

void GradientDescent::train(Mlp& mlp, DataSet& dset)
{
	assert(theBatchSize <= dset.size());

	theError->mlp(&mlp);
	double err = 10;
	double prevErr = 100;
	uint cntr = theNumEpochs; 

	while(cntr-- && !hasConverged(err, prevErr)){//err > theTrainingError){
		prevErr = err;
		err = train(dset);
		updateLearningRate(err, prevErr);
		if(cntr % 100 == 0)
			cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr<<endl;
		//cout<<"Learning rate: "<<theLearningRate<<endl;
	}
	cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr-1<<endl;
}

//PRIVATE--------------------------------------------------------------------//

GradientDescent::GradientDescent(const GradientDescent& gd):
	Trainer(gd.theTrainingError,
			gd.theBatchSize,
			gd.theWeightElimOn,
			gd.theWeightElimAlpha,
			gd.theWeightElimW0)
{*this=gd;}

GradientDescent& GradientDescent::operator=(const GradientDescent& gd)
{
	if(this!=&gd){
		theBatchSize=gd.theBatchSize;
		theLearningRate=gd.theLearningRate;
		theDecLearningRate=gd.theDecLearningRate;
		theMomentum=gd.theMomentum;
	}
	return *this;
}

double GradientDescent::train(DataSet& dset)
{
	Mlp* mlp = theError->mlp();
	theError->dset(&dset);
	double err = theError->gradient();

	for(uint i=0; i<mlp->nLayers(); ++i){
		Layer& l = mlp->layer(i);
		for(uint i=0; i<l.nWeights(); ++i){
			double upd = l.gradients(i);
			if(theWeightElimOn == true)
				upd += weightElimination(l.weights(i));
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
