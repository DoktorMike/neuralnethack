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
	double err = INT_MAX;
	double prevErr = err;
	uint cntr = theNumEpochs; 

	while(cntr-- && err > theTrainingError){
		err = train(dset);
		updateLearningRate(err, prevErr);
		prevErr = err;
		if(cntr % 100 == 0)
			cout<<"Training error: "<<err<<" Epoch: "<<theNumEpochs-cntr<<endl;
		//cout<<"Learning rate: "<<theLearningRate<<endl;
	}
	//cout<<"The training error in epoch "<<theNumEpochs-cntr-1<<" is "<<err<<endl;
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
	Mlp* theMlp = theError->mlp();
	theError->dset(&dset);
	double err = theError->gradient();

	for(uint i=0; i<theMlp->nLayers(); ++i)
		for(uint j=0; j<(*theMlp)[i].nNeurons(); ++j)
			for(uint k=0; k<(*theMlp)[i][j].nWeights(); ++k){
				Neuron& n=(*theMlp)[i][j]; //alias
				double upd = 0;
				upd += -theLearningRate * n.gradient(k);
				if(theWeightElimOn == true)
					upd += -theLearningRate * weightElimination(n.weights(k));
				upd += theMomentum * n.prevWeightUpd(k);
				n.prevWeightUpd(k,upd); //Store it in prevWUpd.
				n[k]+=upd; //Update weights.
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
