#include "GradientDescent.hh"
#include "Error.hh"
#include "MatrixTools.hh"

//#include <ios>

using namespace NetHack;
using namespace DataTools;
using namespace MatrixTools;

GradientDescent::GradientDescent(string e, double te,
	uint bs, double lr, double dlr, double m):Trainer(e, te),
	theBatchSize(bs), theLearningRate(lr), theDecLearningRate(dlr), 
	theMomentum(m)
{
}

GradientDescent::~GradientDescent()
{}

void GradientDescent::train(Committee& committee, DataSet& dset, 
	uint epochs)
{
    for(uint i=0; i<committee.size(); ++i){
	Mlp& mlp=committee[i];
	dset.reset();
	train(mlp, dset, epochs);
    }
}

void GradientDescent::learningRate(double lr){theLearningRate=lr;}

void GradientDescent::decLearningRate(double dlr){theDecLearningRate=dlr;}

void GradientDescent::momentum(double m){theMomentum=m;}

double GradientDescent::learningRate(){return theLearningRate;}

double GradientDescent::decLearningRate(){return theDecLearningRate;}

double GradientDescent::momentum(){return theMomentum;}

//PRIVATE--------------------------------------------------------------------//

GradientDescent::GradientDescent(const GradientDescent& gd)
	:Trainer(gd.theError->type(), gd.theTrainingError)
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

void GradientDescent::train(Mlp& mlp, DataSet& dset, uint epochs)
{
    assert(theBatchSize <= dset.size());

    double err = INT_MAX;
    double prevErr = err;
    uint cntr = epochs; 

    while(cntr-- && err > theTrainingError){
	err = train(mlp, dset);
	updateLearningRate(err, prevErr);
	prevErr = err;
	if(cntr % 100 == 0)
	    cout<<"Error is "<<err<<" in epoch "<<epochs-cntr<<endl;
	//cout<<"Learning rate: "<<theLearningRate<<endl;
    }
    cout<<"The error in epoch "<<epochs-cntr-1<<" is "<<err<<endl;
}

double GradientDescent::train(Mlp& mlp, DataSet& dset)
{
    double err = theError->gradient(mlp,dset,theBatchSize);
    for(uint i=0; i<mlp.nLayers(); ++i)
	for(uint j=0; j<mlp[i].nNeurons(); ++j)
	    for(uint k=0; k<mlp[i][j].nWeights(); ++k){
		Neuron& n=mlp[i][j]; //alias
		double upd=n.gradient(k)*theLearningRate+
		    theMomentum*n.prevWeightUpd(k); //The weight update
		n.prevWeightUpd(k,upd); //Store it in prevWUpd.
		n[k]-=upd; //Update weights.
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
