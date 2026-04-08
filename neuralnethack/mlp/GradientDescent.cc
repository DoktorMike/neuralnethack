#include "GradientDescent.hh"
#include "Error.hh"
#include "../matrixtools/MatrixTools.hh"

#include <ostream>
#include <cassert>
#include <iomanip>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;
using namespace std;
using std::unique_ptr;

GradientDescent::GradientDescent(Mlp& mlp, DataSet& data, Error& error, double te, uint bs, double lr, double dlr, double m):Trainer(mlp, data, error, te, bs), theLearningRate(lr), theDecLearningRate(dlr), theMomentum(m)
{}

GradientDescent::~GradientDescent(){}

double GradientDescent::learningRate() const {return theLearningRate;}

void GradientDescent::learningRate(double lr) {theLearningRate=lr;}

double GradientDescent::decLearningRate() const {return theDecLearningRate;}

void GradientDescent::decLearningRate(double dlr){theDecLearningRate=dlr;}

double GradientDescent::momentum() const {return theMomentum;}

void GradientDescent::momentum(double m){theMomentum=m;}

void GradientDescent::train(ostream& os)
{
	if(theBatchSize > theData->size()){
		cerr<<"Warning: Batch size larger than DataSet, reseting to DataSet size."<<endl;
		theBatchSize = theData->size();
	}

	double origLearnRate = theLearningRate; //Save the learning rate
	double err = theError->outputError(*theMlp, *theData);
	double prevErr = err+1;
	uint cntr = theNumEpochs;
	const uint w = 14; //formatting
	const uint maxRounds = 2; //max number of full batch before var learn rate.
	uint nrounds = 0; //current number of full batch runs.
	DataSet blockData(*theData);
	uint index = 0;

	os.setf(ios::left);
	os<<setw(w)<<"# Epoch"<<setw(w)<<"TrnErr"<<setw(w)<<"LrnRate"<<endl;
	do{
		if(buildBlock(blockData, index) == true) ++nrounds;
		train(blockData);
		if(nrounds >= maxRounds){
			cntr--;
			nrounds = 0;
			prevErr = err;
			err = theError->outputError(*theMlp, *theData);
			updateLearningRate(err, prevErr);
			if(cntr % 100 == 0) 
				os<<setw(w)<<theNumEpochs-cntr<<setw(w)<<err<<setw(w)<<theLearningRate<<endl;
		}
	}while(cntr && !hasConverged(err, prevErr));
	os<<setw(w)<<theNumEpochs-cntr<<setw(w)<<err<<setw(w)<<theLearningRate<<endl;
	theLearningRate = origLearnRate; //Restore the learning rate
}

unique_ptr<Trainer> GradientDescent::clone() const
{
	return unique_ptr<Trainer>(new GradientDescent(*this));
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
	theError->mlp(*theMlp);
	theError->dset(dset);
	double err = theError->gradient();

	const double lr = theLearningRate;
	const double mom = theMomentum;

	for(uint layer = 0; layer < theMlp->nLayers(); ++layer){
		Layer& l = theMlp->layer(layer);
		const uint nw = l.nWeights();
		double* __restrict__ w   = l.weights().data();
		double* __restrict__ g   = l.gradients().data();
		double* __restrict__ upd = l.weightUpdates().data();

		for(uint j = 0; j < nw; ++j){
			double u = -lr * g[j] + mom * upd[j];
			upd[j] = u;
			w[j] += u;
		}
	}
	return err;
}

void GradientDescent::updateLearningRate(double err, double prevErr)
{
	if(err>prevErr) theLearningRate *= theDecLearningRate;
	else{
		double scale = 1.0+(1.0-theDecLearningRate)/10.0;
		theLearningRate *= scale;
	}
}

bool GradientDescent::buildBlock(DataSet& blockData, uint& cntr) const
{
	bool roundabout = false;
	vector<uint> indices(theBatchSize, 0);
	for(uint i=0; i<theBatchSize; ++i, ++cntr){
		if(cntr >= theData->indices().size()){
			cntr = 0;
			roundabout = true;
		}
		indices.at(i) = theData->indices().at(cntr);
	}
	blockData.indices(indices);
	//cout<<"first: "<<indices.front()<<" last: "<<indices.back()<<" cntr: "<<cntr<<" size: "<<blockData.size()<<endl;
	return roundabout;
}
