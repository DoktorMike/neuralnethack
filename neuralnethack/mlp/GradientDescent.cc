/*$Id: GradientDescent.cc 1684 2007-10-12 15:55:07Z michael $*/

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

Trainer* GradientDescent::clone() const
{
	return new GradientDescent(*this);
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
