/*$Id: ModelEstimator.cc 1627 2007-05-08 16:40:20Z michael $*/

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


#include "ModelEstimator.hh"
#include "evaltools/Roc.hh"
#include "PrintUtils.hh"

#include <sstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

//PUBLIC

ModelEstimator::ModelEstimator():theEnsembleBuilder(0), theSampler(0), theSessions(0){}

ModelEstimator::ModelEstimator(EnsembleBuilder& eb, Sampler& s):theEnsembleBuilder(&eb), theSampler(&s), theSessions(0){}

ModelEstimator::~ModelEstimator()
{
	theSessions.clear();
}

//ACCESSOR AND MUTATOR

EnsembleBuilder* ModelEstimator::ensembleBuilder(){return theEnsembleBuilder;}
void ModelEstimator::ensembleBuilder(EnsembleBuilder* eb){theEnsembleBuilder = eb;}

Sampler* ModelEstimator::sampler(){return theSampler;}
void ModelEstimator::sampler(Sampler* s){theSampler = s;}

vector<Session>& ModelEstimator::sessions(){return theSessions;}

pair<double, double>* ModelEstimator::estimateModel(
		double (*errorFunc)(Ensemble& committee, DataSet& data))
{
	double trnAuc = 0;
	double valAuc = 0;

	for(vector<Session>::iterator it = theSessions.begin(); 
			it != theSessions.end(); ++it)
	{
		double tmp = (*errorFunc)(*(it->committee), *(it->trnData));
		//cout<<"----------trnAUC 2: "<<tmp<<endl; //DEBUG
		trnAuc += tmp;
		tmp = (*errorFunc)(*(it->committee), *(it->valData));
		//cout<<"----------valAUC 2: "<<tmp<<endl; //DEBUG
		valAuc += tmp;
	}
	trnAuc /= (double)theSessions.size();
	valAuc /= (double)theSessions.size();
	return new pair<double, double>(trnAuc, valAuc);
}

pair<double, double>* ModelEstimator::runAndEstimateModel(
		double (*errorFunc)(Ensemble& committee, DataSet& data))
{
	assert(theEnsembleBuilder && theSampler);

	//cout<<"Estimating model using Bootstrapper with N="<<n<<endl;
	while(theSampler->hasNext()){
		//cout<<"Run (N): "<<i+1<<"\n";
		pair<DataSet, DataSet>* dataSets = theSampler->next();
		DataSet& trnData = dataSets->first;
		DataSet& valData = dataSets->second;
		//cout<<"TrnData Size: "<<trnData.size()<<endl;
		//cout<<"ValData Size: "<<valData.size()<<endl;
		theEnsembleBuilder->sampler()->data(&trnData);
		theEnsembleBuilder->sampler()->reset();
		Ensemble* committee = theEnsembleBuilder->buildEnsemble();
		//cout<<" Ens Size: "<<committee->size()<<endl;
		Session e(committee, new DataSet(trnData), new DataSet(valData));
		theSessions.push_back(e);
		delete dataSets;
	}
	return estimateModel(errorFunc);
}

//PRIVATE

