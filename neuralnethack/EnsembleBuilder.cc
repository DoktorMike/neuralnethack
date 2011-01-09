/*$Id: EnsembleBuilder.cc 1627 2007-05-08 16:40:20Z michael $*/

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


#include "EnsembleBuilder.hh"
#include "Ensemble.hh"
#include "PrintUtils.hh"

#include <cassert>
#include <ostream>
#include <vector>
#include <sstream>
#include <utility>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;

using std::ostream;
using std::vector;
using std::ostringstream;
using std::cout;
using std::cerr;
using std::endl;
using std::pair;

//PUBLIC


Trainer* EnsembleBuilder::trainer() const {return theTrainer;}
void EnsembleBuilder::trainer(Trainer* t){theTrainer = t;}

Sampler* EnsembleBuilder::sampler() const {return theSampler;}
void EnsembleBuilder::sampler(Sampler* s){theSampler = s;}

vector<Session>& EnsembleBuilder::sessions(){return theSessions;}

Ensemble* EnsembleBuilder::getEnsemble()
{
	Ensemble* ensemble = 0;
	if(theSessions.empty()){
		cerr<<"Error: No ensemble has been built yet"<<endl;
	}else{
		ensemble = new Ensemble();
		for(vector<Session>::iterator it = theSessions.begin(); it != theSessions.end(); ++it)
			ensemble->addMlp(it->committee->mlp(0));
	}
	return ensemble;
}

Ensemble* EnsembleBuilder::buildEnsemble()
{
	assert(isValid());

	Ensemble* ensemble = new Ensemble();
	theSessions.clear();

	//cout<<"Building ensemble using Bagging with N="<<n<<endl;
	while(theSampler->hasNext()){
		//cout<<"Run (N): "<<i+1<<"\n";
		pair<DataSet, DataSet>* dataSets = theSampler->next();
		DataSet& trnData = dataSets->first;
		DataSet& valData = dataSets->second;
		Mlp* newMlp = theTrainer->trainNew(trnData, cout);
		ensemble->addMlp(*newMlp); //This copies the mlp.
		theSessions.push_back(Session(new Ensemble(*newMlp, 1), 
					new DataSet(trnData), new DataSet(valData)));
		delete newMlp;
		delete dataSets;
	}

	return ensemble;
}


//PROTECTED
EnsembleBuilder::EnsembleBuilder():theTrainer(0), theSampler(0), theSessions(0)
{}

EnsembleBuilder::EnsembleBuilder(const EnsembleBuilder& eb){*this = eb;}

EnsembleBuilder::~EnsembleBuilder()
{
	theSessions.clear();
}

EnsembleBuilder& EnsembleBuilder::operator=(const EnsembleBuilder& eb)
{
	if(this != &eb){
		theTrainer = eb.theTrainer;
		theSampler = eb.theSampler;
		theSessions = eb.theSessions;
	}
	return *this;
}

bool EnsembleBuilder::isValid() const {return theTrainer && theTrainer->isValid() && theSampler;}

//PRIVATE

