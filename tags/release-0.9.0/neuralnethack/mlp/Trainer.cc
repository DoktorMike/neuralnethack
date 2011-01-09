/*$Id: Trainer.cc 1627 2007-05-08 16:40:20Z michael $*/

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


#include "Trainer.hh"

#include <cmath>
#include <ostream>
#include <algorithm>

#define CONVERGENCE_TOLERANCE	1e-5

using namespace MultiLayerPerceptron;
using namespace DataTools;

using std::ostream;

Trainer::Trainer(Mlp& mlp, DataSet& data, Error& error, double te, uint bs)
	:theMlp(&mlp), 
	theData(&data),
	theError(&error), 
	theNumEpochs(0), 
	theTrainingError(te), 
	theBatchSize(bs){}

Trainer::~Trainer(){}

Mlp* Trainer::mlp(){return theMlp;}
void Trainer::mlp(Mlp* mlp){theMlp = mlp;}

DataSet* Trainer::data(){return theData;}
void Trainer::data(DataSet* data){theData = data;}

Error* Trainer::error(){return theError;}
void Trainer::error(Error* e){theError = e;}

uint Trainer::numEpochs() const {return theNumEpochs;}
void Trainer::numEpochs(uint ne){theNumEpochs = ne;}

double Trainer::trainingError() const {return theTrainingError;}
void Trainer::trainingError(double te){theTrainingError=te;}

uint Trainer::batchSize() const {return theBatchSize;}
void Trainer::batchSize(uint bs){theBatchSize = bs;}

bool Trainer::hasConverged(double ecurr, double eprev) const
{
	double change = fabs(eprev-ecurr);
	double tol = CONVERGENCE_TOLERANCE * ecurr;
	/*
	std::cout<<"Change: "<<change<<std::endl;
	std::cout<<"Tol: "<<tol<<std::endl;
	*/
	return (change <= tol) ? true : false;
}

bool Trainer::isValid() const {return theError != 0 && theMlp != 0;}

void Trainer::train(Mlp& mlp, DataSet& data, ostream& os)
{
	theMlp = &mlp;
	theData = &data;
	train(os);
}

Mlp* Trainer::trainNew(DataSet& data, ostream& os)
{
	theData = &data;
	return trainNew(os);
}

Mlp* Trainer::trainNew(ostream& os)
{
	Mlp* tmp = theMlp;
	theMlp = new Mlp(*tmp);
	theMlp->regenerateWeights();
	train(os);
	std::swap(tmp, theMlp);
	return tmp;
}

//PROTECTED--------------------------------------------------------------------//

Trainer::Trainer(const Trainer& trainer){*this = trainer;}

Trainer& Trainer::operator=(const Trainer& trainer)
{
	if(this != &trainer){
		theMlp = trainer.theMlp;
		theData = trainer.theData;
		theError = trainer.theError;
		theNumEpochs = trainer.theNumEpochs;
		theTrainingError = trainer.theTrainingError;
		theBatchSize = trainer.theBatchSize;
	}
	return *this;
}

