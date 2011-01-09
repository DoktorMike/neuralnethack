#include "Trainer.hh"

#include <cmath>

#define CONVERGENCE_TOLERANCE	0.0001

using namespace MultiLayerPerceptron;
using namespace DataTools;

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

uint Trainer::numEpochs(){return theNumEpochs;}
void Trainer::numEpochs(uint ne){theNumEpochs = ne;}

double Trainer::trainingError(){return theTrainingError;}
void Trainer::trainingError(double te){theTrainingError=te;}

uint Trainer::batchSize(){return theBatchSize;}
void Trainer::batchSize(uint bs){theBatchSize = bs;}

bool Trainer::hasConverged(double ecurr, double eprev)
{
	double change = fabs(eprev-ecurr);
	double tol = CONVERGENCE_TOLERANCE * ecurr;
	//cout<<"Fraction: "<<change<<"\n";
	//cout<<"Tol: "<<tol<<endl;
	return (change <= tol) ? true : false;
}

bool Trainer::isValid(){return theError != 0 && theMlp != 0;}

void Trainer::train(Mlp& mlp, DataSet& data)
{
	theMlp = &mlp;
	theData = &data;
	train();
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

