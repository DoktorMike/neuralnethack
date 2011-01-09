#include "Trainer.hh"

#include <cmath>

using namespace NeuralNetHack;

Trainer::Trainer(double te, 
		uint bs,
		bool we, 
		double alpha, 
		double w0):theError(0), 
		theNumEpochs(0), 
		theTrainingError(te), 
		theBatchSize(bs),
		theWeightUpdate(0), 
		theWeightElimOn(we), 
		theWeightElimAlpha(alpha),theWeightElimW0(w0) 
{}

Trainer::~Trainer(){}

Mlp* Trainer::mlp(){return theError->mlp();}

void Trainer::mlp(Mlp* mlp){theError->mlp(mlp);}

Error* Trainer::error(){return theError;}

void Trainer::error(Error* e){theError = e;}

uint Trainer::numEpochs(){return theNumEpochs;}

void Trainer::numEpochs(uint ne){theNumEpochs = ne;}

double Trainer::trainingError(){return theTrainingError;}

void Trainer::trainingError(double te){theTrainingError=te;}

uint Trainer::batchSize(){return theBatchSize;}

void Trainer::batchSize(uint bs){theBatchSize = bs;}

bool Trainer::weightElimOn(){return theWeightElimOn;}

void Trainer::weightElimOn(bool on){theWeightElimOn = on;}

double Trainer::weightElimAlpha(){return theWeightElimAlpha;}

void Trainer::weightElimAlpha(double alpha){theWeightElimAlpha = alpha;}

double Trainer::weightElimW0(){return theWeightElimW0;}

void Trainer::weightElimW0(double w0){theWeightElimW0 = w0;}

double Trainer::weightElimination(double wi)
{
	double alpha = theWeightElimAlpha;
	double w0 = theWeightElimW0;
	return alpha*( (2*wi*pow(w0,2))/pow( pow(w0,2)+pow(wi,2), 2) );
}

bool Trainer::validate()
{
	assert(theError != 0);
	assert(theError->mlp() != 0);
	return true;
}

//PRIVATE--------------------------------------------------------------------//

Trainer::Trainer(const Trainer& trainer)
{*this=trainer;}

Trainer& Trainer::operator=(const Trainer& trainer)
{
	if(this != &trainer){
	}
	return *this;
}

