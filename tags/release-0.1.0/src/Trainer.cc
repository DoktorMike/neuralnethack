#include "Trainer.hh"
#include "SummedSquare.hh"

using namespace NetHack;

Trainer::Trainer(string e, double te)
    :theError(0), theTrainingError(te), theWeightUpdate(0) 
{
    if(e == SSE)
	theError = new SummedSquare();
}

Trainer::Trainer()
    :theError(0), theTrainingError(0), theWeightUpdate(0){}

Trainer::~Trainer(){delete theError;}

string Trainer::error(){return SSE;}

void Trainer::error(string e)
{
    if(theError!=0)
	delete theError;
    if(e == SSE)
	theError = new SummedSquare();
}

double Trainer::trainingError(){return theTrainingError;}

void Trainer::trainingError(double te){theTrainingError=te;}

//PRIVATE--------------------------------------------------------------------//

Trainer::Trainer(const Trainer& trainer)
{*this=trainer;}

Trainer& Trainer::operator=(const Trainer& trainer)
{
    if(this != &trainer){
    }
    return *this;
}

