#include "Trainer.hh"

#include <cmath>
#include <ostream>
#include <algorithm>

#define CONVERGENCE_TOLERANCE 1e-5

using namespace MultiLayerPerceptron;
using namespace DataTools;

using std::ostream;
using std::unique_ptr;

Trainer::Trainer(Mlp& mlp, DataSet& data, Error& error, double te, uint bs)
    : theMlp(&mlp), theData(&data), theError(&error), theNumEpochs(0), theTrainingError(te),
      theBatchSize(bs) {}

Trainer::~Trainer() {}

Mlp* Trainer::mlp() {
	return theMlp;
}
void Trainer::mlp(Mlp* mlp) {
	theMlp = mlp;
}

DataSet* Trainer::data() {
	return theData;
}
void Trainer::data(DataSet* data) {
	theData = data;
}

Error* Trainer::error() {
	return theError;
}
void Trainer::error(Error* e) {
	theError = e;
}

uint Trainer::numEpochs() const {
	return theNumEpochs;
}
void Trainer::numEpochs(uint ne) {
	theNumEpochs = ne;
}

double Trainer::trainingError() const {
	return theTrainingError;
}
void Trainer::trainingError(double te) {
	theTrainingError = te;
}

uint Trainer::batchSize() const {
	return theBatchSize;
}
void Trainer::batchSize(uint bs) {
	theBatchSize = bs;
}

bool Trainer::hasConverged(double ecurr, double eprev) const {
	double change = fabs(eprev - ecurr);
	double tol = CONVERGENCE_TOLERANCE * ecurr;
	/*
	std::cout<<"Change: "<<change<<std::endl;
	std::cout<<"Tol: "<<tol<<std::endl;
	*/
	return (change <= tol) ? true : false;
}

bool Trainer::isValid() const {
	return theError != 0 && theMlp != 0;
}

void Trainer::train(Mlp& mlp, DataSet& data, ostream& os) {
	theMlp = &mlp;
	theData = &data;
	theMlp->training(true);
	train(os);
	theMlp->training(false);
}

unique_ptr<Mlp> Trainer::trainNew(DataSet& data, ostream& os) {
	theData = &data;
	return trainNew(os);
}

unique_ptr<Mlp> Trainer::trainNew(ostream& os) {
	Mlp* tmp = theMlp;
	theMlp = new Mlp(*tmp);
	theMlp->regenerateWeights();
	theMlp->training(true);
	train(os);
	theMlp->training(false);
	std::swap(tmp, theMlp);
	return unique_ptr<Mlp>(tmp);
}

// PROTECTED--------------------------------------------------------------------//

Trainer::Trainer(const Trainer& trainer) {
	*this = trainer;
}

Trainer& Trainer::operator=(const Trainer& trainer) {
	if (this != &trainer) {
		theMlp = trainer.theMlp;
		theData = trainer.theData;
		theError = trainer.theError;
		theNumEpochs = trainer.theNumEpochs;
		theTrainingError = trainer.theTrainingError;
		theBatchSize = trainer.theBatchSize;
	}
	return *this;
}
