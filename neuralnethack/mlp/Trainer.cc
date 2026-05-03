#include "Trainer.hh"
#include "Layer.hh"
#include "Mlp.hh"

#include <cmath>
#include <iostream>
#include <limits>
#include <ostream>
#include <algorithm>

#define CONVERGENCE_TOLERANCE 1e-5

using namespace MultiLayerPerceptron;
using namespace DataTools;

using std::ostream;
using std::unique_ptr;

Trainer::Trainer(Mlp& mlp, DataSet& data, Error& error, double te, uint bs)
    : theMlp(&mlp), theData(&data), theError(&error), theOwnedError(nullptr), theNumEpochs(0),
      theTrainingError(te), theBatchSize(bs) {}

Trainer::Trainer(unique_ptr<Error> error, DataSet& data, double te, uint bs)
    : theMlp(&error->mlp()), theData(&data), theError(error.get()),
      theOwnedError(std::move(error)), theNumEpochs(0), theTrainingError(te), theBatchSize(bs) {}

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

DataSet* Trainer::validationData() const {
	return theValData;
}
void Trainer::validationData(DataSet* v) {
	theValData = v;
}

const std::string& Trainer::learningCurveFile() const {
	return theLearningCurvePath;
}
void Trainer::learningCurveFile(const std::string& path) {
	theLearningCurvePath = path;
	theLearningCurveStream.reset();
}

void Trainer::recordLearningPoint(uint epoch, double trainErr) {
	if (theLearningCurvePath.empty()) return;
	if (!theLearningCurveStream) {
		theLearningCurveStream = std::make_unique<std::ofstream>(theLearningCurvePath);
		if (!theLearningCurveStream || !*theLearningCurveStream) {
			std::cerr << "Trainer: cannot open learning-curve file '" << theLearningCurvePath
			          << "'" << std::endl;
			theLearningCurvePath.clear();
			theLearningCurveStream.reset();
			return;
		}
		*theLearningCurveStream << "# epoch  trainErr";
		if (theValData) *theLearningCurveStream << "  valErr";
		*theLearningCurveStream << "\n";
	}
	*theLearningCurveStream << epoch << "  " << trainErr;
	if (theValData) {
		// outputError(Mlp&, DataSet&) repoints the Error at the val data;
		// restore to training pointers afterwards so gradient() stays correct.
		const double valErr = theError->outputError(*theMlp, *theValData);
		theError->mlp(*theMlp);
		theError->dset(*theData);
		*theLearningCurveStream << "  " << valErr;
	}
	*theLearningCurveStream << "\n";
	theLearningCurveStream->flush();
}

void Trainer::earlyStopping(uint patience, double minDelta) {
	theEsPatience = patience;
	theEsMinDelta = minDelta;
}

void Trainer::resetEarlyStopping() {
	theEsBestVal = std::numeric_limits<double>::infinity();
	theEsStaleEpochs = 0;
	theEsHasBest = false;
	theEsTriggered = false;
	theEsBestW.clear();
	theEsBestGammas.clear();
	theEsBestBetas.clear();
}

void Trainer::snapshotBestWeights() {
	if (!theMlp) return;
	theEsBestW = theMlp->weights();
	const uint nL = theMlp->nLayers();
	theEsBestGammas.assign(nL, {});
	theEsBestBetas.assign(nL, {});
	for (uint i = 0; i < nL; ++i) {
		Layer& l = theMlp->layer(i);
		if (l.normType() != NormType::None) {
			theEsBestGammas[i] = l.gamma();
			theEsBestBetas[i] = l.beta();
		}
	}
	theEsHasBest = true;
}

void Trainer::restoreBestWeights() {
	if (!theEsHasBest || !theMlp) return;
	theMlp->weights(theEsBestW);
	const uint nL = theMlp->nLayers();
	for (uint i = 0; i < nL && i < theEsBestGammas.size(); ++i) {
		Layer& l = theMlp->layer(i);
		if (l.normType() != NormType::None && !theEsBestGammas[i].empty()) {
			l.gamma() = theEsBestGammas[i];
			l.beta() = theEsBestBetas[i];
		}
	}
}

bool Trainer::earlyStopCheck() {
	if (theEsPatience == 0 || theValData == nullptr || theMlp == nullptr ||
	    theError == nullptr)
		return false;

	const double valErr = theError->outputError(*theMlp, *theValData);
	// outputError(Mlp&, DataSet&) repoints Error at val data; restore for
	// downstream gradient() calls.
	theError->mlp(*theMlp);
	theError->dset(*theData);

	if (valErr < theEsBestVal - theEsMinDelta) {
		theEsBestVal = valErr;
		theEsStaleEpochs = 0;
		snapshotBestWeights();
		return false;
	}
	++theEsStaleEpochs;
	if (theEsStaleEpochs >= theEsPatience) {
		theEsTriggered = true;
		restoreBestWeights();
		return true;
	}
	return false;
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
		theValData = trainer.theValData;
		// Do not copy the learning-curve path/stream: clones must not
		// share an output file (they would race or clobber each other).
		theLearningCurvePath.clear();
		theLearningCurveStream.reset();
	}
	return *this;
}
