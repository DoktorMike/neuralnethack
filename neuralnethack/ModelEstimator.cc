#include "ModelEstimator.hh"
#include "evaltools/Roc.hh"
#include "PrintUtils.hh"

#include <sstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

// PUBLIC

ModelEstimator::ModelEstimator() : theEnsembleBuilder(0), theSampler(0), theSessions(0) {}

ModelEstimator::ModelEstimator(EnsembleBuilder& eb, Sampler& s)
    : theEnsembleBuilder(&eb), theSampler(&s), theSessions(0) {}

ModelEstimator::~ModelEstimator() {
	theSessions.clear();
}

// ACCESSOR AND MUTATOR

EnsembleBuilder* ModelEstimator::ensembleBuilder() {
	return theEnsembleBuilder;
}
void ModelEstimator::ensembleBuilder(EnsembleBuilder* eb) {
	theEnsembleBuilder = eb;
}

Sampler* ModelEstimator::sampler() {
	return theSampler;
}
void ModelEstimator::sampler(Sampler* s) {
	theSampler = s;
}

vector<Session>& ModelEstimator::sessions() {
	return theSessions;
}

pair<double, double>* ModelEstimator::estimateModel(double (*errorFunc)(Ensemble& ensemble,
                                                                        DataSet& data)) {
	double trnAuc = 0;
	double valAuc = 0;

	for (vector<Session>::iterator it = theSessions.begin(); it != theSessions.end(); ++it) {
		double tmp = (*errorFunc)(*(it->ensemble), *(it->trnData));
		// cout<<"----------trnAUC 2: "<<tmp<<endl; //DEBUG
		trnAuc += tmp;
		tmp = (*errorFunc)(*(it->ensemble), *(it->valData));
		// cout<<"----------valAUC 2: "<<tmp<<endl; //DEBUG
		valAuc += tmp;
	}
	trnAuc /= (double)theSessions.size();
	valAuc /= (double)theSessions.size();
	return new pair<double, double>(trnAuc, valAuc);
}

pair<double, double>* ModelEstimator::runAndEstimateModel(double (*errorFunc)(Ensemble& ensemble,
                                                                              DataSet& data)) {
	assert(theEnsembleBuilder && theSampler);

	cout << "Estimating model in " << theSampler->howMany() << " runs" << endl;
	uint cntr = 1;
	while (theSampler->hasNext()) {
		cout << "Estimation run " << cntr++ << " of " << theSampler->howMany() << endl;
		pair<DataSet, DataSet>* dataSets = theSampler->next();
		DataSet& trnData = dataSets->first;
		DataSet& valData = dataSets->second;
		theEnsembleBuilder->sampler()->data(&trnData);
		theEnsembleBuilder->sampler()->reset();
		Ensemble* ensemble = theEnsembleBuilder->buildEnsemble();
		theSessions.push_back(Session(std::unique_ptr<Ensemble>(ensemble),
		                              std::make_unique<DataSet>(trnData),
		                              std::make_unique<DataSet>(valData)));
		delete dataSets;
	}
	return estimateModel(errorFunc);
}

// PRIVATE
