#include "ModelEstimator.hh"
#include "evaltools/Roc.hh"

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

// PUBLIC

ModelEstimator::ModelEstimator() : theEnsembleBuilder(nullptr), theSampler(nullptr), theSessions() {}

ModelEstimator::~ModelEstimator() = default;

// ACCESSOR AND MUTATOR

EnsembleBuilder* ModelEstimator::ensembleBuilder() {
	return theEnsembleBuilder.get();
}
void ModelEstimator::ensembleBuilder(std::unique_ptr<EnsembleBuilder> eb) {
	theEnsembleBuilder = std::move(eb);
}

Sampler* ModelEstimator::sampler() {
	return theSampler.get();
}
void ModelEstimator::sampler(std::unique_ptr<Sampler> s) {
	theSampler = std::move(s);
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
		auto dataSets = theSampler->next();
		DataSet& trnData = dataSets.first;
		DataSet& valData = dataSets.second;
		theEnsembleBuilder->sampler()->data(&trnData);
		theEnsembleBuilder->sampler()->reset();
		Ensemble* ensemble = theEnsembleBuilder->buildEnsemble();
		theSessions.push_back(Session(std::unique_ptr<Ensemble>(ensemble),
		                              std::make_unique<DataSet>(trnData),
		                              std::make_unique<DataSet>(valData)));
	}
	return estimateModel(errorFunc);
}

// PRIVATE
