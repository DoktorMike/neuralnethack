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

using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
using std::ostringstream;
using std::pair;
using std::vector;

// PUBLIC

Trainer* EnsembleBuilder::trainer() const {
	return theTrainer;
}
void EnsembleBuilder::trainer(Trainer* t) {
	theTrainer = t;
}

Sampler* EnsembleBuilder::sampler() const {
	return theSampler;
}
void EnsembleBuilder::sampler(Sampler* s) {
	theSampler = s;
}

vector<Session>& EnsembleBuilder::sessions() {
	return theSessions;
}

Ensemble* EnsembleBuilder::getEnsemble() {
	Ensemble* ensemble = 0;
	if (theSessions.empty()) {
		cerr << "Error: No ensemble has been built yet" << endl;
	} else {
		ensemble = new Ensemble();
		for (auto it = theSessions.begin(); it != theSessions.end(); ++it)
			ensemble->addMlp(it->ensemble->mlp(0));
	}
	return ensemble;
}

Ensemble* EnsembleBuilder::buildEnsemble() {
	assert(isValid());

	Ensemble* ensemble = new Ensemble();
	theSessions.clear();

	uint cntr = 1;
	cout << "Building ensemble of size " << theSampler->howMany() << endl;
	while (theSampler->hasNext()) {
		cout << "Building MLP " << cntr++ << " of " << theSampler->howMany() << endl;
		pair<DataSet, DataSet>* dataSets = theSampler->next();
		DataSet& trnData = dataSets->first;
		DataSet& valData = dataSets->second;
		auto newMlp = theTrainer->trainNew(trnData, cout);
		ensemble->addMlp(*newMlp); // This copies the mlp.
		theSessions.push_back(Session(std::make_unique<Ensemble>(*newMlp, 1),
		                              std::make_unique<DataSet>(trnData),
		                              std::make_unique<DataSet>(valData)));
		delete dataSets;
	}

	return ensemble;
}

// PROTECTED
EnsembleBuilder::EnsembleBuilder() : theTrainer(0), theSampler(0), theSessions(0) {}

EnsembleBuilder::EnsembleBuilder(const EnsembleBuilder& eb) {
	*this = eb;
}

EnsembleBuilder::~EnsembleBuilder() {
	theSessions.clear();
}

EnsembleBuilder& EnsembleBuilder::operator=(const EnsembleBuilder& eb) {
	if (this != &eb) {
		theTrainer = eb.theTrainer;
		theSampler = eb.theSampler;
		theSessions = eb.theSessions;
	}
	return *this;
}

bool EnsembleBuilder::isValid() const {
	return theTrainer && theTrainer->isValid() && theSampler;
}

// PRIVATE
