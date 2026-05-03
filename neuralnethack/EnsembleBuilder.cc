#include "EnsembleBuilder.hh"
#include "Ensemble.hh"
#include "Random.hh"

#include <cassert>
#include <cstdlib>
#include <iostream>
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
	return theTrainer.get();
}
void EnsembleBuilder::trainer(std::unique_ptr<Trainer> t) {
	theTrainer = std::move(t);
}

Sampler* EnsembleBuilder::sampler() const {
	return theSampler.get();
}
void EnsembleBuilder::sampler(std::unique_ptr<Sampler> s) {
	theSampler = std::move(s);
}

void EnsembleBuilder::trainerFactory(TrainerFactory f) {
	theTrainerFactory = std::move(f);
}

void EnsembleBuilder::baseSeed(uint64_t s) {
	theBaseSeed = s;
}

void EnsembleBuilder::learningCurvePathBase(const std::string& base) {
	theLearningCurvePathBase = base;
}
const std::string& EnsembleBuilder::learningCurvePathBase() const {
	return theLearningCurvePathBase;
}

namespace {
// Insert "_NNN" before the last "." in `base`, or append if no extension.
// Index is 1-based and zero-padded to 3 digits for sortable file names.
std::string makeMemberPath(const std::string& base, int oneBasedIndex) {
	char idx[16];
	std::snprintf(idx, sizeof(idx), "_%03d", oneBasedIndex);
	const auto dot = base.find_last_of('.');
	const auto slash = base.find_last_of('/');
	if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
		return base.substr(0, dot) + idx + base.substr(dot);
	return base + idx;
}
} // namespace

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
	if (!isValid()) {
		std::cerr << "EnsembleBuilder: trainer or sampler not set. "
		          << "Config is likely missing the [ensemble] section "
		          << "(method/runs/parts/split)." << std::endl;
		std::abort();
	}

	Ensemble* ensemble = new Ensemble();
	theSessions.clear();

	// Drain the sampler upfront so the parallel loop has indexable input.
	std::vector<std::pair<DataSet, DataSet>> samples;
	while (theSampler->hasNext())
		samples.push_back(theSampler->next());
	const int N = static_cast<int>(samples.size());

	cout << "Building ensemble of size " << N << endl;

	std::vector<std::unique_ptr<MultiLayerPerceptron::Mlp>> trained(N);
	std::vector<Session> sessions(N);

	if (theTrainerFactory) {
		// Parallel path: each iteration owns a fresh trainer + Mlp + Error.
#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < N; ++i) {
			nnh::rand::seed(theBaseSeed + static_cast<uint64_t>(i));
			auto trainer = theTrainerFactory(samples[i].first);
			if (!theLearningCurvePathBase.empty())
				trainer->learningCurveFile(makeMemberPath(theLearningCurvePathBase, i + 1));
			if (!theLearningCurvePathBase.empty() || trainer->earlyStoppingPatience() > 0)
				trainer->validationData(&samples[i].second);
			ostringstream local;
			trained[i] = trainer->trainNew(samples[i].first, local);
			sessions[i] = Session(std::make_unique<Ensemble>(*trained[i], 1),
			                      std::make_unique<DataSet>(samples[i].first),
			                      std::make_unique<DataSet>(samples[i].second));
#pragma omp critical
			cout << "Built MLP " << (i + 1) << " of " << N << endl;
		}
	} else {
		// Serial fallback for callers that haven't supplied a factory.
		for (int i = 0; i < N; ++i) {
			cout << "Building MLP " << (i + 1) << " of " << N << endl;
			if (!theLearningCurvePathBase.empty())
				theTrainer->learningCurveFile(makeMemberPath(theLearningCurvePathBase, i + 1));
			if (!theLearningCurvePathBase.empty() || theTrainer->earlyStoppingPatience() > 0)
				theTrainer->validationData(&samples[i].second);
			trained[i] = theTrainer->trainNew(samples[i].first, cout);
			sessions[i] = Session(std::make_unique<Ensemble>(*trained[i], 1),
			                      std::make_unique<DataSet>(samples[i].first),
			                      std::make_unique<DataSet>(samples[i].second));
		}
	}

	for (int i = 0; i < N; ++i)
		ensemble->addMlp(*trained[i]);
	theSessions = std::move(sessions);
	return ensemble;
}

// PROTECTED
EnsembleBuilder::EnsembleBuilder() : theTrainer(nullptr), theSampler(nullptr), theSessions() {}

EnsembleBuilder::~EnsembleBuilder() = default;

bool EnsembleBuilder::isValid() const {
	return theTrainer && theTrainer->isValid() && theSampler;
}

// PRIVATE
