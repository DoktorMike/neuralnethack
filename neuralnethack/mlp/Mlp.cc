#include "Mlp.hh"
#include "SigmoidLayer.hh"
#include "TanHypLayer.hh"
#include "LinearLayer.hh"
#include "ReLULayer.hh"
#include "LeakyReLULayer.hh"
#include "ELULayer.hh"

#include <cassert>
#include <iostream>

using namespace MultiLayerPerceptron;
using namespace std;

Mlp::Mlp(const vector<uint>& a, const vector<string>& t, bool s)
    : theArch(a), theTypes(t), theSoftmax(s) {
	createLayers();
	theSkipFrom.assign(theLayers.size(), -1);
}

Mlp::Mlp(const MlpModel& mlpmodel)
    : theArch(mlpmodel.architecture), theTypes(mlpmodel.types), theSoftmax(mlpmodel.softmax) {
	createLayers();
	theSkipFrom.assign(theLayers.size(), -1);
}

Mlp::Mlp(const Mlp& mlp)
    : theArch(mlp.theArch), theTypes(mlp.theTypes), theSoftmax(mlp.theSoftmax),
      theSkipFrom(mlp.theSkipFrom) {
	theLayers.reserve(mlp.theLayers.size());
	for (const auto& l : mlp.theLayers)
		theLayers.push_back(l->clone());
}

Mlp::~Mlp() = default;

Mlp& Mlp::operator=(const Mlp& mlp) {
	if (this != &mlp) {
		theArch = mlp.theArch;
		theTypes = mlp.theTypes;
		theSoftmax = mlp.theSoftmax;
		theSkipFrom = mlp.theSkipFrom;
		theLayers.clear();
		theLayers.reserve(mlp.theLayers.size());
		for (const auto& l : mlp.theLayers)
			theLayers.push_back(l->clone());
	}
	return *this;
}

Layer& Mlp::operator[](const uint i) {
	assert(i < theLayers.size());
	return *(theLayers[i]);
}

vector<double> Mlp::weights() const {
	vector<double> w;
	for (const auto& l : theLayers) {
		vector<double>& tmp = l->weights();
		w.insert(w.end(), tmp.begin(), tmp.end());
	}
	return w;
}

void Mlp::weights(vector<double>& w) {
	assert(w.size() == nWeights());
	auto itw = w.begin();
	for (auto& l : theLayers) {
		vector<double>& tmp = l->weights();
		for (auto ittmp = tmp.begin(); ittmp != tmp.end(); ++ittmp, ++itw)
			*ittmp = *itw;
	}
}

vector<double> Mlp::gradients() const {
	vector<double> g;
	g.reserve(nWeights());
	for (const auto& l : theLayers) {
		vector<double>& tmp = l->gradients();
		g.insert(g.end(), tmp.begin(), tmp.end());
	}
	return g;
}

void Mlp::gradients(vector<double>& g) {
	assert(g.size() == nWeights());
	auto itg = g.begin();
	for (auto& l : theLayers) {
		vector<double>& tmp = l->gradients();
		for (auto ittmp = tmp.begin(); ittmp != tmp.end(); ++ittmp, ++itg)
			*ittmp = *itg;
	}
}

Layer& Mlp::layer(uint index) {
	assert(index < theLayers.size());
	return *(theLayers[index]);
}

uint Mlp::nLayers() const {
	return theLayers.size();
}

uint Mlp::nWeights() const {
	uint tmp = 0;
	for (const auto& l : theLayers)
		tmp += l->nWeights();
	return tmp;
}

uint Mlp::size() const {
	return nLayers();
}

void Mlp::regenerateWeights() {
	for (auto& l : theLayers)
		l->regenerateWeights();
}

void Mlp::training(bool t) {
	for (auto& l : theLayers)
		l->training(t);
}

void Mlp::dropoutRate(double rate) {
	// Apply to hidden layers only, not the output layer
	for (uint i = 0; i + 1 < theLayers.size(); ++i)
		theLayers[i]->dropoutRate(rate);
}

void Mlp::normType(NormType nt) {
	// Apply to hidden layers only, not the output layer
	for (uint i = 0; i + 1 < theLayers.size(); ++i)
		theLayers[i]->normType(nt);
}

const vector<double>& Mlp::propagate(const vector<double>& input) {
	const vector<double>* inOut = &input;
	for (uint i = 0; i < theLayers.size(); ++i) {
		int src = theSkipFrom[i];
		const double* skipPtr = (src >= 0) ? theLayers[src]->outputs().data() : nullptr;
		inOut = &(theLayers[i]->propagate(*inOut, skipPtr));
	}
	return *inOut;
}

const double* Mlp::propagateBatch(const double* input, uint B) {
	const double* layerInput = input;
	uint n_in = theArch[0];
	for (uint i = 0; i < theLayers.size(); ++i) {
		int src = theSkipFrom[i];
		const double* skipPtr = (src >= 0) ? theLayers[src]->batchOutputs().data() : nullptr;
		layerInput = theLayers[i]->propagateBatch(layerInput, B, n_in, skipPtr);
		n_in = theLayers[i]->nNeurons();
	}
	return layerInput;
}

void Mlp::printWeights(ostream& os) const {
	for (const auto& l : theLayers)
		l->printWeights(os);
}

void Mlp::printGradients(ostream& os) const {
	for (const auto& l : theLayers)
		l->printGradients(os);
}

void Mlp::skipFrom(uint target, int source) {
	assert(target < theLayers.size());
	if (source < 0) {
		theSkipFrom[target] = -1;
		return;
	}
	uint s = static_cast<uint>(source);
	if (s >= target) {
		cerr << "Mlp::skipFrom: source " << s << " must be < target " << target << endl;
		abort();
	}
	if (theLayers[s]->nNeurons() != theLayers[target]->nNeurons()) {
		cerr << "Mlp::skipFrom: dim mismatch (source layer " << s << " has "
		     << theLayers[s]->nNeurons() << " neurons, target layer " << target << " has "
		     << theLayers[target]->nNeurons() << ")" << endl;
		abort();
	}
	theSkipFrom[target] = source;
}

int Mlp::skipFrom(uint target) const {
	assert(target < theSkipFrom.size());
	return theSkipFrom[target];
}

// PRIVATE--------------------------------------------------------------------//

void Mlp::createLayers() {
	auto it = theArch.begin();
	int i = 0;

	for (it = theArch.begin() + 1; it != theArch.end(); ++it, ++i) {
		string t = theTypes.at(i);
		if (t == SIGMOID)
			theLayers.push_back(make_unique<SigmoidLayer>(*(it), *(it - 1)));
		else if (t == TANHYP)
			theLayers.push_back(make_unique<TanHypLayer>(*(it), *(it - 1)));
		else if (t == LINEAR)
			theLayers.push_back(make_unique<LinearLayer>(*(it), *(it - 1)));
		else if (t == RELU)
			theLayers.push_back(make_unique<ReLULayer>(*(it), *(it - 1)));
		else if (t == LEAKYRELU)
			theLayers.push_back(make_unique<LeakyReLULayer>(*(it), *(it - 1)));
		else if (t == ELU_ACT)
			theLayers.push_back(make_unique<ELULayer>(*(it), *(it - 1)));
	}
}
