#include "HoldOutSampler.hh"

#include <iostream>
#include <cassert>

using namespace DataTools;
using namespace std;

// PUBLIC
HoldOutSampler::HoldOutSampler(DataSet& data, const double rat, const uint numSplits)
    : Sampler(data), ratio(rat), n(numSplits), index(0) {}

HoldOutSampler::HoldOutSampler(const HoldOutSampler& ho) : Sampler(*(ho.theData)) {
	*this = ho;
}

HoldOutSampler::~HoldOutSampler() {}

HoldOutSampler& HoldOutSampler::operator=(const HoldOutSampler& ho) {
	if (this != &ho) {
		Sampler::operator=(ho);
		ratio = ho.ratio;
		n = ho.n;
	}
	return *this;
}

pair<DataSet, DataSet>* HoldOutSampler::next() {
	++index;
	return theDataManager->split(*theData, ratio);
	// return new pair<DataSet, DataSet>(tmp.first, tmp.second);
}

void HoldOutSampler::reset() {
	index = 0;
	if (theDataManager != 0) delete theDataManager;
	theDataManager = new DataManager();
}

bool HoldOutSampler::hasNext() const {
	return (index < n) ? true : false;
}

uint HoldOutSampler::howMany() const {
	return n;
}

uint HoldOutSampler::numRuns() const {
	return n;
}

void HoldOutSampler::numRuns(uint n) {
	this->n = n;
}

// PRIVATE
