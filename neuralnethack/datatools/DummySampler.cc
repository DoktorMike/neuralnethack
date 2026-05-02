#include "DummySampler.hh"

#include <iostream>
#include <cassert>

using namespace DataTools;
using namespace std;

// PUBLIC
DummySampler::DummySampler(DataSet& data, const uint numSplits)
    : Sampler(data), n(numSplits), index(0) {}

DummySampler::DummySampler(const DummySampler& bs) : Sampler(*(bs.theData)) {
	*this = bs;
}

DummySampler::~DummySampler() {}

DummySampler& DummySampler::operator=(const DummySampler& bs) {
	if (this != &bs) {
		Sampler::operator=(bs);
		n = bs.n;
	}
	return *this;
}

pair<DataSet, DataSet> DummySampler::next() {
	++index;
	DataSet valData = *theData;
	valData.indices().clear();
	return {*theData, std::move(valData)};
}

void DummySampler::reset() {
	index = 0;
	theDataManager = std::make_unique<DataManager>();
}

bool DummySampler::hasNext() const {
	return (index < n) ? true : false;
}

uint DummySampler::howMany() const {
	return n;
}

uint DummySampler::numRuns() const {
	return n;
}

void DummySampler::numRuns(uint n) {
	this->n = n;
}

// PRIVATE
