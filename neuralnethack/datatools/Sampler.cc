#include "Sampler.hh"

#include <cassert>

using namespace DataTools;

DataSet* Sampler::data() const {
	return theData;
}
void Sampler::data(DataSet* d) {
	theData = d;
}

bool Sampler::randomSampling() const {
	assert(theDataManager);
	return theDataManager->random();
}

void Sampler::randomSampling(const bool rs) {
	assert(theDataManager);
	theDataManager->random(rs);
}

// PROTECTED

Sampler::Sampler(DataSet& data)
    : theDataManager(std::make_unique<DataManager>()), theData(&data), theSplits(nullptr) {}

Sampler::Sampler(const Sampler& s)
    : theDataManager(s.theDataManager ? std::make_unique<DataManager>(*s.theDataManager) : nullptr),
      theData(s.theData),
      theSplits(s.theSplits ? std::make_unique<std::vector<DataSet>>(*s.theSplits) : nullptr) {}

Sampler::~Sampler() = default;

Sampler& Sampler::operator=(const Sampler& s) {
	if (this != &s) {
		theDataManager =
		    s.theDataManager ? std::make_unique<DataManager>(*s.theDataManager) : nullptr;
		theData = s.theData;
		theSplits =
		    s.theSplits ? std::make_unique<std::vector<DataSet>>(*s.theSplits) : nullptr;
	}
	return *this;
}
