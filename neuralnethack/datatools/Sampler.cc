#include "Sampler.hh"

#include <cassert>
#include <ostream>
#include <vector>
#include <sstream>

using namespace DataTools;

using std::ostream;
using std::ostringstream;
using std::vector;

// PUBLIC

DataSet* Sampler::data() const {
	return theData;
}
void Sampler::data(DataSet* d) {
	theData = d;
}

bool Sampler::randomSampling() const {
	assert(theDataManager != 0);
	return theDataManager->random();
}

void Sampler::randomSampling(const bool rs) {
	assert(theDataManager != 0);
	theDataManager->random(rs);
}

// PROTECTED

Sampler::Sampler(DataSet& data) : theDataManager(new DataManager()), theData(&data), theSplits(0) {}

Sampler::Sampler(const Sampler& s) {
	*this = s;
}

Sampler::~Sampler() {
	delete theDataManager;
	if (theSplits != 0) {
		theSplits->clear();
		delete theSplits;
	}
}

Sampler& Sampler::operator=(const Sampler& s) {
	if (this != &s) {
		theDataManager = new DataManager(*s.theDataManager);
		theData = s.theData;
		theSplits = new vector<DataSet>(s.theSplits->begin(), s.theSplits->end());
	}
	return *this;
}

// PRIVATE
