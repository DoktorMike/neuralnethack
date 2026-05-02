#include "CrossSplitSampler.hh"

#include <ostream>
#include <cassert>

using namespace DataTools;
using namespace std;

// PUBLIC
CrossSplitSampler::CrossSplitSampler(DataSet& data, const uint numRuns, const uint numParts)
    : Sampler(data), n(numRuns), k(numParts), index(0), runCntr(0) {
	assert(n && k);
	if (k < 2) {
		cerr << "Warning: Can't do cross validation on 1 part. Resetting to 2." << endl;
		k = 2;
	}
	theSplits = std::make_unique<std::vector<DataSet>>(theDataManager->split(data, k));
}

CrossSplitSampler::CrossSplitSampler(const CrossSplitSampler& cv) : Sampler(*(cv.theData)) {
	*this = cv;
}

CrossSplitSampler::~CrossSplitSampler() {}

CrossSplitSampler& CrossSplitSampler::operator=(const CrossSplitSampler& cv) {
	if (this != &cv) {
		Sampler::operator=(cv);
		n = cv.n;
		k = cv.k;
		index = cv.index;
		runCntr = cv.runCntr;
	}
	return *this;
}

pair<DataSet, DataSet> CrossSplitSampler::next() {
	if (index >= k) {
		if (++runCntr >= n) {
			return {};
		} else {
			uint tmp = runCntr;
			this->reset();
			runCntr = tmp;
		}
	}
	index++;
	DataSet valData = theSplits->front();
	theSplits->erase(theSplits->begin());
	DataSet trnData = theDataManager->join(*theSplits);
	theSplits->push_back(valData);
	return {std::move(trnData), std::move(valData)};
}

void CrossSplitSampler::reset() {
	index = 0;
	runCntr = 0;
	theDataManager = std::make_unique<DataManager>();
	theSplits = std::make_unique<std::vector<DataSet>>(theDataManager->split(*theData, k));
}

bool CrossSplitSampler::hasNext() const {
	if (index >= k) {
		if (runCntr >= n - 1)
			return false;
		else
			return true;
	} else
		return true;
}

uint CrossSplitSampler::howMany() const {
	return n * k;
}

// PRIVATE
