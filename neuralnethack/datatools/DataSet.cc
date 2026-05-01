#include "DataSet.hh"

#include <cassert>

using namespace DataTools;
using namespace std;

DataSet::DataSet() : theIndices(), itp(theIndices.begin()), theCoreDataSet(nullptr) {}

DataSet::DataSet(const DataSet& dataSet) {
	*this = dataSet;
}

DataSet::~DataSet() = default;

DataSet& DataSet::operator=(const DataSet& dataSet) {
	if (this != &dataSet) {
		this->theIndices = dataSet.theIndices;
		this->itp = theIndices.begin();
		this->theCoreDataSet = dataSet.theCoreDataSet;
	}
	return *this;
}

Pattern& DataSet::pattern(uint index) {
	assert(theCoreDataSet);
	assert(index < theIndices.size());
	return theCoreDataSet->pattern(theIndices[index]);
}

vector<uint>& DataSet::indices() {
	return theIndices;
}

void DataSet::indices(vector<uint>& i) {
	assert(theCoreDataSet && theCoreDataSet->size() >= i.size());
	theIndices.assign(i.begin(), i.end());
}

CoreDataSet& DataSet::coreDataSet() {
	assert(theCoreDataSet);
	return *theCoreDataSet;
}

void DataSet::coreDataSet(shared_ptr<CoreDataSet> cds) {
	assert(cds);
	theCoreDataSet = std::move(cds);
	theIndices.assign(theCoreDataSet->size(), 0);
	for (uint i = 0; i < theIndices.size(); ++i) theIndices[i] = i;
}

shared_ptr<CoreDataSet> DataSet::sharedCoreDataSet() const {
	return theCoreDataSet;
}

uint DataSet::nInput() const {
	assert(theCoreDataSet);
	return theCoreDataSet->nInput();
}

uint DataSet::nOutput() const {
	assert(theCoreDataSet);
	return theCoreDataSet->nOutput();
}

uint DataSet::size() const {
	return theIndices.size();
}

void DataSet::print(ostream& os) const {
	assert(os);
	for (uint idx : theIndices)
		theCoreDataSet->pattern(idx).print(os);
}
