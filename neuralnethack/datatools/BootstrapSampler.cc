#include "BootstrapSampler.hh"

#include <iostream>
#include <cassert>

using namespace DataTools;
using namespace std;

//PUBLIC
BootstrapSampler::BootstrapSampler(DataSet& data, const uint numSplits):Sampler(data), n(numSplits), index(0)
{}

BootstrapSampler::BootstrapSampler(const BootstrapSampler& bs):Sampler(*(bs.theData)) { *this = bs; }

BootstrapSampler::~BootstrapSampler()
{
}

BootstrapSampler& BootstrapSampler::operator=(const BootstrapSampler& bs)
{
	if(this != &bs){
		Sampler::operator=(bs);
		n = bs.n;
	}
	return *this;
}

pair<DataSet, DataSet>* BootstrapSampler::next()
{
	++index;
	return theDataManager->split(*theData);
	//return new pair<DataSet, DataSet>(tmp.first, tmp.second);
}

void BootstrapSampler::reset()
{
	index = 0;
	if(theDataManager != 0) delete theDataManager;
	theDataManager = new DataManager();
}

bool BootstrapSampler::hasNext() const {return (index < n) ? true : false;}

uint BootstrapSampler::howMany() const {return n;}

uint BootstrapSampler::numRuns() const {return n;}

void BootstrapSampler::numRuns(uint n) {this->n = n;}

//PRIVATE

