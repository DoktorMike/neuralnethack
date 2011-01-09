#include "DataSet.hh"

#include <cassert>

using namespace DataTools;
using namespace std;

DataSet::DataSet()
	:theIndices(vector<uint>(0)), 
	itp(theIndices.begin()), theCoreDataSet(0)
{}

DataSet::DataSet(const DataSet& dataSet)
{*this = dataSet;}

DataSet::~DataSet()
{}

DataSet& DataSet::operator=(const DataSet& dataSet)
{
	if(this != &dataSet){
		this->theIndices = dataSet.theIndices;
		this->itp = dataSet.itp;
		this->theCoreDataSet = dataSet.theCoreDataSet;
	}
	return *this;
}

Pattern& DataSet::pattern(uint index)
{
	assert(index < theIndices.size());
	return theCoreDataSet->pattern(theIndices[index]);
}

vector<uint>& DataSet::indices()
{
	return theIndices;
}

void DataSet::indices(vector<uint>& i)
{
	assert(theCoreDataSet != 0 && theCoreDataSet->size() >= i.size());
	theIndices.assign(i.begin(), i.end());
}

CoreDataSet& DataSet::coreDataSet()
{
	return *theCoreDataSet;
}

void DataSet::coreDataSet(CoreDataSet& cds)
{
	theCoreDataSet = &cds;
	theIndices = vector<uint>(theCoreDataSet->size(),0);
	for(uint i = 0; i < theIndices.size(); ++i)
		theIndices.at(i) = i;
}

uint DataSet::nInput() const
{
	assert(theCoreDataSet != 0);
	return theCoreDataSet->nInput();
}

uint DataSet::nOutput() const
{
	assert(theCoreDataSet != 0);
	return theCoreDataSet->nOutput();
}

uint DataSet::size() const
{return theIndices.size();}

void DataSet::print(ostream& os)
{
	assert(os);
	vector<uint>::iterator itp;
	for(itp=theIndices.begin(); itp!=theIndices.end(); ++itp)
		theCoreDataSet->pattern(*itp).print(os);
}


