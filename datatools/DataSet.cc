#include "DataSet.hh"

using namespace DataTools;

DataSet::DataSet()
	:nLeft(0), theIndices(vector<uint>(0)), 
	itp(theIndices.begin()), theCoreDataSet(0)
{}

DataSet::DataSet(const DataSet& dataSet)
{*this = dataSet;}

DataSet::~DataSet()
{}

DataSet& DataSet::operator=(const DataSet& dataSet)
{
	if(this != &dataSet){
		this->nLeft = dataSet.nLeft;
		this->theIndices = dataSet.theIndices;
		this->itp = dataSet.itp;
		this->theCoreDataSet = dataSet.theCoreDataSet;
	}
	return *this;
}

int DataSet::remaining() const {return nLeft;}

Pattern& DataSet::nextPattern()
{
	if(++itp < theIndices.end()){
		nLeft--;
		return theCoreDataSet->pattern(*itp);
	}
	itp = theIndices.begin();
	nLeft = theIndices.size() - 1;
	return theCoreDataSet->pattern(*itp);
}

Pattern& DataSet::currentPattern()
{return theCoreDataSet->pattern(*itp);}

Pattern& DataSet::previousPattern()
{
	if( (--itp) > theIndices.begin() ){
		nLeft++;
		return theCoreDataSet->pattern(*itp);
	}
	reset();
	return theCoreDataSet->pattern(*itp);
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
	reset();
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

void DataSet::reset()
{
	itp = theIndices.end()-1;
	nLeft = theIndices.size();
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


