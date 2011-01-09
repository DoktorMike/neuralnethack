#include "DataSet.hh"

using namespace DataTools;

DataSet::DataSet():nLeft(0)
{

}

DataSet::DataSet(const DataSet& dataSet)
{*this = dataSet;}

DataSet::~DataSet()
{}

DataSet& DataSet::operator=(const DataSet& dataSet)
{
    if(this != &dataSet){
	this->nLeft = dataSet.nLeft;
	this->patterns = dataSet.patterns;
	this->itp = dataSet.itp;
    }
    return *this;
}

int DataSet::remaining(){return nLeft;}

Pattern& DataSet::nextPattern()
{
    if(++itp < patterns.end()){
	nLeft--;
	return *itp;
    }
    itp = patterns.begin();
    nLeft = patterns.size() - 1;
    return *itp;
}

Pattern& DataSet::currentPattern()
{return *itp;}

Pattern& DataSet::previousPattern()
{
    if( (--itp) > patterns.begin() ){
	nLeft++;
	return *itp;
    }
    reset();
    return *itp;
}

void DataSet::addPattern(const Pattern& pattern)
{
    patterns.push_back(pattern);
    reset();
}

uint DataSet::nInput()
{
    assert(patterns.size()>0);
    return patterns[0].nInput();
}

uint DataSet::nOutput()
{
    assert(patterns.size()>0);
    return patterns[0].nOutput();
}

void DataSet::reset()
{
    itp = patterns.end()-1;
    nLeft = patterns.size();
}

uint DataSet::size()
{return patterns.size();}

void DataSet::print(ostream& os)
{
    if(!os){
	cerr<<"Couldn't open stream for writing."<<endl;
	abort();
    }
    vector<Pattern>::iterator itp;
    for(itp=patterns.begin(); itp!=patterns.end(); itp++)
	(*itp).print(os);
}


