#include "CoreDataSet.hh"

using namespace DataTools;

CoreDataSet::CoreDataSet()
	:nLeft(0), patterns(vector<Pattern>(0)), itp(patterns.begin())
{}

CoreDataSet::CoreDataSet(const CoreDataSet& coreDataSet)
{*this = coreDataSet;}

CoreDataSet::~CoreDataSet()
{}

CoreDataSet& CoreDataSet::operator=(const CoreDataSet& coreDataSet)
{
	if(this != &coreDataSet){
		this->nLeft = coreDataSet.nLeft;
		this->patterns = coreDataSet.patterns;
		this->itp = coreDataSet.itp;
	}
	return *this;
}

int CoreDataSet::remaining() const {return nLeft;}

Pattern& CoreDataSet::nextPattern()
{
	if(++itp < patterns.end()){
		nLeft--;
		return *itp;
	}
	itp = patterns.begin();
	nLeft = patterns.size() - 1;
	return *itp;
}

Pattern& CoreDataSet::currentPattern()
{return *itp;}

Pattern& CoreDataSet::previousPattern()
{
	if( (--itp) > patterns.begin() ){
		nLeft++;
		return *itp;
	}
	reset();
	return *itp;
}

Pattern& CoreDataSet::pattern(uint index)
{
	itp = patterns.begin() + index;
	nLeft = patterns.size() - 1 - index;
	return *itp;
}

vector<Pattern>& CoreDataSet::patternVector()
{return patterns;}

void CoreDataSet::addPattern(const Pattern& pattern)
{
	patterns.push_back(pattern);
	reset();
}

uint CoreDataSet::nInput() const
{
	assert(patterns.size()>0);
	return patterns[0].nInput();
}

uint CoreDataSet::nOutput() const
{
	assert(patterns.size()>0);
	return patterns[0].nOutput();
}

void CoreDataSet::reset()
{
	itp = patterns.end()-1;
	nLeft = patterns.size();
}

uint CoreDataSet::size() const
{return patterns.size();}

void CoreDataSet::print(ostream& os)
{
	assert(os);
	vector<Pattern>::iterator itp;
	for(itp=patterns.begin(); itp!=patterns.end(); itp++)
		(*itp).print(os);
}


