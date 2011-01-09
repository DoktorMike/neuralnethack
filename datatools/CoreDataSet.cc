#include "CoreDataSet.hh"

#include <cassert>

using namespace DataTools;
using namespace std;

CoreDataSet::CoreDataSet()
	:patterns(vector<Pattern>(0)), itp(patterns.begin()){}

CoreDataSet::CoreDataSet(const CoreDataSet& coreDataSet)
{*this = coreDataSet;}

CoreDataSet::~CoreDataSet(){}

CoreDataSet& CoreDataSet::operator=(const CoreDataSet& coreDataSet)
{
	if(this != &coreDataSet){
		this->patterns = coreDataSet.patterns;
		this->itp = coreDataSet.itp;
	}
	return *this;
}

Pattern& CoreDataSet::pattern(uint index)
{
	assert(index < patterns.size());
	return patterns[index];
}

vector<Pattern>& CoreDataSet::patternVector()
{return patterns;}

void CoreDataSet::addPattern(const Pattern& pattern)
{
	patterns.push_back(pattern);
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

uint CoreDataSet::size() const
{return patterns.size();}

void CoreDataSet::print(ostream& os)
{
	assert(os);
	vector<Pattern>::iterator itp;
	for(itp=patterns.begin(); itp!=patterns.end(); itp++)
		(*itp).print(os);
}


