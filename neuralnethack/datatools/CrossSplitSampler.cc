/*$Id: CrossSplitSampler.cc 1678 2007-10-01 14:42:23Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#include "CrossSplitSampler.hh"

#include <ostream>
#include <cassert>
#include <sstream>
#include <fstream>

using namespace DataTools;
using namespace std;

//PUBLIC
CrossSplitSampler::CrossSplitSampler(DataSet& data, const uint numRuns, 
		const uint numParts):Sampler(data), n(numRuns), k(numParts), index(0), runCntr(0)
{
	assert(n && k);
	if(k < 2){
		cerr<<"Warning: Can't do cross validation on 1 part. Resetting to 2."<<endl;
		k = 2;
	}
	theSplits = theDataManager->split(data, k);
} 

CrossSplitSampler::CrossSplitSampler(const CrossSplitSampler& cv):Sampler(*(cv.theData)) { *this = cv; }

CrossSplitSampler::~CrossSplitSampler()
{}

CrossSplitSampler& CrossSplitSampler::operator=(const CrossSplitSampler& cv)
{
	if(this != &cv){
		Sampler::operator=(cv);
		n = cv.n;
		k = cv.k;
		index = cv.index;
		runCntr = cv.runCntr;
	}
	return *this;
}

pair<DataSet, DataSet>* CrossSplitSampler::next()
{
	//cout<<"BEFORE: runCntr: "<<runCntr<<" index: "<<index<<endl;
	if(index >= k){
		if(++runCntr >= n){
			return 0;
		}else{
			//cout<<"Resetting"<<endl;
			uint tmp = runCntr;
			this->reset();
			runCntr = tmp;
		}
	}
	index++;
	//cout<<"AFTER: runCntr: "<<runCntr<<" index: "<<index<<endl;
	DataSet valData = theSplits->front();
	theSplits->erase(theSplits->begin());
	DataSet trnData = theDataManager->join(*theSplits);
	theSplits->push_back(valData);
	return new pair<DataSet, DataSet>(trnData, valData);
}

void CrossSplitSampler::reset()
{
	index = 0;
	runCntr = 0;
	if(theDataManager != 0) delete theDataManager;
	if(theSplits != 0){
		theSplits->clear();
		delete theSplits;
	}
	theDataManager = new DataManager();
	theSplits = theDataManager->split(*theData, k);
}

bool CrossSplitSampler::hasNext() const 
{
	if(index >= k){
		if(runCntr >= n - 1) return false; 
		else return true;
	}else return true;
}

uint CrossSplitSampler::howMany() const {return n*k;}

//PRIVATE

