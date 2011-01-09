/*$Id: DummySampler.cc 1605 2007-01-24 20:47:36Z michael $*/

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


#include "DummySampler.hh"

#include <iostream>
#include <cassert>

using namespace DataTools;
using namespace std;

//PUBLIC
DummySampler::DummySampler(DataSet& data, const uint numSplits):Sampler(data), n(numSplits), index(0)
{}

DummySampler::DummySampler(const DummySampler& bs):Sampler(*(bs.theData)) { *this = bs; }

DummySampler::~DummySampler()
{
}

DummySampler& DummySampler::operator=(const DummySampler& bs)
{
	if(this != &bs){
		Sampler::operator=(bs);
		n = bs.n;
	}
	return *this;
}

pair<DataSet, DataSet>* DummySampler::next()
{
	++index;
	//Make an empty copy of training data
	DataSet valData = *theData;
	valData.indices().clear();
	return new pair<DataSet, DataSet>(*theData, valData);
}

void DummySampler::reset()
{
	index = 0;
	if(theDataManager != 0) delete theDataManager;
	theDataManager = new DataManager();
}

bool DummySampler::hasNext() const {return (index < n) ? true : false;}

uint DummySampler::numRuns() const {return n;}

void DummySampler::numRuns(uint n) {this->n = n;}

//PRIVATE

