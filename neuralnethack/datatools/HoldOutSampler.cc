/*$Id: HoldOutSampler.cc 1678 2007-10-01 14:42:23Z michael $*/

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


#include "HoldOutSampler.hh"

#include <iostream>
#include <cassert>

using namespace DataTools;
using namespace std;

//PUBLIC
HoldOutSampler::HoldOutSampler(DataSet& data, const double rat, const uint numSplits):Sampler(data), ratio(rat), n(numSplits), index(0)
{}

HoldOutSampler::HoldOutSampler(const HoldOutSampler& ho):Sampler(*(ho.theData)) { *this = ho; }

HoldOutSampler::~HoldOutSampler()
{
}

HoldOutSampler& HoldOutSampler::operator=(const HoldOutSampler& ho)
{
	if(this != &ho){
		Sampler::operator=(ho);
		ratio = ho.ratio;
		n = ho.n;
	}
	return *this;
}

pair<DataSet, DataSet>* HoldOutSampler::next()
{
	++index;
	return theDataManager->split(*theData, ratio);
	//return new pair<DataSet, DataSet>(tmp.first, tmp.second);
}

void HoldOutSampler::reset()
{
	index = 0;
	if(theDataManager != 0) delete theDataManager;
	theDataManager = new DataManager();
}

bool HoldOutSampler::hasNext() const {return (index < n) ? true : false;}

uint HoldOutSampler::howMany() const {return n;}

uint HoldOutSampler::numRuns() const {return n;}

void HoldOutSampler::numRuns(uint n) {this->n = n;}

//PRIVATE

