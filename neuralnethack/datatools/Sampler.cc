/*$Id: Sampler.cc 1605 2007-01-24 20:47:36Z michael $*/

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


#include "Sampler.hh"

#include <cassert>
#include <ostream>
#include <vector>
#include <sstream>

using namespace DataTools;

using std::ostream;
using std::vector;
using std::ostringstream;

//PUBLIC

DataSet* Sampler::data() const {return theData;}
void Sampler::data(DataSet* d){theData = d;}

bool Sampler::randomSampling() const
{
	assert(theDataManager != 0); 
	return theDataManager->random();
}

void Sampler::randomSampling(const bool rs)
{
	assert(theDataManager != 0); 
	theDataManager->random(rs);
}

//PROTECTED

Sampler::Sampler(DataSet& data):theDataManager(new DataManager()), theData(&data), theSplits(0)
{}

Sampler::Sampler(const Sampler& s){*this = s;}

Sampler::~Sampler()
{
	delete theDataManager;
	if(theSplits != 0){
		theSplits->clear();
		delete theSplits;
	}
}

Sampler& Sampler::operator=(const Sampler& s)
{
	if(this != &s){
		theDataManager = new DataManager(*s.theDataManager);
		theData = s.theData;
		theSplits = new vector<DataSet>(s.theSplits->begin(), s.theSplits->end());
	}
	return *this;
}

//PRIVATE

