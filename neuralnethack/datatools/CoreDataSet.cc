/*$Id: CoreDataSet.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "CoreDataSet.hh"

#include <cassert>
#include <functional>
#include <algorithm>

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

void CoreDataSet::print(ostream& os) const
{
	assert(os);
	//for_each(patterns.begin(), patterns.end(), bind2nd(mem_fun_ref(&Pattern::print), os));
	vector<Pattern>::const_iterator itp;
	for(itp=patterns.begin(); itp!=patterns.end(); itp++)
		(*itp).print(os);
}


