/*$Id: DataSet.cc 1623 2007-05-08 08:30:14Z michael $*/

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

void DataSet::print(ostream& os) const
{
	assert(os);
	vector<uint>::const_iterator itp;
	for(itp=theIndices.begin(); itp!=theIndices.end(); ++itp)
		theCoreDataSet->pattern(*itp).print(os);
}


