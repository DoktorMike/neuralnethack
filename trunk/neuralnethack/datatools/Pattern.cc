/*$Id: Pattern.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Pattern.hh"

#include <cassert>

using namespace DataTools;
using namespace std;

Pattern::Pattern(string id, vector<double>& in, vector<double>& out)
	:id(id), in(in), out(out){}

Pattern::Pattern(){}

Pattern::Pattern(const Pattern& pattern)
{*this = pattern;}

Pattern::~Pattern(){}

Pattern& Pattern::operator=(const Pattern& pattern)
{
	if(this != &pattern){
		this->id = pattern.id;
		this->in = pattern.in;
		this->out = pattern.out;
	}
	return *this;
}

vector<double>& Pattern::input()
{return in;}

void Pattern::input(vector<double>& in)
{this->in = in;}

uint Pattern::nInput() const 
{return in.size();}

vector<double>& Pattern::output()
{return out;}

void Pattern::output(vector<double>& out)
{this->out = out;}

uint Pattern::nOutput() const
{return out.size();}

void Pattern::print(ostream& os) const
{
	assert(os);
	vector<double>::const_iterator it;
	os.setf(ios::ios_base::fixed, ios::ios_base::floatfield);
	os.setf(ios::ios_base::right, ios::ios_base::adjustfield);
	for(it=in.begin(); it!=in.end(); it++){
		os.width(14);
		os<<*it<<"\t";
	}
	for(it=out.begin(); it!=out.end(); it++){
		os.width(14);
		os<<*it<<"\t";
	}
	os<<id;
	os<<endl;
}
