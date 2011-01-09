/*$Id: Evaluator.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Evaluator.hh"

#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>

using namespace EvalTools;
using namespace std;

Evaluator::Evaluator():theTnf(0), theTpf(0), theCut(0), nTp(0), nTn(0), nP(0), nN(0)
{}

Evaluator::Evaluator(const Evaluator& eval){*this=eval;}

Evaluator::~Evaluator(){}

Evaluator& Evaluator::operator=(const Evaluator& eval)
{
	if(this!=&eval){
		theTnf=eval.theTnf;
		theTpf=eval.theTpf;
		theCut=eval.theCut;
		nTp=eval.nTp;
		nTn=eval.nTn;
		nP=eval.nP;
		nN=eval.nN;
	}
	return *this;
}

double Evaluator::tpf(){return theTpf;}

double Evaluator::fnf(){return 1.0 - theTpf;}

double Evaluator::tnf(){return theTnf;}

double Evaluator::fpf(){return 1.0 - theTnf;}

double Evaluator::cut(){return theCut;}

void Evaluator::cut(double c){theCut=c;}

void Evaluator::evaluate(vector<double>& out, vector<uint>& dout)
{
	assert(out.size()==dout.size());
	reset();
	vector<uint> o=cutOutput(out);
	vector<uint>::iterator ito=o.begin();
	vector<uint>::iterator itd=dout.begin();

	for(; itd!=dout.end(); ++itd, ++ito)
		switch(*itd){
			case NEG:
				++nN;
				if(*ito==*itd)
					++nTn;
				break;
			case POS:
				++nP;
				if(*ito==*itd)
					++nTp;
				break;
		}
	calcRates();
}

void Evaluator::print(ostream& os)
{
	if(!os){
		cerr<<"Output stream error.";
		abort();
	}
	os<<"\tTrue Positive Fraction: "<<theTpf<<endl;
	os<<"\tTrue Negative Fraction: "<<theTnf<<endl;
}

//PRIVATE--------------------------------------------------------------------//

void Evaluator::reset()
{
	nTp=0; 
	nTn=0; 
	nP=0;
	nN=0;
}

vector<uint> Evaluator::cutOutput(vector<double>& out)
{
	vector<uint> tmp(0);
	vector<double>::iterator it;
	for(it=out.begin(); it!=out.end(); ++it){
		if(*it<theCut)
			tmp.push_back(NEG);
		else
			tmp.push_back(POS);
	}
	assert(tmp.size()==out.size());
	return tmp;
}

vector<uint> Evaluator::vectorDoubleToUint(vector<double>& vec)
{
	vector<uint> tmp(0);
	vector<double>::iterator it;
	for(it=vec.begin(); it!=vec.end(); ++it)
		tmp.push_back((uint)round(*it));
	return tmp;
}

void Evaluator::calcRates()
{
	theTnf=nTn/(double)nN;
	theTpf=nTp/(double)nP;
}

