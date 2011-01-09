/*$Id: Ensemble.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Ensemble.hh"
#include "matrixtools/MatrixTools.hh"

#include <cassert>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace MatrixTools;
using std::vector;

Ensemble::Ensemble():theEnsemble(0), theScales(0){}

Ensemble::Ensemble(Mlp& mlp, double s):theEnsemble(1, new Mlp(mlp)), theScales(1,s){}

Ensemble::Ensemble(const Ensemble& c){*this=c;}

Ensemble::~Ensemble()
{
	vector<Mlp*>::iterator it;
	for(it = theEnsemble.begin(); it != theEnsemble.end(); ++it)
		delete *it;
	theEnsemble.clear();
}

Ensemble& Ensemble::operator=(const Ensemble& c)
{
	if(this!=&c){
		for(uint i=0; i<c.theEnsemble.size(); ++i)
			theEnsemble.push_back(new Mlp(*(c.theEnsemble[i])));
		theScales=c.theScales;
	}
	return *this;
}

Mlp& Ensemble::operator[](const uint i)
{return mlp(i);}

Mlp& Ensemble::mlp(const uint i)
{
	assert(i<theEnsemble.size());
	return *(theEnsemble[i]);
}

void Ensemble::delMlp(const uint i)
{
	//I may need to think more here.
	assert(i<theEnsemble.size());
	delete theEnsemble[i];
	theEnsemble.erase(theEnsemble.begin()+i);
	theScales.erase(theScales.begin()+i);
}

void Ensemble::addMlp(Mlp& mlp, double s)
{
	theEnsemble.push_back(new Mlp(mlp));
	theScales.push_back(s);
}

void Ensemble::addMlp(Mlp& mlp)
{
	theEnsemble.push_back(new Mlp(mlp));
	theScales.assign(theEnsemble.size(), 1.0/theEnsemble.size());
}

void Ensemble::addMlp(Mlp* mlp, double s)
{
	theEnsemble.push_back(mlp);
	theScales.push_back(s);
}

void Ensemble::addMlp(Mlp* mlp)
{
	theEnsemble.push_back(mlp);
	theScales.assign(theEnsemble.size(), 1.0/theEnsemble.size());
}

double Ensemble::scale(const uint i) const
{
	assert(i<theScales.size());
	return theScales[i];
}

void Ensemble::scale(const uint i, double s)
{
	assert(i<theScales.size());
	theScales[i]=s;
}

uint Ensemble::size() const {return theEnsemble.size();}

vector<double> Ensemble::propagate(vector<double>& input)
{
	assert(theEnsemble.size() == theScales.size());
	assert(!theEnsemble.empty());
	vector<Mlp*>::iterator itm = theEnsemble.begin();
	vector<double>::iterator its = theScales.begin();
	vector<double> output = (*itm)->propagate(input); ++itm;
	mul(output, *its++);
	for(; itm != theEnsemble.end(); ++itm){
		vector<double> tmp = (*itm)->propagate(input);
		mul(tmp, *its++);
		add(output, tmp, output);
	}
	return output;
}

