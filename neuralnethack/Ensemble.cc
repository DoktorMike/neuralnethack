/*$Id: Ensemble.cc 1684 2007-10-12 15:55:07Z michael $*/

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
using std::unique_ptr;
using std::make_unique;

Ensemble::Ensemble():theEnsemble(), theScales(){}

Ensemble::Ensemble(Mlp& mlp, double s):theEnsemble(), theScales(1,s)
{
	theEnsemble.push_back(make_unique<Mlp>(mlp));
}

Ensemble::Ensemble(const Ensemble& c)
	:theScales(c.theScales)
{
	theEnsemble.reserve(c.theEnsemble.size());
	for(const auto& m : c.theEnsemble)
		theEnsemble.push_back(make_unique<Mlp>(*m));
}

Ensemble::~Ensemble() = default;

Ensemble& Ensemble::operator=(const Ensemble& c)
{
	if(this!=&c){
		theEnsemble.clear();
		theEnsemble.reserve(c.theEnsemble.size());
		for(const auto& m : c.theEnsemble)
			theEnsemble.push_back(make_unique<Mlp>(*m));
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

const Mlp& Ensemble::mlp(const uint i) const
{
	assert(i<theEnsemble.size());
	return *(theEnsemble[i]);
}

void Ensemble::delMlp(const uint i)
{
	assert(i<theEnsemble.size());
	theEnsemble.erase(theEnsemble.begin()+i);
	theScales.erase(theScales.begin()+i);
}

void Ensemble::addMlp(Mlp& mlp, double s)
{
	theEnsemble.push_back(make_unique<Mlp>(mlp));
	theScales.push_back(s);
}

void Ensemble::addMlp(Mlp& mlp)
{
	theEnsemble.push_back(make_unique<Mlp>(mlp));
	theScales.assign(theEnsemble.size(), 1.0/theEnsemble.size());
}

void Ensemble::addMlp(unique_ptr<Mlp> mlp, double s)
{
	theEnsemble.push_back(std::move(mlp));
	theScales.push_back(s);
}

void Ensemble::addMlp(unique_ptr<Mlp> mlp)
{
	theEnsemble.push_back(std::move(mlp));
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

vector<double> Ensemble::propagate(const vector<double>& input)
{
	assert(theEnsemble.size() == theScales.size());
	assert(!theEnsemble.empty());
	auto itm = theEnsemble.begin();
	auto its = theScales.begin();
	vector<double> output = (*itm)->propagate(input); ++itm;
	mul(output, *its++);
	for(; itm != theEnsemble.end(); ++itm){
		vector<double> tmp = (*itm)->propagate(input);
		mul(tmp, *its++);
		add(output, tmp, output);
	}
	return output;
}

