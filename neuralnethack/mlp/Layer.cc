/*$Id: Layer.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Layer.hh"

#include <iostream>
#include <cassert>

#include <algorithm>
#include <iterator>
#include <numeric>

using namespace MultiLayerPerceptron;
using namespace std;

Layer::Layer(const uint nc, const uint np, const string t):
	ncurr(nc),
	nprev(np),
	theType(t),
	theWeights(ncurr*(nprev+1), 0),
	theOutputs(ncurr,0),
	theLocalGradients(ncurr,0),
	theGradients(ncurr*(nprev+1), 0),
	theWeightUpdates(ncurr*(nprev+1), 0)
{
	regenerateWeights();
}

Layer::Layer(const Layer& layer)
{*this = layer;}

Layer::~Layer()
{
}

Layer& Layer::operator=(const Layer& layer)
{
	if(this != &layer){
		ncurr=layer.ncurr;
		nprev=layer.nprev;
		theType=layer.theType;
		theWeights=layer.theWeights;
		theOutputs=layer.theOutputs;
		theLocalGradients=layer.theLocalGradients;
		theGradients=layer.theGradients;
		theWeightUpdates=layer.theWeightUpdates;
	}
	return *this;
}

double& Layer::operator[](const uint i)
{
	assert(i < theOutputs.size());
	return theOutputs[i];
}

//ACCESSOR FUNCTIONS

//PRINTS

void Layer::printWeights(ostream& os) const
{ copy(theWeights.begin(), theWeights.end(), ostream_iterator<double>(os, " ")); }

void Layer::printGradients(ostream& os) const
{ copy(theGradients.begin(), theGradients.end(), ostream_iterator<double>(os, " ")); }

//UTILS

void Layer::regenerateWeights()
{ for_each(theWeights.begin(), theWeights.end(), newRand<double>()); }

vector<double> Layer::calcLifs(const vector<double>& input)
{
	vector<double> lif(ncurr, 0);
	vector<double>::iterator itw = theWeights.begin(), ito;
	for(ito = lif.begin(); ito != lif.end(); ++ito){
		*ito = inner_product(input.begin(), input.end(), itw, *(itw+input.size()));
		advance(itw, input.size()+1);
	}
	return lif;
}

vector<double>& Layer::propagate(const vector<double>& input)
{
	vector<double>::iterator itw = theWeights.begin(), ito;
	for(ito = theOutputs.begin(); ito != theOutputs.end(); ++ito){
		*ito = inner_product(input.begin(), input.end(), itw, *(itw+input.size()));
		advance(itw, input.size()+1);
		*ito = fire(*ito);
	}
	return theOutputs;
}



