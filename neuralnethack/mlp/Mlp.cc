/*$Id: Mlp.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Mlp.hh"
#include "SigmoidLayer.hh"
#include "TanHypLayer.hh"
#include "LinearLayer.hh"

#include <cassert>

using namespace MultiLayerPerceptron;
using namespace std;

Mlp::Mlp(const vector<uint>& a, const vector<string>& t, bool s):
	theArch(a), theTypes(t), theSoftmax(s)
{
	theLayers = vector<Layer*>(0);
	createLayers();
}

Mlp::Mlp(const MlpModel& mlpmodel):
	theArch(mlpmodel.architecture), theTypes(mlpmodel.types), theSoftmax(mlpmodel.softmax)
{
	theLayers = vector<Layer*>(0);
	createLayers();
}

Mlp::Mlp(const Mlp& mlp)
{*this = mlp;}

Mlp::~Mlp()
{
	for(vector<Layer*>::iterator it = theLayers.begin(); it != theLayers.end(); ++it) 
		delete *it;
	theLayers.clear();
}

Mlp& Mlp::operator=(const Mlp& mlp)
{
	if(this != &mlp){
		theArch = mlp.theArch;
		theTypes = mlp.theTypes;
		theSoftmax = mlp.theSoftmax;
		theLayers = vector<Layer*>(0);
		createLayers();
		assert(theLayers.size()==mlp.theLayers.size());
		for(uint i=0; i<theTypes.size(); ++i)
			*(theLayers[i]) = *(mlp.theLayers[i]);
	}
	return *this;
}


Layer& Mlp::operator[](const uint i)
{
	assert(i < theLayers.size());
	return *(theLayers[i]);
}

vector<double> Mlp::weights() const
{
	vector<double> w(0);
	vector<Layer*>::const_iterator it;

	for(it=theLayers.begin(); it!=theLayers.end(); ++it){
		vector<double>& tmp = (*it)->weights();
		w.insert(w.end(),tmp.begin(), tmp.end());
	}
	return w;
}

void Mlp::weights(vector<double>& w)
{
	assert(w.size()==nWeights());
	vector<Layer*>::iterator itl;
	vector<double>::iterator itw = w.begin();
	for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl){
		vector<double>& tmp = (*itl)->weights();
		vector<double>::iterator ittmp = tmp.begin();
		for(; ittmp != tmp.end(); ++ittmp, ++itw)
			*ittmp = *itw;
	}
}

vector<double> Mlp::gradients() const
{
	vector<double> g(0);
	g.reserve(nWeights());
	vector<Layer*>::const_iterator it;

	for(it=theLayers.begin(); it!=theLayers.end(); ++it){
		vector<double>& tmp = (*it)->gradients();
		g.insert(g.end(),tmp.begin(),tmp.end());
	}
	return g;
}

void Mlp::gradients(vector<double>& g)
{
	assert(g.size()==nWeights());
	vector<Layer*>::iterator itl = theLayers.begin();
	vector<double>::iterator itg = g.begin();
	while(itl != theLayers.end()){
		vector<double>& tmp = (*itl)->gradients();
		vector<double>::iterator ittmp = tmp.begin();
		for(; ittmp != tmp.end(); ++ittmp, ++itg) *ittmp = *itg;
		++itl;
	}
}

Layer& Mlp::layer(uint index)
{
	assert(index < theLayers.size());
	return *(theLayers[index]);
}

uint Mlp::nLayers() const
{return theLayers.size();}

uint Mlp::nWeights() const
{
	vector<Layer*>::const_iterator it;
	uint tmp=0;
	for(it=theLayers.begin(); it!=theLayers.end(); ++it)
		tmp+=(*it)->nWeights();
	return tmp;
}

uint Mlp::size() const
{return nLayers();}

void Mlp::regenerateWeights()
{
	vector<Layer*>::iterator itl;
	for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
		(*itl)->regenerateWeights();
}

vector<double>& Mlp::propagate(vector<double>& input)
{
	vector<Layer*>::iterator it; 
	vector<double>* inOut = &input;

	for(it=theLayers.begin(); it!=theLayers.end(); ++it)
		inOut = &((*it)->propagate(*inOut));
	return *inOut;
}

void Mlp::printWeights(ostream& os) const
{
	vector<Layer*>::const_iterator itl;
	for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
		(*itl)->printWeights(os);
}

void Mlp::printGradients(ostream& os) const
{
	vector<Layer*>::const_iterator itl;
	for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
		(*itl)->printGradients(os);
}

//PRIVATE--------------------------------------------------------------------//

void Mlp::createLayers()
{
	vector<uint>::iterator it;
	int i=0;

	for(it=theArch.begin()+1; it!=theArch.end(); ++it, ++i){
		string t = theTypes.at(i);
		Layer* l = 0;
		if(t == SIGMOID)
			l = new SigmoidLayer(*(it),*(it-1));
		else if(t == TANHYP)
			l = new TanHypLayer(*(it),*(it-1));
		else if(t == LINEAR)
			l = new LinearLayer(*(it),*(it-1));
		theLayers.push_back(l);	
	}
}

