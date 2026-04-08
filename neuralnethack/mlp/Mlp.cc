/*$Id: Mlp.cc 1684 2007-10-12 15:55:07Z michael $*/

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
#include "ReLULayer.hh"
#include "LeakyReLULayer.hh"
#include "ELULayer.hh"

#include <cassert>

using namespace MultiLayerPerceptron;
using namespace std;

Mlp::Mlp(const vector<uint>& a, const vector<string>& t, bool s):
	theArch(a), theTypes(t), theSoftmax(s)
{
	createLayers();
}

Mlp::Mlp(const MlpModel& mlpmodel):
	theArch(mlpmodel.architecture), theTypes(mlpmodel.types), theSoftmax(mlpmodel.softmax)
{
	createLayers();
}

Mlp::Mlp(const Mlp& mlp)
	:theArch(mlp.theArch), theTypes(mlp.theTypes), theSoftmax(mlp.theSoftmax)
{
	theLayers.reserve(mlp.theLayers.size());
	for(const auto& l : mlp.theLayers)
		theLayers.push_back(l->clone());
}

Mlp::~Mlp() = default;

Mlp& Mlp::operator=(const Mlp& mlp)
{
	if(this != &mlp){
		theArch = mlp.theArch;
		theTypes = mlp.theTypes;
		theSoftmax = mlp.theSoftmax;
		theLayers.clear();
		theLayers.reserve(mlp.theLayers.size());
		for(const auto& l : mlp.theLayers)
			theLayers.push_back(l->clone());
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
	vector<double> w;
	for(const auto& l : theLayers){
		vector<double>& tmp = l->weights();
		w.insert(w.end(), tmp.begin(), tmp.end());
	}
	return w;
}

void Mlp::weights(vector<double>& w)
{
	assert(w.size()==nWeights());
	auto itw = w.begin();
	for(auto& l : theLayers){
		vector<double>& tmp = l->weights();
		for(auto ittmp = tmp.begin(); ittmp != tmp.end(); ++ittmp, ++itw)
			*ittmp = *itw;
	}
}

vector<double> Mlp::gradients() const
{
	vector<double> g;
	g.reserve(nWeights());
	for(const auto& l : theLayers){
		vector<double>& tmp = l->gradients();
		g.insert(g.end(), tmp.begin(), tmp.end());
	}
	return g;
}

void Mlp::gradients(vector<double>& g)
{
	assert(g.size()==nWeights());
	auto itg = g.begin();
	for(auto& l : theLayers){
		vector<double>& tmp = l->gradients();
		for(auto ittmp = tmp.begin(); ittmp != tmp.end(); ++ittmp, ++itg)
			*ittmp = *itg;
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
	uint tmp=0;
	for(const auto& l : theLayers)
		tmp += l->nWeights();
	return tmp;
}

uint Mlp::size() const
{return nLayers();}

void Mlp::regenerateWeights()
{
	for(auto& l : theLayers)
		l->regenerateWeights();
}

void Mlp::training(bool t)
{
	for(auto& l : theLayers)
		l->training(t);
}

void Mlp::dropoutRate(double rate)
{
	// Apply to hidden layers only, not the output layer
	for(uint i = 0; i + 1 < theLayers.size(); ++i)
		theLayers[i]->dropoutRate(rate);
}

const vector<double>& Mlp::propagate(const vector<double>& input)
{
	const vector<double>* inOut = &input;
	for(auto& l : theLayers)
		inOut = &(l->propagate(*inOut));
	return *inOut;
}

const double* Mlp::propagateBatch(const double* input, uint B)
{
	const double* layerInput = input;
	uint n_in = theArch[0];
	for(auto& l : theLayers){
		layerInput = l->propagateBatch(layerInput, B, n_in);
		n_in = l->nNeurons();
	}
	return layerInput;
}

void Mlp::printWeights(ostream& os) const
{
	for(const auto& l : theLayers)
		l->printWeights(os);
}

void Mlp::printGradients(ostream& os) const
{
	for(const auto& l : theLayers)
		l->printGradients(os);
}

//PRIVATE--------------------------------------------------------------------//

void Mlp::createLayers()
{
	auto it = theArch.begin();
	int i=0;

	for(it=theArch.begin()+1; it!=theArch.end(); ++it, ++i){
		string t = theTypes.at(i);
		if(t == SIGMOID)
			theLayers.push_back(make_unique<SigmoidLayer>(*(it),*(it-1)));
		else if(t == TANHYP)
			theLayers.push_back(make_unique<TanHypLayer>(*(it),*(it-1)));
		else if(t == LINEAR)
			theLayers.push_back(make_unique<LinearLayer>(*(it),*(it-1)));
		else if(t == RELU)
			theLayers.push_back(make_unique<ReLULayer>(*(it),*(it-1)));
		else if(t == LEAKYRELU)
			theLayers.push_back(make_unique<LeakyReLULayer>(*(it),*(it-1)));
		else if(t == ELU_ACT)
			theLayers.push_back(make_unique<ELULayer>(*(it),*(it-1)));
	}
}

