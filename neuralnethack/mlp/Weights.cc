/*$Id: Weights.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Weights.hh"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace MultiLayerPerceptron;
using namespace std;

Weights::Weights(vector<uint>& a):arch(a)
{
	vector<uint>::iterator it = arch.begin();
	int prev = *it;
	int num = 0;
	
	while(++it != arch.end()){
		num += (prev+1)*(*it);
		prev = *it;
	}
	
	theWeights = new vector<double>(num);
	for(int i=0; i<num; i++)
		(*theWeights)[i] = 0.5-drand48(); 
}

Weights::Weights(const Weights& w) {*this = w;}

Weights& Weights::operator=(const Weights& w)
{
	if(this != &w){
		theWeights = new vector<double>(*(w.theWeights));
		arch = w.arch;
	}
	return *this;
}

void Weights::print()
{
	vector<uint>::iterator itaprev = arch.begin();
	vector<uint>::iterator itacurr = itaprev;
	vector<double>::iterator itw = (*theWeights).begin();
	int i=0;

	while(++itacurr != arch.end()){
		cout<<endl<<"Layer "<<++i<<":"<<endl;
		for(uint j=0; j<((*itacurr)*((*itaprev)+1)); j++){
			cout.width(4);
			cout<<*itw<<" ";
			itw++;
		}
		itaprev++;
		cout<<endl;
	}
}

void Weights::kill(uint n){update(n, 0.0);}

void Weights::kill(uint layer, uint ncurr, uint nprev)
{
	update(index(layer, ncurr, nprev), 0.0);
}

void Weights::update(uint n, double w){(*theWeights)[n] = w;}

void Weights::update(uint layer, uint ncurr, uint nprev, double w)
{
	update(index(layer,ncurr,nprev),w);
}

vector<double>& Weights::weights()
{return (*theWeights);}

void Weights::weights(uint layer, vector<double>::iterator& first, 
		vector<double>::iterator& last)
{itor(layer,first,last);}

void Weights::weights(uint layer, uint neuron,
		vector<double>::iterator& first, 
		vector<double>::iterator& last)
{itor(layer,neuron,first,last);}

void Weights::weights(uint layer, uint ncurr, uint nprev,
		vector<double>::iterator& first, 
		vector<double>::iterator& last)
{itor(layer,ncurr,nprev,first,last);}

Weights::~Weights(){delete theWeights;}

uint Weights::size()
{return (*theWeights).size();}

/*---------------------------------------------------------------------------*/

uint Weights::index(uint layer)
{
    assert((layer>=0) && (layer<arch.size()-1));
    uint index = 0;
    for(uint i=1; i<=layer; ++i)
	index += arch[i]*(arch[i-1]+1);
    return index;
}

uint Weights::index(uint layer, uint neuron)
{
    assert((neuron>=0) && (neuron<arch[layer+1]));
    return index(layer)+(arch[layer]+1)*neuron;
}

uint Weights::index(uint layer, uint ncurr, uint nprev)
{
    assert((ncurr>=0) && (ncurr<arch[layer+1]));
    assert((nprev>=0) && (nprev<arch[layer]+1));
    return index(layer,ncurr)+nprev;
}

void Weights::itor(uint layer, vector<double>::iterator& first,
	vector<double>::iterator& last)
{
    assert(layer>=0 && layer<arch.size()-1); //Throw an exception instead!!
    first = (*theWeights).begin() + index(layer);
    last = first+arch[layer+1]*(arch[layer]+1)-1;
}

void Weights::itor(uint layer, uint neuron, vector<double>::iterator& first,
	vector<double>::iterator& last)
{
    assert(layer>=0 && layer<arch.size()-1); //Throw an exception instead!!
    assert(neuron>=0 && neuron<arch[layer+1]); //Throw an exception instead!!
    itor(layer, first, last);
    first += neuron*(arch[layer]+1);
    last = first + arch[layer]; //+1 and -1 cancel each other.
}

void Weights::itor(uint layer, uint ncurr,uint nprev, 
	vector<double>::iterator& first,
	vector<double>::iterator& last)
{
    assert(ncurr>=0 && ncurr<arch[layer+1]); //Throw an exception instead!!
    assert(nprev>=0 && nprev<arch[layer]+1); //Throw an exception instead!!
    itor(layer,ncurr,first,last);
    last = (first+=nprev);
}

