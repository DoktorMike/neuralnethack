/*$Id: CrossEntropy.cc 1684 2007-10-12 15:55:07Z michael $*/

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


#include "CrossEntropy.hh"
#include "../matrixtools/MatrixTools.hh"

#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;
using std::vector;

CrossEntropy::CrossEntropy(Mlp& mlp, DataSet& dset):Error(mlp,dset){}

CrossEntropy::~CrossEntropy(){}

double CrossEntropy::gradient(Mlp& mlp, DataSet& dset)
{
	theMlp=&mlp;
	theDset=&dset;
	return gradient();
}

double CrossEntropy::gradient()
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;

	//Set all gradients to zero.
	killGradients();

	uint bs=theDset->size();
	for(uint i=0; i<bs; ++i){
		Pattern& p = theDset->pattern(i);
		const vector<double>& out = theMlp->propagate(p.input());
		Layer& last = (*theMlp)[theMlp->nLayers()-1];
		localGradient(last, out, p.output());
		backpropagate();
		gradientBatch((*theMlp)[0], p.input());

		for(uint i=1; i<theMlp->nLayers(); ++i)
			gradientBatch((*theMlp)[i],(*theMlp)[i-1]);
		err += outputError(out, p.output());
	}

	for(uint i=0; i<theMlp->nLayers(); ++i){
		Layer& l = theMlp->layer(i);
		vector<double>& g = l.gradients();
		div(g, -(double)bs);
		if(theWeightElimOn == true)
			weightElimGradLayer(g, l.weights(), l.nNeurons(), l.nPrevious());
	}

	//cout<<"Error: "<<-err/bs<<endl;
	return -err/(double)bs;
}

double CrossEntropy::outputError(Mlp& mlp, DataSet& dset)
{
	theMlp=&mlp;
	theDset=&dset;
	return outputError();
}

double CrossEntropy::outputError() const
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;
	uint bs=theDset->size();

	for(uint i=0; i<bs; ++i){
		Pattern& p=theDset->pattern(i);
		const vector<double>& output=theMlp->propagate(p.input());
		err+=outputError(output, p.output());
	}
	//cout<<"Error: "<<-err/bs<<endl;
	return -err/bs;
}

//PRIVATE--------------------------------------------------------------------//

CrossEntropy::CrossEntropy(const CrossEntropy& sse):Error(*(sse.theMlp), *(sse.theDset))
{*this = sse;}

CrossEntropy& CrossEntropy::operator=(const CrossEntropy& sse)
{
	if(this != &sse){
	}
	return *this;
}

void CrossEntropy::localGradient(Layer& ol, const vector<double>& out, 
		const vector<double>& dout)
{
	assert(out.size() == ol.size() && dout.size() == out.size());

	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itdo = dout.begin();

	for(uint i=0; i<ol.nNeurons(); ++i, ++ito, ++itdo)
		ol.localGradients(i) = (*itdo - *ito);
}

void CrossEntropy::backpropagate()
{
	for(int i=theMlp->size()-1; i>0; --i)
		localGradient((*theMlp)[i-1], (*theMlp)[i]);
}

void CrossEntropy::localGradient(Layer& curr, Layer& next)
{
	for(uint j=0; j<curr.nNeurons(); ++j){
		double err = 0;
		for(uint i=0; i<next.nNeurons(); ++i)
			err += next.localGradients(i)*next.weights(i,j);
		err = err*curr.firePrime(j);
		curr.localGradients(j) = err;
	}
}

void CrossEntropy::gradient(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i,j) = first.localGradients(i) * in[j];
		first.gradients(i, in.size()) = first.localGradients(i); //bias
	}
}

void CrossEntropy::gradientBatch(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i, j) += first.localGradients(i) * in[j];
		first.gradients(i, in.size()) += first.localGradients(i); //bias
	}
}

void CrossEntropy::gradient(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) = curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) = curr.localGradients(i); //bias
	}
}

void CrossEntropy::gradientBatch(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) += curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) += curr.localGradients(i); //bias
	}
}

double CrossEntropy::outputError(const vector<double>& out, const vector<double>& dout) const
{
	assert(out.size()==dout.size());

	double power = -20;
	double tiny = exp(power);

	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itd = dout.begin();
	//return *itd * log(*ito) + (1.0 - *itd) * log(1.0 - *ito);
	if(dout.size() == 1) 
		if(*itd == 0.0) return (1.0 - *ito > tiny) ? log(1.0 - *ito) : power;
		else return (*ito > tiny) ? log(*ito) : power;

	double e = 0;
	for(; ito!=out.end(); ++ito, ++itd){
		if(*itd == 0.0) e += 0;
		else e += (*ito > tiny) ? log(*ito) : power;
	}
	return e;
}

void CrossEntropy::killGradients()
{
	for(uint i=0; i<theMlp->nLayers(); ++i){
		Layer& l = theMlp->layer(i);
		vector<double>& g = l.gradients();
		g.assign(g.size(), 0);
	}
}
