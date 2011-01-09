/*$Id: Saliency.cc 1620 2007-05-07 17:27:52Z michael $*/

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


#include "Saliency.hh"
#include "matrixtools/MatrixTools.hh"

#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>

using namespace NeuralNetHack;
using MultiLayerPerceptron::Mlp;
using MultiLayerPerceptron::Layer;
using DataTools::DataSet;
using DataTools::Pattern;

using std::vector;
using std::plus;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;

//Saliency calculations based on gradients.

vector<double> Saliency::saliency(Ensemble& committee, DataSet& data)
{
	vector<double> sal(data.nInput(),0);
	for(uint i=0; i<committee.size(); ++i){
		vector<double> tmp = saliency(committee.mlp(i), data);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	for(vector<double>::iterator it=sal.begin(); it!=sal.end(); ++it)
		(*it) /= (double)committee.size();
	return sal;
}

vector<double> Saliency::saliency(Ensemble& committee, Pattern& pattern)
{
	vector<double> sal(pattern.nInput(),0);
	for(uint i=0; i<committee.size(); ++i){
		vector<double> tmp = saliency(committee.mlp(i), pattern);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	for(vector<double>::iterator it=sal.begin(); it!=sal.end(); ++it)
		(*it) /= (double)committee.size();
	return sal;
}

vector<double> Saliency::saliency(Mlp& mlp, DataSet& data)
{
	vector<double> sal(data.nInput(),0);

	for(uint i=0; i<data.size(); ++i){
		vector<double> tmp = saliency(mlp, data.pattern(i));
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	for(vector<double>::iterator it=sal.begin(); it!=sal.end(); ++it)
		(*it) /= (double)data.size();
	return sal;
}

vector<double> Saliency::saliency(Mlp& mlp, Pattern& pattern)
{
	vector<double> input = pattern.input();
	vector<double> sal(input.size(), 0);

	for(uint i=0; i<input.size(); ++i){
		sal[i] = derivative(mlp, input, i);
		//cout<<"Saliency: "<<sal[i]<<"\n";
	}
	return sal;
}

/** \todo generalise the derivative to any number of layers. */
double Saliency::derivative(Mlp& mlp, vector<double>& input, uint index)
{
	mlp.propagate(input);

	/*Special two layer case*/
	Layer& curr = mlp.layer(1);
	Layer& prev = mlp.layer(0);

	double sum = 0;
	for(uint j=0; j<curr.nPrevious(); ++j)
		sum += curr.weights(0,j) * prev.firePrime(j) * prev.weights(j, index);
	return sum * curr.firePrime((uint)0);
}

/** \todo generalise the derivative to any number of layers. */
double Saliency::derivative_inner(Mlp& mlp, vector<double>& input, uint index)
{
	mlp.propagate(input);

	/*Special two layer case*/
	Layer& curr = mlp.layer(1);
	Layer& prev = mlp.layer(0);

	double sum = 0;
	for(uint j=0; j<curr.nPrevious(); ++j)
		sum += curr.weights(0,j) * prev.firePrime(j) * prev.weights(j, index);
	return sum;
}

//Utilities like printing and stuff

void Saliency::print(ostream& os, vector<double>& sal)
{
	if(!os){
		cerr<<"Saliency::print: Problem with output stream."<<endl;
		return;
	}

	os<<"Input\tSaliency"<<endl;
	for(uint i=0; i<sal.size(); ++i) os<<i<<"\t"<<sal[i]<<endl;
}
