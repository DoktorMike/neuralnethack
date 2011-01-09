/*$Id: Saliency.cc 1690 2007-11-13 14:38:39Z michael $*/

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
using std::divides;
using std::bind2nd;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::binary_function;

//Saliency calculations based on gradients.

template<class T>
struct plusMag : public binary_function<T,T,T> {
	T operator()(T& x, T& y) { return fabs(x) + fabs(y);}
};

vector<double> Saliency::saliencyMagnitude(Ensemble& ensemble, DataSet& data, bool inner)
{
	vector<double> sal(data.nInput(),0);
	for(uint i=0; i<ensemble.size(); ++i){
		vector<double> tmp = saliencyMagnitude(ensemble.mlp(i), data, inner);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	transform(sal.begin(), sal.end(), sal.begin(), 
			bind2nd(divides<double>(), (double)ensemble.size()));
	return sal;
}

vector<double> Saliency::saliency(Ensemble& ensemble, DataSet& data, bool inner)
{
	vector<double> sal(data.nInput(),0);
	for(uint i=0; i<ensemble.size(); ++i){
		vector<double> tmp = saliency(ensemble.mlp(i), data, inner);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	transform(sal.begin(), sal.end(), sal.begin(), 
			bind2nd(divides<double>(), (double)ensemble.size()));
	return sal;
}

vector<double> Saliency::saliency(Ensemble& ensemble, Pattern& pattern, bool inner)
{
	vector<double> sal(pattern.nInput(),0);
	for(uint i=0; i<ensemble.size(); ++i){
		vector<double> tmp = saliency(ensemble.mlp(i), pattern, inner);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	transform(sal.begin(), sal.end(), sal.begin(), 
			bind2nd(divides<double>(), (double)ensemble.size()));
	return sal;
}

vector<double> Saliency::saliencyMagnitude(Mlp& mlp, DataSet& data, bool inner)
{
	vector<double> sal(data.nInput(),0);

	for(uint i=0; i<data.size(); ++i){
		vector<double> tmp = saliency(mlp, data.pattern(i), inner);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plusMag<double>());
	}
	transform(sal.begin(), sal.end(), sal.begin(), 
			bind2nd(divides<double>(), (double)data.size()));
	return sal;
}

vector<double> Saliency::saliency(Mlp& mlp, DataSet& data, bool inner)
{
	vector<double> sal(data.nInput(),0);

	for(uint i=0; i<data.size(); ++i){
		vector<double> tmp = saliency(mlp, data.pattern(i), inner);
		transform(sal.begin(), sal.end(), tmp.begin(), 
				sal.begin(), plus<double>());
	}
	transform(sal.begin(), sal.end(), sal.begin(), 
			bind2nd(divides<double>(), (double)data.size()));
	return sal;
}

vector<double> Saliency::saliency(Mlp& mlp, Pattern& pattern, bool inner)
{
	vector<double> input = pattern.input();
	vector<double> sal(input.size(), 0);

	for(uint i=0; i<input.size(); ++i)
		sal[i] = (inner == true) ? derivative_inner(mlp, input, i) : derivative(mlp, input, i);
	return sal;
}

double Saliency::derivative_debug(Mlp& mlp, vector<double>& input, uint index)
{
	mlp.propagate(input);

	/*Special two layer case*/
	Layer& curr = mlp.layer(1);
	Layer& prev = mlp.layer(0);

	//cout<<"OLD: \n";
	double sum = 0;
	for(uint j=0; j<curr.nPrevious(); ++j){
		sum += curr.weights(0,j) * prev.firePrime(j) * prev.weights(j, index);
		//cout<<"sum += curr.weights(0,"<<j<<")*prev.firePrime("<<j<<")*prev.weights("<<j<<", "<<index<<")";
		//cout<<endl;
		//cout<<"sum += "<<curr.weights(0,j)<<"*"<<prev.firePrime(j)<<"*"<<prev.weights(j, index);
		//cout<<endl;
	}
	return sum * curr.firePrime((uint)0);
}

double calcSum(uint i, uint layerIndex, uint varIndex, Mlp& mlp)
{
	double sum = 0;
	Layer& curr = mlp.layer(layerIndex);
	if(layerIndex == 0){
		//cout<<"* curr.weights("<<i<<","<<varIndex<<")"<<endl;
		//cout<<"* "<<curr.weights(i,varIndex)<<endl;
		sum = curr.weights(i,varIndex); 
	}else{
		Layer& prev = mlp.layer(layerIndex-1);
		for(uint l=0; l<curr.nPrevious(); ++l){
			//cout<<"sum += curr.weights("<<i<<","<<l<<")*prev.firePrime("<<l<<")*"<<"calcSum("<<l<<","<<layerIndex-1<<", mlp, "<<sum<<")";
			//cout<<"sum += "<<curr.weights(i,l)<<"*"<<prev.firePrime(l)<<"*"<<calcSum(l, layerIndex-1, varIndex, mlp);
			sum += curr.weights(i,l) * prev.firePrime(l)*calcSum(l, layerIndex-1, varIndex, mlp);
		}
	}
	return sum;
}

double Saliency::derivative(Mlp& mlp, vector<double>& input, uint index)
{
	mlp.propagate(input);
	double sum = calcSum(0, mlp.nLayers()-1, index, mlp);
	return sum * mlp.layer(mlp.nLayers()-1).firePrime((uint)0);
}

double Saliency::derivative_inner(Mlp& mlp, vector<double>& input, uint index)
{
	mlp.propagate(input);
	double sum = calcSum(0, mlp.nLayers()-1, index, mlp);
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
	for(uint i=1; i<=sal.size(); ++i) os<<i<<"\t"<<sal[i-1]<<endl;
}
