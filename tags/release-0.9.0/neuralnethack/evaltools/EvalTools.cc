/*$Id: EvalTools.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "EvalTools.hh"
#include "Roc.hh"
#include "Gof.hh"

#include <cassert>
#include <cmath>

using namespace EvalTools;
using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

double ErrorMeasures::crossEntropy(Ensemble& committee, DataSet& data)
{
	double err=0;
	uint bs = data.size();
	
	for(uint i=0; i<bs; ++i){
		Pattern& p = data.pattern(i);
		vector<double> output = committee.propagate(p.input());
		vector<double>& target = p.output();
		err += crossEntropy(output, target);
	}
	return -err/(double)bs;
}

double ErrorMeasures::crossEntropy(const vector<double>& output, 
		const vector<double>& target)
{
	assert(output.size() == target.size());

	double power = -20;
	double tiny = exp(power);

	//return target[0] * log(output[0]) + (1.0 - target[0]) * log(1.0 - output[0]);
	double e = 0;
	if(target.size() == 1){
		if(target[0] == 0.0)
			e = (1.0 - output[0] > tiny) ? log(1.0 - output[0]) : power;
		else if(target[0] == 1.0)
			e = (output[0] > tiny) ? log(output[0]) : power;
		else
			cerr<<"Target is neither 1.0 nor 0.0 in the single class case!"<<endl;
	}else{
		for(uint i=0; i<target.size(); ++i)
			if(target[i] == 0.0) 
				e += 0;
			else if(target[i] == 1.0) 
				e += (output[i] > tiny) ? log(output[i]) : power;
	}
	return e;
}

double ErrorMeasures::summedSquare(Ensemble& committee, DataSet& data)
{
	double err=0;
	uint bs = data.size();

	for(uint i=0; i<bs; ++i){
		Pattern& p = data.pattern(i);
		vector<double> output = committee.propagate(p.input());
		vector<double>& target = p.output();
		err += summedSquare(output, target);
	}
	return 0.5*err/(double)bs;
}

double ErrorMeasures::summedSquare(const vector<double>& output, 
		const vector<double>& target)
{
	assert(output.size() == target.size());
	double e = 0;
	for(uint i=0; i<target.size(); ++i) e += pow(target[i] - output[i],2);
	return e;
}

double ErrorMeasures::auc(Ensemble& committee, DataSet& data)
{
	using EvalTools::Roc;
	vector<double> output;
	vector<uint> target;
	buildOutputTargetVectors(committee, data, output, target);
	Roc roc;
	//return roc.calcAucWmw(output, target);
	return roc.calcAucTrapezoidal(output, target);
}

double ErrorMeasures::gof(Ensemble& committee, DataSet& data)
{
	using EvalTools::Gof;
	vector<double> output;
	vector<uint> target;
	buildOutputTargetVectors(committee, data, output, target);
	Gof gof(10);
	return gof.goodnessOfFit(output, target);
}

void ErrorMeasures::buildOutputTargetVectors(Ensemble& committee, 
		DataSet& data, vector<double>& output, vector<uint>& target)
{
	output.clear();
	target.clear();

	for(uint i=0; i<data.size(); ++i){
		Pattern& pat = data.pattern(i);
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
	}
}

void ErrorMeasures::buildOutputTargetVectors(Ensemble& committee, 
		DataSet& data, vector< vector<double> >& output, vector< vector<double> >& target)
{
	output.clear();
	target.clear();

	for(uint i=0; i<data.size(); ++i){
		Pattern& pat = data.pattern(i);
		output.push_back(committee.propagate(pat.input()));
		target.push_back(pat.output());
	}
}
