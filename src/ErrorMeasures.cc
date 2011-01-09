#include "ErrorMeasures.hh"
#include "evaltools/Roc.hh"

#include <cassert>
#include <cmath>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

double ErrorMeasures::crossEntropy(Committee& committee, DataSet& data)
{
	double err=0;
	uint bs = data.size();
	
	for(uint i=0; i<bs; ++i){
		Pattern& p = data.pattern(i);
		vector<double> output = committee.propagate(p.input());
		vector<double>& target = p.output();
		err += crossEntropy(output, target);
	}
	return err/(double)bs;
}

double ErrorMeasures::crossEntropy(vector<double>& output, vector<double>& target)
{
	assert(output.size() == target.size());

	if(target.size() == 1) 
		return target[0] * log(output[0]) + (1.0 - target[0]) * log(1.0 - output[0]);

	double e = 0;
	for(uint i=0; i<target.size(); ++i)
		if(target[i] == 0.0) 
			e += 0;
		else if(target[i] == 1.0) 
			e += (output[i] != 0.0) ? log(output[i]) : -1000;
	return e;
}

double ErrorMeasures::summedSquare(Committee& committee, DataSet& data)
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

double ErrorMeasures::summedSquare(vector<double>& output, vector<double>& target)
{
	assert(output.size() == target.size());
	double e = 0;
	for(uint i=0; i<target.size(); ++i) e += pow(target[i] - output[i],2);
	return e;
}

double ErrorMeasures::auc(Committee& committee, DataSet& data)
{
	using EvalTools::Roc;
	vector<double> output;
	vector<uint> target;
	buildOutputTargetVectors(committee, data, output, target);
	Roc roc;
	//return roc.calcAucWmw(output, target);
	return roc.calcAucTrapezoidal(output, target);
}

void ErrorMeasures::buildOutputTargetVectors(Committee& committee, 
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

void ErrorMeasures::buildOutputTargetVectors(Committee& committee, 
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
