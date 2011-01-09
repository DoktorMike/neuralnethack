#include "CrossEntropy.hh"
#include "MatrixTools.hh"

#include <cmath>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace MatrixTools;

CrossEntropy::CrossEntropy():Error(CEE){}

CrossEntropy::CrossEntropy(Mlp* mlp, DataSet* dset):Error(mlp,dset,CEE){}

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
	theDset->reset();
	for(uint i=0; i<bs; ++i){
		Pattern& p = theDset->nextPattern();
		vector<double>& out = theMlp->propagate(p.input());
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

double CrossEntropy::outputError()
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;
	uint bs=theDset->size();
	theDset->reset();
	
	for(uint i=0; i<bs; ++i){
		Pattern& p=theDset->nextPattern();
		vector<double>& output=theMlp->propagate(p.input());
		err+=outputError(output, p.output());
	}
	//cout<<"Error: "<<-err/bs<<endl;
	return -err/bs;
}

//PRIVATE--------------------------------------------------------------------//

CrossEntropy::CrossEntropy(const CrossEntropy& sse):Error(sse.theMlp, 
		sse.theDset, sse.theType){*this = sse;}

CrossEntropy& CrossEntropy::operator=(const CrossEntropy& sse)
{
	if(this != &sse){
	}
	return *this;
}

void CrossEntropy::localGradient(Layer& ol, vector<double>& out, 
		vector<double>& dout)
{
	assert(out.size() == ol.size() && dout.size() == out.size());

	vector<double>::iterator ito = out.begin();
	vector<double>::iterator itdo = dout.begin();

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

double CrossEntropy::outputError(vector<double>& out, vector<double>& dout)
{
	assert(out.size()==dout.size());

	vector<double>::iterator ito = out.begin();
	vector<double>::iterator itd = dout.begin();
	double e = 0;
	//double target = 0;
	//double output = 0;
	for(; ito!=out.end(); ++ito, ++itd){
		//if(*itd != 0.0) target = *itd; else target = 1.0/(double)INT_MAX;
		//if(*ito != 0.0) output = *ito; else output = 1.0/(double)INT_MAX;
		//cout<<"Target: "<<target<<" Output: "<<*ito;
		if(*itd == 0.0) e = log(1.0-*ito);
		if(*itd == 1.0) e = log(*ito);
		//e += target * log(output) + (1.0 - target)*log(1.0 - output);
		//e += target * ( log(output) - log(target) );
		//cout<<" Error: "<<e<<endl;
	}
	return (e > 0.0) ? -0.0 : e;
}

void CrossEntropy::killGradients()
{
	for(uint i=0; i<theMlp->nLayers(); ++i){
		Layer& l = theMlp->layer(i);
		vector<double>& g = l.gradients();
		g.assign(g.size(), 0);
	}
}