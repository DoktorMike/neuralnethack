#include "SummedSquare.hh"
#include "matrixtools/MatrixTools.hh"

#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;

SummedSquare::SummedSquare():Error(0, 0){}

SummedSquare::SummedSquare(Mlp* mlp, DataSet* dset):Error(mlp, dset){}

SummedSquare::~SummedSquare(){}

double SummedSquare::gradient(Mlp& mlp, DataSet& dset)
{
	theMlp=&mlp;
	theDset=&dset;
	return gradient();
}

double SummedSquare::gradient()
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;

	//Set all gradients to zero.
	killGradients();

	uint bs=theDset->size();
	for(uint i=0; i<bs; ++i){
		Pattern& p = theDset->pattern(i);
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
		weightElimGradLayer(g, l.nNeurons(), l.nPrevious());
	}
	
	return 0.5*err/bs;
}

double SummedSquare::outputError(Mlp& mlp, DataSet& dset)
{
	theMlp=&mlp;
	theDset=&dset;
	return outputError();
}

double SummedSquare::outputError()
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;
	uint bs=theDset->size();
	
	for(uint i=0; i<bs; ++i){
		Pattern& p=theDset->pattern(i);
		vector<double> output=theMlp->propagate(p.input());
		err+=outputError(output, p.output());
	}
	return 0.5*err/bs;
}

//PRIVATE--------------------------------------------------------------------//

SummedSquare::SummedSquare(const SummedSquare& sse):Error(sse.theMlp, 
		sse.theDset){*this = sse;}

SummedSquare& SummedSquare::operator=(const SummedSquare& sse)
{
	if(this != &sse){
	}
	return *this;
}

void SummedSquare::localGradient(Layer& ol, vector<double>& out, 
		vector<double>& dout)
{
	assert(out.size() == ol.size() && dout.size() == out.size());

	vector<double>::iterator ito = out.begin();
	vector<double>::iterator itdo = dout.begin();

	for(uint i=0; i<ol.nNeurons(); ++i, ++ito, ++itdo)
		ol.localGradients(i) = (*itdo - *ito) * ol.firePrime(i);
}

void SummedSquare::backpropagate()
{
	for(int i=theMlp->size()-1; i>0; --i)
		localGradient((*theMlp)[i-1], (*theMlp)[i]);
}

void SummedSquare::localGradient(Layer& curr, Layer& next)
{
	for(uint j=0; j<curr.nNeurons(); ++j){
		double err = 0;
		for(uint i=0; i<next.nNeurons(); ++i)
			err += next.localGradients(i)*next.weights(i,j);
		err = err*curr.firePrime(j);
		curr.localGradients(j) = err;
	}
}

void SummedSquare::gradient(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i,j) = first.localGradients(i) * in[j];
		first.gradients(i, in.size()) = first.localGradients(i); //bias
	}
}

void SummedSquare::gradientBatch(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i, j) += first.localGradients(i) * in[j];
		first.gradients(i, in.size()) += first.localGradients(i); //bias
	}
}

void SummedSquare::gradient(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) = curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) = curr.localGradients(i); //bias
	}
}

void SummedSquare::gradientBatch(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) += curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) += curr.localGradients(i); //bias
	}
}

double SummedSquare::outputError(vector<double>& out, vector<double>& dout)
{
	assert(out.size()==dout.size());

	vector<double>::iterator ito = out.begin();
	vector<double>::iterator itd = dout.begin();
	double e = 0;
	for(; ito!=out.end(); ++ito, ++itd)
		e += pow(*itd - *ito,2);
	return e;
}

void SummedSquare::killGradients()
{
	for(uint i=0; i<theMlp->nLayers(); ++i){
		Layer& l = theMlp->layer(i);
		vector<double>& g = l.gradients();
		g.assign(g.size(), 0);
	}
}
