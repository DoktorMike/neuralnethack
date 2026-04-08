/*$Id: SummedSquare.cc 1684 2007-10-12 15:55:07Z michael $*/

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


#include "SummedSquare.hh"
#include "../matrixtools/MatrixTools.hh"
#include "../datatools/Pattern.hh"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

#include <cmath>
#include <cassert>
#include <algorithm>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;
using std::vector;

SummedSquare::SummedSquare(Mlp& mlp, DataSet& dset):Error(mlp, dset){}

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

	killGradients();

	const uint bs = theDset->size();
	const uint nOut = theMlp->layer(theMlp->nLayers()-1).nNeurons();

	// Pack dataset into contiguous batch matrices
	vector<double> inputMatrix, targetMatrix;
	packBatch(*theDset, inputMatrix, targetMatrix);

	// Batch forward pass (one GEMM per layer)
	const double* batchOut = theMlp->propagateBatch(inputMatrix.data(), bs);

	// Compute output-layer local gradients: delta = (target - output) * f'(output)
	Layer& last = (*theMlp)[theMlp->nLayers()-1];
	last.batchLocalGradients().resize(bs * nOut);
	{
		double* delta = last.batchLocalGradients().data();
		const double* t = targetMatrix.data();
		const double* o = batchOut;
		for(uint i = 0; i < bs * nOut; ++i)
			delta[i] = t[i] - o[i];
	}
	// SummedSquare applies derivative to output layer (unlike CrossEntropy)
	last.applyDerivativeBatch(bs);

	// Batch backpropagate deltas through hidden layers (one GEMM per layer)
	for(int l = theMlp->size()-1; l > 0; --l){
		Layer& curr = (*theMlp)[l-1];
		Layer& next = (*theMlp)[l];
		const uint nc = curr.nNeurons();
		const uint nn = next.nNeurons();
		const uint nextStride = nc + 1;

		curr.batchLocalGradients().resize(bs * nc);
		double* clg = curr.batchLocalGradients().data();
		const double* nlg = next.batchLocalGradients().data();
		const double* wt = next.weights().data();

#ifdef USE_BLAS
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					bs, nc, nn,
					1.0,
					nlg, nn,
					wt, nextStride,
					0.0,
					clg, nc);
#else
		for(uint b = 0; b < bs; ++b){
			for(uint j = 0; j < nc; ++j){
				double err = 0;
				for(uint k = 0; k < nn; ++k)
					err += nlg[b * nn + k] * wt[k * nextStride + j];
				clg[b * nc + j] = err;
			}
		}
#endif
		curr.applyDerivativeBatch(bs);
	}

	// Batch gradient accumulation (one GEMM per layer)
	(*theMlp)[0].accumulateGradientsBatch(inputMatrix.data(), bs);
	for(uint l = 1; l < theMlp->nLayers(); ++l)
		(*theMlp)[l].accumulateGradientsBatch((*theMlp)[l-1].batchOutputs().data(), bs);

	// Compute total error
	double err = 0;
	{
		const double* o = batchOut;
		const double* t = targetMatrix.data();
		for(uint b = 0; b < bs; ++b)
			for(uint j = 0; j < nOut; ++j){
				double diff = t[b * nOut + j] - o[b * nOut + j];
				err += diff * diff;
			}
	}

	// Divide gradients by -bs and apply weight elimination
	for(uint l = 0; l < theMlp->nLayers(); ++l){
		Layer& layer = theMlp->layer(l);
		vector<double>& g = layer.gradients();
		std::transform(g.begin(), g.end(), g.begin(),
				[bs](double v){ return v / -(double)bs; });
		if(theWeightElimOn == true)
			weightElimGradLayer(g, layer.weights(), layer.nNeurons(), layer.nPrevious());
	}

	return err/(double)bs;
}

double SummedSquare::outputError(Mlp& mlp, DataSet& dset)
{
	theMlp=&mlp;
	theDset=&dset;
	return outputError();
}

double SummedSquare::outputError() const
{
	assert(theDset!=0 && theMlp!=0);
	double err=0;
	uint bs=theDset->size();

	for(uint i=0; i<bs; ++i){
		Pattern& p=theDset->pattern(i);
		vector<double> output=theMlp->propagate(p.input());
		err+=outputError(output, p.output());
	}
	return err/(double)bs;
}

//PRIVATE--------------------------------------------------------------------//

SummedSquare::SummedSquare(const SummedSquare& sse):Error(*(sse.theMlp),
		*(sse.theDset)){*this = sse;}

SummedSquare& SummedSquare::operator=(const SummedSquare& sse)
{
	if(this != &sse){
	}
	return *this;
}

void SummedSquare::localGradient(Layer& ol, const vector<double>& out,
		const vector<double>& dout)
{
	assert(out.size() == ol.size() && dout.size() == out.size());
	const uint n = ol.nNeurons();
	double* __restrict__ lg = ol.localGradients().data();
	const double* __restrict__ o  = out.data();
	const double* __restrict__ d  = dout.data();
	for(uint i = 0; i < n; ++i)
		lg[i] = d[i] - o[i];
	ol.applyDerivative(ol.localGradients());
}

void SummedSquare::backpropagate()
{
	for(int i=theMlp->size()-1; i>0; --i)
		localGradient((*theMlp)[i-1], (*theMlp)[i]);
}

void SummedSquare::localGradient(Layer& curr, Layer& next)
{
	const uint nc = curr.nNeurons();
	const uint nn = next.nNeurons();
	double* __restrict__ clg = curr.localGradients().data();
	const double* __restrict__ nlg = next.localGradients().data();
	for(uint j = 0; j < nc; ++j){
		double err = 0;
		for(uint i = 0; i < nn; ++i)
			err += nlg[i] * next.weights(i, j);
		clg[j] = err;
	}
	curr.applyDerivative(curr.localGradients());
}

void SummedSquare::gradient(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i,j) = first.localGradients(i) * in[j];
		first.gradients(i, in.size()) = first.localGradients(i);
	}
}

void SummedSquare::gradientBatch(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i, j) += first.localGradients(i) * in[j];
		first.gradients(i, in.size()) += first.localGradients(i);
	}
}

void SummedSquare::gradient(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) = curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) = curr.localGradients(i);
	}
}

void SummedSquare::gradientBatch(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) += curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) += curr.localGradients(i);
	}
}

double SummedSquare::outputError(const vector<double>& out, const vector<double>& dout) const
{
	assert(out.size()==dout.size());
	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itd = dout.begin();
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

