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

	killGradients();

	const uint bs = theDset->size();
	const uint nOut = theMlp->layer(theMlp->nLayers()-1).nNeurons();

	// Pack dataset into contiguous batch matrices
	vector<double> inputMatrix, targetMatrix;
	packBatch(*theDset, inputMatrix, targetMatrix);

	// Batch forward pass (one GEMM per layer)
	const double* batchOut = theMlp->propagateBatch(inputMatrix.data(), bs);

	// Compute output-layer local gradients: delta = target - output
	// (CrossEntropy + sigmoid: derivative cancels)
	Layer& last = (*theMlp)[theMlp->nLayers()-1];
	last.batchLocalGradients().resize(bs * nOut);
	{
		double* delta = last.batchLocalGradients().data();
		const double* t = targetMatrix.data();
		const double* o = batchOut;
		for(uint i = 0; i < bs * nOut; ++i)
			delta[i] = t[i] - o[i];
	}

	// Batch backpropagate deltas through hidden layers (one GEMM per layer)
	for(int l = theMlp->size()-1; l > 0; --l){
		Layer& curr = (*theMlp)[l-1];
		Layer& next = (*theMlp)[l];
		const uint nc = curr.nNeurons();
		const uint nn = next.nNeurons();
		const uint nextStride = nc + 1; // next layer weight layout: [nn x (nc+1)]

		curr.batchLocalGradients().resize(bs * nc);
		double* clg = curr.batchLocalGradients().data();
		const double* nlg = next.batchLocalGradients().data();
		const double* wt = next.weights().data();

#ifdef USE_BLAS
		// delta_curr[B x nc] = delta_next[B x nn] * W_next[nn x nc]
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
		const double power = -20;
		const double tiny = exp(power);
		for(uint b = 0; b < bs; ++b){
			if(nOut == 1){
				if(t[b] == 0.0)
					err += (1.0 - o[b] > tiny) ? log(1.0 - o[b]) : power;
				else
					err += (o[b] > tiny) ? log(o[b]) : power;
			}else{
				for(uint j = 0; j < nOut; ++j){
					uint idx = b * nOut + j;
					if(t[idx] != 0.0)
						err += (o[idx] > tiny) ? log(o[idx]) : power;
				}
			}
		}
	}

	// Divide gradients by -bs and apply weight elimination
	for(uint l = 0; l < theMlp->nLayers(); ++l){
		Layer& layer = theMlp->layer(l);
		vector<double>& g = layer.gradients();
		div(g, -(double)bs);
		if(theWeightElimOn == true)
			weightElimGradLayer(g, layer.weights(), layer.nNeurons(), layer.nPrevious());
	}

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

void CrossEntropy::gradient(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i,j) = first.localGradients(i) * in[j];
		first.gradients(i, in.size()) = first.localGradients(i);
	}
}

void CrossEntropy::gradientBatch(Layer& first, vector<double>& in)
{
	for(uint i=0; i<first.size(); ++i){
		for(uint j=0; j<in.size(); ++j)
			first.gradients(i, j) += first.localGradients(i) * in[j];
		first.gradients(i, in.size()) += first.localGradients(i);
	}
}

void CrossEntropy::gradient(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) = curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) = curr.localGradients(i);
	}
}

void CrossEntropy::gradientBatch(Layer& curr, Layer& prev)
{
	for(uint i=0; i<curr.size(); ++i){
		for(uint j=0; j<prev.size(); ++j)
			curr.gradients(i, j) += curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) += curr.localGradients(i);
	}
}

double CrossEntropy::outputError(const vector<double>& out, const vector<double>& dout) const
{
	assert(out.size()==dout.size());

	double power = -20;
	double tiny = exp(power);

	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itd = dout.begin();
	if(dout.size() == 1){
		if(*itd == 0.0) return (1.0 - *ito > tiny) ? log(1.0 - *ito) : power;
		else return (*ito > tiny) ? log(*ito) : power;
	}

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

