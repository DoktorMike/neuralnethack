/*$Id: Layer.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Layer.hh"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

#include <iostream>
#include <cassert>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>

using namespace MultiLayerPerceptron;
using namespace std;

// Vectorizable batch activation functions ------------------------------------

static void sigmoidActivation(double* __restrict__ out, uint n) {
	for(uint i = 0; i < n; ++i)
		out[i] = 1.0 / (1.0 + exp(-out[i]));
}

static void tanhypActivation(double* __restrict__ out, uint n) {
	for(uint i = 0; i < n; ++i)
		out[i] = tanh(out[i]);
}

static void linearActivation(double* __restrict__, uint) {
	// identity: no-op
}

// Vectorizable batch derivative-scale functions ------------------------------
// Each computes deltas[i] *= f'(outputs[i])

static void sigmoidDerivScale(const double* __restrict__ out,
                               double* __restrict__ deltas, uint n) {
	for(uint i = 0; i < n; ++i)
		deltas[i] *= out[i] * (1.0 - out[i]);
}

static void tanhypDerivScale(const double* __restrict__ out,
                              double* __restrict__ deltas, uint n) {
	for(uint i = 0; i < n; ++i)
		deltas[i] *= (1.0 - out[i] * out[i]);
}

static void linearDerivScale(const double* __restrict__,
                              double* __restrict__, uint) {
	// f'(x) = 1, so deltas unchanged
}

// Layer implementation -------------------------------------------------------

Layer::Layer(const uint nc, const uint np, const string t):
	ncurr(nc),
	nprev(np),
	theType(t),
	theWeights(ncurr*(nprev+1), 0),
	theOutputs(ncurr,0),
	theLocalGradients(ncurr,0),
	theGradients(ncurr*(nprev+1), 0),
	theWeightUpdates(ncurr*(nprev+1), 0),
	theActivation(nullptr),
	theDerivScale(nullptr)
{
	if(t == SIGMOID)     { theActivation = sigmoidActivation; theDerivScale = sigmoidDerivScale; }
	else if(t == TANHYP) { theActivation = tanhypActivation;  theDerivScale = tanhypDerivScale; }
	else if(t == LINEAR) { theActivation = linearActivation;  theDerivScale = linearDerivScale; }
	regenerateWeights();
}

Layer::Layer(const Layer& layer)
{*this = layer;}

Layer::~Layer()
{
}

Layer& Layer::operator=(const Layer& layer)
{
	if(this != &layer){
		ncurr=layer.ncurr;
		nprev=layer.nprev;
		theType=layer.theType;
		theWeights=layer.theWeights;
		theOutputs=layer.theOutputs;
		theLocalGradients=layer.theLocalGradients;
		theGradients=layer.theGradients;
		theWeightUpdates=layer.theWeightUpdates;
		theActivation=layer.theActivation;
		theDerivScale=layer.theDerivScale;
	}
	return *this;
}

double& Layer::operator[](const uint i)
{
	assert(i < theOutputs.size());
	return theOutputs[i];
}

//PRINTS

void Layer::printWeights(ostream& os) const
{ copy(theWeights.begin(), theWeights.end(), ostream_iterator<double>(os, " ")); }

void Layer::printGradients(ostream& os) const
{ copy(theGradients.begin(), theGradients.end(), ostream_iterator<double>(os, " ")); }

//UTILS

void Layer::regenerateWeights()
{ for_each(theWeights.begin(), theWeights.end(), newRand<double>()); }

vector<double> Layer::calcLifs(const vector<double>& input)
{
	vector<double> lif(ncurr, 0);
	vector<double>::iterator itw = theWeights.begin(), ito;
	for(ito = lif.begin(); ito != lif.end(); ++ito){
		*ito = inner_product(input.begin(), input.end(), itw, *(itw+input.size()));
		advance(itw, input.size()+1);
	}
	return lif;
}

vector<double>& Layer::propagate(const vector<double>& input)
{
	// Phase 1: compute local induced fields (weighted sums + bias)
	const uint ni = input.size();
	const uint stride = ni + 1;
	const double* __restrict__ wt = theWeights.data();
	double* __restrict__ out = theOutputs.data();

#ifdef USE_BLAS
	const double* __restrict__ inp = input.data();
	for(uint i = 0; i < ncurr; ++i)
		out[i] = cblas_ddot(ni, inp, 1, wt + i * stride, 1) + wt[i * stride + ni];
#else
	const double* __restrict__ inp = input.data();
	for(uint i = 0; i < ncurr; ++i){
		const double* __restrict__ row = wt + i * stride;
		double sum = row[ni]; // bias
		for(uint j = 0; j < ni; ++j)
			sum += inp[j] * row[j];
		out[i] = sum;
	}
#endif

	// Phase 2: apply activation in a single vectorizable loop
	theActivation(out, ncurr);

	return theOutputs;
}

void Layer::applyDerivative(vector<double>& deltas)
{
	theDerivScale(theOutputs.data(), deltas.data(), ncurr);
}

const double* Layer::propagateBatch(const double* input, uint B, uint n_in)
{
	assert(n_in == nprev);
	theBatchOutputs.resize(B * ncurr);
	double* out = theBatchOutputs.data();
	const double* wt = theWeights.data();
	const uint stride = nprev + 1;

#ifdef USE_BLAS
	// Out[B x ncurr] = Input[B x nprev] * W[ncurr x nprev]^T
	// W is [ncurr x (nprev+1)] row-major; ldb=stride skips bias column
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				B, ncurr, nprev,
				1.0,
				input, nprev,
				wt, stride,
				0.0,
				out, ncurr);

	// Add bias to each row
	for(uint b = 0; b < B; ++b)
		for(uint j = 0; j < ncurr; ++j)
			out[b * ncurr + j] += wt[j * stride + nprev];
#else
	for(uint b = 0; b < B; ++b){
		for(uint j = 0; j < ncurr; ++j){
			const double* row = wt + j * stride;
			double sum = row[nprev]; // bias
			for(uint k = 0; k < nprev; ++k)
				sum += input[b * nprev + k] * row[k];
			out[b * ncurr + j] = sum;
		}
	}
#endif

	// Apply activation to all B*ncurr elements
	theActivation(out, B * ncurr);

	return out;
}

void Layer::applyDerivativeBatch(uint B)
{
	theDerivScale(theBatchOutputs.data(), theBatchLocalGradients.data(), B * ncurr);
}

void Layer::accumulateGradientsBatch(const double* input, uint B)
{
	const double* delta = theBatchLocalGradients.data();
	double* grad = theGradients.data();
	const uint stride = nprev + 1;

#ifdef USE_BLAS
	// dW[ncurr x nprev] += Delta^T[ncurr x B] * Input[B x nprev]
	// grad has ldc=stride to skip bias column
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				ncurr, nprev, B,
				1.0,
				delta, ncurr,
				input, nprev,
				1.0,
				grad, stride);
#else
	for(uint i = 0; i < ncurr; ++i){
		for(uint j = 0; j < nprev; ++j){
			double sum = 0;
			for(uint b = 0; b < B; ++b)
				sum += delta[b * ncurr + i] * input[b * nprev + j];
			grad[i * stride + j] += sum;
		}
	}
#endif

	// Bias gradients: column-sum of delta
	for(uint i = 0; i < ncurr; ++i){
		double sum = 0;
		for(uint b = 0; b < B; ++b)
			sum += delta[b * ncurr + i];
		grad[i * stride + nprev] += sum;
	}
}

