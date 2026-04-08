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
	for (uint i = 0; i < n; ++i)
		out[i] = 1.0 / (1.0 + exp(-out[i]));
}

static void tanhypActivation(double* __restrict__ out, uint n) {
	for (uint i = 0; i < n; ++i)
		out[i] = tanh(out[i]);
}

static void linearActivation(double* __restrict__, uint) {
	// identity: no-op
}

// Vectorizable batch derivative-scale functions ------------------------------
// Each computes deltas[i] *= f'(outputs[i])

static void sigmoidDerivScale(const double* __restrict__ out, double* __restrict__ deltas, uint n) {
	for (uint i = 0; i < n; ++i)
		deltas[i] *= out[i] * (1.0 - out[i]);
}

static void tanhypDerivScale(const double* __restrict__ out, double* __restrict__ deltas, uint n) {
	for (uint i = 0; i < n; ++i)
		deltas[i] *= (1.0 - out[i] * out[i]);
}

static void linearDerivScale(const double* __restrict__, double* __restrict__, uint) {
	// f'(x) = 1, so deltas unchanged
}

// ReLU
static void reluActivation(double* __restrict__ out, uint n) {
	for (uint i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : 0.0;
}
static void reluDerivScale(const double* __restrict__ out, double* __restrict__ deltas, uint n) {
	for (uint i = 0; i < n; ++i)
		deltas[i] *= (out[i] > 0.0 ? 1.0 : 0.0);
}

// Leaky ReLU (alpha = 0.01)
static constexpr double LEAKY_ALPHA = 0.01;
static void leakyReluActivation(double* __restrict__ out, uint n) {
	for (uint i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : LEAKY_ALPHA * out[i];
}
static void leakyReluDerivScale(const double* __restrict__ out, double* __restrict__ deltas,
                                uint n) {
	for (uint i = 0; i < n; ++i)
		deltas[i] *= (out[i] > 0.0 ? 1.0 : LEAKY_ALPHA);
}

// ELU (alpha = 1.0)
static constexpr double ELU_ALPHA_VAL = 1.0;
static void eluActivation(double* __restrict__ out, uint n) {
	for (uint i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : ELU_ALPHA_VAL * (exp(out[i]) - 1.0);
}
static void eluDerivScale(const double* __restrict__ out, double* __restrict__ deltas, uint n) {
	for (uint i = 0; i < n; ++i)
		deltas[i] *= (out[i] >= 0.0 ? 1.0 : out[i] + ELU_ALPHA_VAL);
}

// Layer implementation -------------------------------------------------------

Layer::Layer(const uint nc, const uint np, const string t)
    : ncurr(nc), nprev(np), theType(t), theWeights(ncurr * (nprev + 1), 0), theOutputs(ncurr, 0),
      theLocalGradients(ncurr, 0), theGradients(ncurr * (nprev + 1), 0),
      theWeightUpdates(ncurr * (nprev + 1), 0), theDropoutRate(0.0), theTraining(false),
      theDropoutMask(ncurr, 1.0), theNormType(NormType::None), theGamma(ncurr, 1.0),
      theBeta(ncurr, 0.0), theGammaGrad(ncurr, 0.0), theBetaGrad(ncurr, 0.0),
      theGammaUpdate(ncurr, 0.0), theBetaUpdate(ncurr, 0.0), theRunningMean(ncurr, 0.0),
      theRunningVar(ncurr, 1.0), theBNMomentum(0.1), theActivation(nullptr),
      theDerivScale(nullptr) {
	if (t == SIGMOID) {
		theActivation = sigmoidActivation;
		theDerivScale = sigmoidDerivScale;
	} else if (t == TANHYP) {
		theActivation = tanhypActivation;
		theDerivScale = tanhypDerivScale;
	} else if (t == LINEAR) {
		theActivation = linearActivation;
		theDerivScale = linearDerivScale;
	} else if (t == RELU) {
		theActivation = reluActivation;
		theDerivScale = reluDerivScale;
	} else if (t == LEAKYRELU) {
		theActivation = leakyReluActivation;
		theDerivScale = leakyReluDerivScale;
	} else if (t == ELU_ACT) {
		theActivation = eluActivation;
		theDerivScale = eluDerivScale;
	}
	regenerateWeights();
}

Layer::Layer(const Layer& layer) {
	*this = layer;
}

Layer::~Layer() {}

Layer& Layer::operator=(const Layer& layer) {
	if (this != &layer) {
		ncurr = layer.ncurr;
		nprev = layer.nprev;
		theType = layer.theType;
		theWeights = layer.theWeights;
		theOutputs = layer.theOutputs;
		theLocalGradients = layer.theLocalGradients;
		theGradients = layer.theGradients;
		theWeightUpdates = layer.theWeightUpdates;
		theDropoutRate = layer.theDropoutRate;
		theTraining = layer.theTraining;
		theDropoutMask = layer.theDropoutMask;
		theNormType = layer.theNormType;
		theGamma = layer.theGamma;
		theBeta = layer.theBeta;
		theGammaGrad = layer.theGammaGrad;
		theBetaGrad = layer.theBetaGrad;
		theGammaUpdate = layer.theGammaUpdate;
		theBetaUpdate = layer.theBetaUpdate;
		theRunningMean = layer.theRunningMean;
		theRunningVar = layer.theRunningVar;
		theBNMomentum = layer.theBNMomentum;
		theActivation = layer.theActivation;
		theDerivScale = layer.theDerivScale;
	}
	return *this;
}

double& Layer::operator[](const uint i) {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

// PRINTS

void Layer::printWeights(ostream& os) const {
	copy(theWeights.begin(), theWeights.end(), ostream_iterator<double>(os, " "));
}

void Layer::printGradients(ostream& os) const {
	copy(theGradients.begin(), theGradients.end(), ostream_iterator<double>(os, " "));
}

// UTILS

void Layer::regenerateWeights() {
	for_each(theWeights.begin(), theWeights.end(), newRand<double>());
}

vector<double> Layer::calcLifs(const vector<double>& input) {
	vector<double> lif(ncurr, 0);
	vector<double>::iterator itw = theWeights.begin(), ito;
	for (ito = lif.begin(); ito != lif.end(); ++ito) {
		*ito = inner_product(input.begin(), input.end(), itw, *(itw + input.size()));
		advance(itw, input.size() + 1);
	}
	return lif;
}

vector<double>& Layer::propagate(const vector<double>& input) {
	// Phase 1: compute local induced fields (weighted sums + bias)
	const uint ni = input.size();
	const uint stride = ni + 1;
	const double* __restrict__ wt = theWeights.data();
	double* __restrict__ out = theOutputs.data();

#ifdef USE_BLAS
	const double* __restrict__ inp = input.data();
	for (uint i = 0; i < ncurr; ++i)
		out[i] = cblas_ddot(ni, inp, 1, wt + i * stride, 1) + wt[i * stride + ni];
#else
	const double* __restrict__ inp = input.data();
	for (uint i = 0; i < ncurr; ++i) {
		const double* __restrict__ row = wt + i * stride;
		double sum = row[ni]; // bias
		for (uint j = 0; j < ni; ++j)
			sum += inp[j] * row[j];
		out[i] = sum;
	}
#endif

	// Phase 2: apply activation in a single vectorizable loop
	theActivation(out, ncurr);

	// Phase 3: inverted dropout
	if (theTraining && theDropoutRate > 0.0) {
		const double scale = 1.0 / (1.0 - theDropoutRate);
		for (uint i = 0; i < ncurr; ++i) {
			theDropoutMask[i] = (drand48() >= theDropoutRate) ? scale : 0.0;
			out[i] *= theDropoutMask[i];
		}
	}

	return theOutputs;
}

void Layer::applyDerivative(vector<double>& deltas) {
	theDerivScale(theOutputs.data(), deltas.data(), ncurr);
	if (theTraining && theDropoutRate > 0.0)
		for (uint i = 0; i < ncurr; ++i)
			deltas[i] *= theDropoutMask[i];
}

const double* Layer::propagateBatch(const double* input, uint B, uint n_in) {
	assert(n_in == nprev);
	theBatchOutputs.resize(B * ncurr);
	double* out = theBatchOutputs.data();
	const double* wt = theWeights.data();
	const uint stride = nprev + 1;

#ifdef USE_BLAS
	// Out[B x ncurr] = Input[B x nprev] * W[ncurr x nprev]^T
	// W is [ncurr x (nprev+1)] row-major; ldb=stride skips bias column
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, ncurr, nprev, 1.0, input, nprev, wt,
	            stride, 0.0, out, ncurr);

	// Add bias to each row
	for (uint b = 0; b < B; ++b)
		for (uint j = 0; j < ncurr; ++j)
			out[b * ncurr + j] += wt[j * stride + nprev];
#else
	for (uint b = 0; b < B; ++b) {
		for (uint j = 0; j < ncurr; ++j) {
			const double* row = wt + j * stride;
			double sum = row[nprev]; // bias
			for (uint k = 0; k < nprev; ++k)
				sum += input[b * nprev + k] * row[k];
			out[b * ncurr + j] = sum;
		}
	}
#endif

	// Apply normalization (between linear and activation)
	if (theNormType != NormType::None) {
		const uint total = B * ncurr;
		theBatchZHat.resize(total);
		if (theNormType == NormType::BatchNorm) {
			theBatchNormMean.resize(ncurr);
			theBatchNormVar.resize(ncurr);
			if (theTraining && B > 1) {
				// Compute per-neuron mean and variance across batch
				for (uint j = 0; j < ncurr; ++j) {
					double mean = 0;
					for (uint b = 0; b < B; ++b)
						mean += out[b * ncurr + j];
					mean /= B;
					double var = 0;
					for (uint b = 0; b < B; ++b) {
						double d = out[b * ncurr + j] - mean;
						var += d * d;
					}
					var /= B;
					theBatchNormMean[j] = mean;
					theBatchNormVar[j] = var;
					// Update running stats
					theRunningMean[j] = (1.0 - theBNMomentum) * theRunningMean[j] + theBNMomentum * mean;
					theRunningVar[j] = (1.0 - theBNMomentum) * theRunningVar[j] + theBNMomentum * var;
					// Normalize and scale/shift
					double inv_std = 1.0 / sqrt(var + NORM_EPS);
					for (uint b = 0; b < B; ++b) {
						double zh = (out[b * ncurr + j] - mean) * inv_std;
						theBatchZHat[b * ncurr + j] = zh;
						out[b * ncurr + j] = theGamma[j] * zh + theBeta[j];
					}
				}
			} else {
				// Inference: use running stats
				for (uint j = 0; j < ncurr; ++j) {
					double inv_std = 1.0 / sqrt(theRunningVar[j] + NORM_EPS);
					for (uint b = 0; b < B; ++b) {
						double zh = (out[b * ncurr + j] - theRunningMean[j]) * inv_std;
						theBatchZHat[b * ncurr + j] = zh;
						out[b * ncurr + j] = theGamma[j] * zh + theBeta[j];
					}
				}
			}
		} else { // LayerNorm
			theBatchNormMean.resize(B);
			theBatchNormVar.resize(B);
			for (uint b = 0; b < B; ++b) {
				double mean = 0;
				for (uint j = 0; j < ncurr; ++j)
					mean += out[b * ncurr + j];
				mean /= ncurr;
				double var = 0;
				for (uint j = 0; j < ncurr; ++j) {
					double d = out[b * ncurr + j] - mean;
					var += d * d;
				}
				var /= ncurr;
				theBatchNormMean[b] = mean;
				theBatchNormVar[b] = var;
				double inv_std = 1.0 / sqrt(var + NORM_EPS);
				for (uint j = 0; j < ncurr; ++j) {
					double zh = (out[b * ncurr + j] - mean) * inv_std;
					theBatchZHat[b * ncurr + j] = zh;
					out[b * ncurr + j] = theGamma[j] * zh + theBeta[j];
				}
			}
		}
	}

	// Apply activation to all B*ncurr elements
	theActivation(out, B * ncurr);

	// Inverted dropout for batch
	if (theTraining && theDropoutRate > 0.0) {
		const double scale = 1.0 / (1.0 - theDropoutRate);
		const uint total = B * ncurr;
		theBatchDropoutMask.resize(total);
		for (uint i = 0; i < total; ++i) {
			theBatchDropoutMask[i] = (drand48() >= theDropoutRate) ? scale : 0.0;
			out[i] *= theBatchDropoutMask[i];
		}
	}

	return out;
}

void Layer::applyDerivativeBatch(uint B) {
	theDerivScale(theBatchOutputs.data(), theBatchLocalGradients.data(), B * ncurr);
	if (theTraining && theDropoutRate > 0.0 && !theBatchDropoutMask.empty()) {
		double* delta = theBatchLocalGradients.data();
		const double* mask = theBatchDropoutMask.data();
		const uint total = B * ncurr;
		for (uint i = 0; i < total; ++i)
			delta[i] *= mask[i];
	}
}

void Layer::accumulateGradientsBatch(const double* input, uint B) {
	const double* delta = theBatchLocalGradients.data();
	double* grad = theGradients.data();
	const uint stride = nprev + 1;

#ifdef USE_BLAS
	// dW[ncurr x nprev] += Delta^T[ncurr x B] * Input[B x nprev]
	// grad has ldc=stride to skip bias column
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ncurr, nprev, B, 1.0, delta, ncurr, input,
	            nprev, 1.0, grad, stride);
#else
	for (uint i = 0; i < ncurr; ++i) {
		for (uint j = 0; j < nprev; ++j) {
			double sum = 0;
			for (uint b = 0; b < B; ++b)
				sum += delta[b * ncurr + i] * input[b * nprev + j];
			grad[i * stride + j] += sum;
		}
	}
#endif

	// Bias gradients: column-sum of delta
	for (uint i = 0; i < ncurr; ++i) {
		double sum = 0;
		for (uint b = 0; b < B; ++b)
			sum += delta[b * ncurr + i];
		grad[i * stride + nprev] += sum;
	}
}

void Layer::applyNormBackwardBatch(uint B) {
	if (theNormType == NormType::None)
		return;

	double* delta = theBatchLocalGradients.data();
	const double* zhat = theBatchZHat.data();

	if (theNormType == NormType::BatchNorm) {
		for (uint j = 0; j < ncurr; ++j) {
			// Accumulate gamma/beta gradients
			double dg = 0, db = 0;
			for (uint b = 0; b < B; ++b) {
				uint idx = b * ncurr + j;
				dg += delta[idx] * zhat[idx];
				db += delta[idx];
			}
			theGammaGrad[j] += dg;
			theBetaGrad[j] += db;

			// Propagate through BN
			double inv_std = 1.0 / sqrt(theBatchNormVar[j] + NORM_EPS);
			double mean_dz = 0, mean_dz_zh = 0;
			for (uint b = 0; b < B; ++b) {
				uint idx = b * ncurr + j;
				double dz = delta[idx] * theGamma[j];
				mean_dz += dz;
				mean_dz_zh += dz * zhat[idx];
			}
			mean_dz /= B;
			mean_dz_zh /= B;
			for (uint b = 0; b < B; ++b) {
				uint idx = b * ncurr + j;
				double dz = delta[idx] * theGamma[j];
				delta[idx] = inv_std * (dz - mean_dz - zhat[idx] * mean_dz_zh);
			}
		}
	} else { // LayerNorm
		for (uint b = 0; b < B; ++b) {
			// Accumulate gamma/beta gradients
			for (uint j = 0; j < ncurr; ++j) {
				uint idx = b * ncurr + j;
				theGammaGrad[j] += delta[idx] * zhat[idx];
				theBetaGrad[j] += delta[idx];
			}
			// Propagate through LN
			double inv_std = 1.0 / sqrt(theBatchNormVar[b] + NORM_EPS);
			double mean_dz = 0, mean_dz_zh = 0;
			for (uint j = 0; j < ncurr; ++j) {
				uint idx = b * ncurr + j;
				double dz = delta[idx] * theGamma[j];
				mean_dz += dz;
				mean_dz_zh += dz * zhat[idx];
			}
			mean_dz /= ncurr;
			mean_dz_zh /= ncurr;
			for (uint j = 0; j < ncurr; ++j) {
				uint idx = b * ncurr + j;
				double dz = delta[idx] * theGamma[j];
				delta[idx] = inv_std * (dz - mean_dz - zhat[idx] * mean_dz_zh);
			}
		}
	}
}
