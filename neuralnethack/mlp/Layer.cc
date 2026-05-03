#include "Layer.hh"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>

using namespace MultiLayerPerceptron;
using namespace std;

// Layer implementation -------------------------------------------------------

Layer::Layer(const uint nc, const uint np, const string t)
    : Layer(nc, np, activationFromTag(t)) {}

Layer::Layer(const uint nc, const uint np, Activation act)
    : ncurr(nc), nprev(np), theType(activationToTag(act)), theAct(std::move(act)),
      theWeights(ncurr * (nprev + 1), 0), theOutputs(ncurr, 0), theLocalGradients(ncurr, 0),
      theGradients(ncurr * (nprev + 1), 0), theWeightUpdates(ncurr * (nprev + 1), 0),
      theDropoutRate(0.0), theTraining(false), theDropoutMask(ncurr, 1.0),
      theNormType(NormType::None), theGamma(ncurr, 1.0), theBeta(ncurr, 0.0),
      theGammaGrad(ncurr, 0.0), theBetaGrad(ncurr, 0.0), theGammaUpdate(ncurr, 0.0),
      theBetaUpdate(ncurr, 0.0), theRunningMean(ncurr, 0.0), theRunningVar(ncurr, 1.0),
      theBNMomentum(0.1) {
	regenerateWeights();
}

double& Layer::operator[](const uint i) {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

// Scalar activation API ------------------------------------------------------

double Layer::fire(double lif) const {
	return std::visit([lif](const auto& a) { return MultiLayerPerceptron::fire(a, lif); }, theAct);
}

double Layer::firePrime(double lif) const {
	return std::visit([lif](const auto& a) { return MultiLayerPerceptron::firePrime(a, lif); },
	                  theAct);
}

double Layer::firePrime(const uint i) const {
	assert(i < theOutputs.size());
	const double y = theOutputs[i];
	return std::visit([y](const auto& a) { return firePrimeFromOutput(a, y); }, theAct);
}

double Layer::firePrimePrime(double lif) const {
	return std::visit([lif](const auto& a) { return MultiLayerPerceptron::firePrimePrime(a, lif); },
	                  theAct);
}

double Layer::firePrimePrime(const uint i) const {
	assert(i < theOutputs.size());
	const double y = theOutputs[i];
	return std::visit([y](const auto& a) { return firePrimePrimeFromOutput(a, y); }, theAct);
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
	const uint stride = nprev + 1;
	if (theInitScheme == InitScheme::LegacyUniform) {
		for (auto& w : theWeights)
			w = 0.5 - nnh::rand::uniform();
		return;
	}
	// Per-activation Xavier-family init. Saturating activations get Glorot
	// (a = sqrt(6/(n_in+n_out))); ReLU-family get He (a = sqrt(6/n_in))
	// because that's what compensates for the half of inputs they zero out.
	// Biases (the last column in each row of the [ncurr, nprev+1] flat
	// layout) are zeroed.
	const bool isRelu = std::holds_alternative<ReLU>(theAct) ||
	                    std::holds_alternative<LeakyReLU>(theAct) ||
	                    std::holds_alternative<ELU>(theAct);
	const double a = isRelu ? std::sqrt(6.0 / static_cast<double>(nprev))
	                        : std::sqrt(6.0 / static_cast<double>(nprev + ncurr));
	for (uint o = 0; o < ncurr; ++o) {
		for (uint i = 0; i < nprev; ++i)
			theWeights[o * stride + i] = a * (2.0 * nnh::rand::uniform() - 1.0);
		theWeights[o * stride + nprev] = 0.0;
	}
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

vector<double>& Layer::propagate(const vector<double>& input, const double* preactSkip) {
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

	// Phase 1.5: pre-activation skip add.
	if (preactSkip)
		for (uint i = 0; i < ncurr; ++i)
			out[i] += preactSkip[i];

	// Phase 2: apply activation in a single vectorizable loop. std::visit
	// inlines applyActivation per-alternative so the compiler sees a
	// concrete activation type at the inner-loop call site.
	std::visit([out, this](const auto& a) { applyActivation(a, out, ncurr); }, theAct);

	// Phase 3: inverted dropout
	if (theTraining && theDropoutRate > 0.0) {
		const double scale = 1.0 / (1.0 - theDropoutRate);
		for (uint i = 0; i < ncurr; ++i) {
			theDropoutMask[i] = (nnh::rand::uniform() >= theDropoutRate) ? scale : 0.0;
			out[i] *= theDropoutMask[i];
		}
	}

	return theOutputs;
}

void Layer::applyDerivative(vector<double>& deltas) {
	std::visit(
	    [&](const auto& a) { applyDerivScale(a, theOutputs.data(), deltas.data(), ncurr); },
	    theAct);
	if (theTraining && theDropoutRate > 0.0)
		for (uint i = 0; i < ncurr; ++i)
			deltas[i] *= theDropoutMask[i];
}

const double* Layer::propagateBatch(const double* input, uint B, uint n_in,
                                    const double* preactSkip) {
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

	// Add bias to each row. Pre-pack biases into a contiguous buffer so
	// the inner loop over j has unit-stride loads on both sides; the
	// stride-load wt[j*stride+nprev] otherwise blocks vectorisation.
	theBiasBuf.resize(ncurr);
	for (uint j = 0; j < ncurr; ++j)
		theBiasBuf[j] = wt[j * stride + nprev];
	for (uint b = 0; b < B; ++b) {
		double* row = out + b * ncurr;
		const double* bias = theBiasBuf.data();
#pragma omp simd
		for (uint j = 0; j < ncurr; ++j)
			row[j] += bias[j];
	}
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
					theRunningMean[j] =
					    (1.0 - theBNMomentum) * theRunningMean[j] + theBNMomentum * mean;
					theRunningVar[j] =
					    (1.0 - theBNMomentum) * theRunningVar[j] + theBNMomentum * var;
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

	// Pre-activation skip add (after linear + bias + norm, before activation).
	if (preactSkip) {
		const uint total = B * ncurr;
		for (uint i = 0; i < total; ++i)
			out[i] += preactSkip[i];
	}

	// Apply activation to all B*ncurr elements. std::visit lets the compiler
	// inline the per-element activation kernel inside each variant branch.
	std::visit([out, B, this](const auto& a) { applyActivation(a, out, B * ncurr); }, theAct);

	// Inverted dropout for batch
	if (theTraining && theDropoutRate > 0.0) {
		const double scale = 1.0 / (1.0 - theDropoutRate);
		const uint total = B * ncurr;
		theBatchDropoutMask.resize(total);
		for (uint i = 0; i < total; ++i) {
			theBatchDropoutMask[i] = (nnh::rand::uniform() >= theDropoutRate) ? scale : 0.0;
			out[i] *= theBatchDropoutMask[i];
		}
	}

	return out;
}

void Layer::applyDerivativeBatch(uint B) {
	std::visit(
	    [B, this](const auto& a) {
		    applyDerivScale(a, theBatchOutputs.data(), theBatchLocalGradients.data(), B * ncurr);
	    },
	    theAct);
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

	// Bias gradients: column-sum of delta. Loop with the contiguous
	// dimension innermost so it vectorises, then scatter the strided
	// write to grad's bias column at the end.
	theBiasBuf.assign(ncurr, 0.0);
	double* bias_acc = theBiasBuf.data();
	for (uint b = 0; b < B; ++b) {
		const double* drow = delta + b * ncurr;
#pragma omp simd
		for (uint i = 0; i < ncurr; ++i)
			bias_acc[i] += drow[i];
	}
	for (uint i = 0; i < ncurr; ++i)
		grad[i * stride + nprev] += bias_acc[i];
}

void Layer::applyNormBackwardBatch(uint B) {
	if (theNormType == NormType::None) return;

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
