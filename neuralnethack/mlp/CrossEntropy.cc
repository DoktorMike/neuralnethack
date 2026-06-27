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

CrossEntropy::CrossEntropy(Mlp& mlp, DataSet& dset) : Error(mlp, dset) {}

CrossEntropy::CrossEntropy(std::unique_ptr<Mlp> mlp, DataSet& dset) : Error(std::move(mlp), dset) {}

CrossEntropy::~CrossEntropy() {}

double CrossEntropy::gradient(Mlp& mlp, DataSet& dset) {
	theMlp = &mlp;
	theDset = &dset;
	return gradient();
}

double CrossEntropy::gradient() {
	assert(theDset != 0 && theMlp != 0);

	killGradients();

	const uint bs = theDset->size();
	const uint nOut = theMlp->layer(theMlp->nLayers() - 1).nNeurons();

	// Pack dataset into reusable Error-owned batch matrices
	packBatch(*theDset);

	// Optional per-class weighting (empty pw => uniform, denom == bs)
	vector<double> pw;
	double denom;
	patternWeights(bs, nOut, pw, denom);

	// Batch forward pass (one GEMM per layer)
	const double* batchOut = theMlp->propagateBatch(theInputMatrix.data(), bs);

	// Per-layer accumulator for skip-connection deltas. See SummedSquare.cc
	// for the derivation; the same routing applies here.
	vector<vector<double>> skipDelta(theMlp->nLayers());

	// Compute output-layer local gradients: delta = target - output
	// (CrossEntropy + sigmoid: derivative cancels, the result is dL/dz_eff)
	Layer& last = (*theMlp)[theMlp->nLayers() - 1];
	const uint lastIdx = theMlp->nLayers() - 1;
	last.batchLocalGradients().resize(bs * nOut);
	{
		double* delta = last.batchLocalGradients().data();
		const double* t = theTargetMatrix.data();
		const double* o = batchOut;
		for (uint i = 0; i < bs * nOut; ++i)
			delta[i] = t[i] - o[i];
		if (!pw.empty())
			for (uint b = 0; b < bs; ++b)
				for (uint j = 0; j < nOut; ++j)
					delta[b * nOut + j] *= pw[b];
	}
	if (theMlp->skipFrom(lastIdx) >= 0) {
		int src = theMlp->skipFrom(lastIdx);
		auto& bin = skipDelta[src];
		const auto& clg = last.batchLocalGradients();
		if (bin.empty())
			bin = clg;
		else
			for (uint k = 0; k < bin.size(); ++k)
				bin[k] += clg[k];
	}

	// Batch backpropagate deltas through hidden layers (one GEMM per layer)
	for (int l = theMlp->size() - 1; l > 0; --l) {
		Layer& curr = (*theMlp)[l - 1];
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
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, nc, nn, 1.0, nlg, nn, wt,
		            nextStride, 0.0, clg, nc);
#else
		for (uint b = 0; b < bs; ++b) {
			for (uint j = 0; j < nc; ++j) {
				double err = 0;
				for (uint k = 0; k < nn; ++k)
					err += nlg[b * nn + k] * wt[k * nextStride + j];
				clg[b * nc + j] = err;
			}
		}
#endif
		if (!skipDelta[l - 1].empty()) {
			const auto& bin = skipDelta[l - 1];
			for (uint k = 0; k < bin.size(); ++k)
				clg[k] += bin[k];
		}
		curr.applyDerivativeBatch(bs);
		curr.applyNormBackwardBatch(bs);
		if (theMlp->skipFrom(l - 1) >= 0) {
			int src = theMlp->skipFrom(l - 1);
			auto& bin = skipDelta[src];
			if (bin.empty())
				bin.assign(clg, clg + bs * nc);
			else
				for (uint k = 0; k < bin.size(); ++k)
					bin[k] += clg[k];
		}
	}

	// Batch gradient accumulation (one GEMM per layer)
	(*theMlp)[0].accumulateGradientsBatch(theInputMatrix.data(), bs);
	for (uint l = 1; l < theMlp->nLayers(); ++l)
		(*theMlp)[l].accumulateGradientsBatch((*theMlp)[l - 1].batchOutputs().data(), bs);

	// Compute total error
	double err = 0;
	{
		const double* o = batchOut;
		const double* t = theTargetMatrix.data();
		const double power = -20;
		const double tiny = exp(power);
		for (uint b = 0; b < bs; ++b) {
			double pe = 0;
			if (nOut == 1) {
				if (t[b] == 0.0)
					pe += (1.0 - o[b] > tiny) ? log(1.0 - o[b]) : power;
				else
					pe += (o[b] > tiny) ? log(o[b]) : power;
			} else {
				for (uint j = 0; j < nOut; ++j) {
					uint idx = b * nOut + j;
					if (t[idx] != 0.0) pe += (o[idx] > tiny) ? log(o[idx]) : power;
				}
			}
			err += pw.empty() ? pe : pw[b] * pe;
		}
	}

	// Divide gradients by -denom (== -bs when unweighted) and apply weight elim
	for (uint l = 0; l < theMlp->nLayers(); ++l) {
		Layer& layer = theMlp->layer(l);
		vector<double>& g = layer.gradients();
		div(g, -denom);
		if (theWeightElimOn == true)
			weightElimGradLayer(g, layer.weights(), layer.nNeurons(), layer.nPrevious());
		if (layer.normType() != NormType::None) {
			div(layer.gammaGradients(), -denom);
			div(layer.betaGradients(), -denom);
		}
	}

	return -err / denom;
}

double CrossEntropy::outputError(Mlp& mlp, DataSet& dset) {
	theMlp = &mlp;
	theDset = &dset;
	return outputError();
}

double CrossEntropy::outputError() const {
	assert(theDset != 0 && theMlp != 0);
	const uint bs = theDset->size();
	const uint nOut = theMlp->layer(theMlp->nLayers() - 1).nNeurons();

	packBatch(*theDset);
	const double* o = theMlp->propagateBatch(theInputMatrix.data(), bs);
	const double* t = theTargetMatrix.data();

	const double power = -20;
	const double tiny = exp(power);
	double err = 0;
	for (uint b = 0; b < bs; ++b) {
		if (nOut == 1) {
			if (t[b] == 0.0)
				err += (1.0 - o[b] > tiny) ? log(1.0 - o[b]) : power;
			else
				err += (o[b] > tiny) ? log(o[b]) : power;
		} else {
			for (uint j = 0; j < nOut; ++j) {
				const uint idx = b * nOut + j;
				if (t[idx] != 0.0) err += (o[idx] > tiny) ? log(o[idx]) : power;
			}
		}
	}
	return -err / bs;
}

// PRIVATE--------------------------------------------------------------------//

CrossEntropy::CrossEntropy(const CrossEntropy& sse) : Error(*(sse.theMlp), *(sse.theDset)) {
	*this = sse;
}

CrossEntropy& CrossEntropy::operator=(const CrossEntropy& sse) {
	if (this != &sse) {
	}
	return *this;
}

void CrossEntropy::localGradient(Layer& ol, const vector<double>& out, const vector<double>& dout) {
	assert(out.size() == ol.size() && dout.size() == out.size());
	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itdo = dout.begin();
	for (uint i = 0; i < ol.nNeurons(); ++i, ++ito, ++itdo)
		ol.localGradients(i) = (*itdo - *ito);
}

void CrossEntropy::backpropagate() {
	for (int i = theMlp->size() - 1; i > 0; --i)
		localGradient((*theMlp)[i - 1], (*theMlp)[i]);
}

void CrossEntropy::localGradient(Layer& curr, Layer& next) {
	const uint nc = curr.nNeurons();
	const uint nn = next.nNeurons();
	double* __restrict__ clg = curr.localGradients().data();
	const double* __restrict__ nlg = next.localGradients().data();
	for (uint j = 0; j < nc; ++j) {
		double err = 0;
		for (uint i = 0; i < nn; ++i)
			err += nlg[i] * next.weights(i, j);
		clg[j] = err;
	}
	curr.applyDerivative(curr.localGradients());
}

void CrossEntropy::gradient(Layer& first, vector<double>& in) {
	for (uint i = 0; i < first.size(); ++i) {
		for (uint j = 0; j < in.size(); ++j)
			first.gradients(i, j) = first.localGradients(i) * in[j];
		first.gradients(i, in.size()) = first.localGradients(i);
	}
}

void CrossEntropy::gradientBatch(Layer& first, vector<double>& in) {
	for (uint i = 0; i < first.size(); ++i) {
		for (uint j = 0; j < in.size(); ++j)
			first.gradients(i, j) += first.localGradients(i) * in[j];
		first.gradients(i, in.size()) += first.localGradients(i);
	}
}

void CrossEntropy::gradient(Layer& curr, Layer& prev) {
	for (uint i = 0; i < curr.size(); ++i) {
		for (uint j = 0; j < prev.size(); ++j)
			curr.gradients(i, j) = curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) = curr.localGradients(i);
	}
}

void CrossEntropy::gradientBatch(Layer& curr, Layer& prev) {
	for (uint i = 0; i < curr.size(); ++i) {
		for (uint j = 0; j < prev.size(); ++j)
			curr.gradients(i, j) += curr.localGradients(i) * prev.outputs(j);
		curr.gradients(i, prev.size()) += curr.localGradients(i);
	}
}

double CrossEntropy::outputError(const vector<double>& out, const vector<double>& dout) const {
	assert(out.size() == dout.size());

	double power = -20;
	double tiny = exp(power);

	vector<double>::const_iterator ito = out.begin();
	vector<double>::const_iterator itd = dout.begin();
	if (dout.size() == 1) {
		if (*itd == 0.0)
			return (1.0 - *ito > tiny) ? log(1.0 - *ito) : power;
		else
			return (*ito > tiny) ? log(*ito) : power;
	}

	double e = 0;
	for (; ito != out.end(); ++ito, ++itd) {
		if (*itd == 0.0)
			e += 0;
		else
			e += (*ito > tiny) ? log(*ito) : power;
	}
	return e;
}

void CrossEntropy::killGradients() {
	for (uint i = 0; i < theMlp->nLayers(); ++i) {
		Layer& l = theMlp->layer(i);
		vector<double>& g = l.gradients();
		g.assign(g.size(), 0);
		if (l.normType() != NormType::None) {
			l.gammaGradients().assign(l.gammaGradients().size(), 0);
			l.betaGradients().assign(l.betaGradients().size(), 0);
		}
	}
}
