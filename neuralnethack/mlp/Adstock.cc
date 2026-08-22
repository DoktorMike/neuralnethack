#include "Adstock.hh"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace MultiLayerPerceptron;
using std::vector;

Adstock::Adstock(uint channels, uint lags, uint passthrough, Kernel k)
    : theChannels(channels), theLags(lags), thePassthrough(passthrough), theKernel(k),
      theParams(nParams(), 0.0), theGradients(nParams(), 0.0), theUpdates(nParams(), 0.0),
      theW(static_cast<size_t>(channels) * lags, 0.0),
      theDw(static_cast<size_t>(channels) * nParamsPerChannel() * lags, 0.0) {
	assert(channels > 0 && lags > 0);
	initParams();
}

void Adstock::killGradients() {
	theGradients.assign(theGradients.size(), 0.0);
}

void Adstock::initParams() {
	const uint ppc = nParamsPerChannel();
	for (uint c = 0; c < theChannels; ++c) {
		if (theKernel == Kernel::Geometric) {
			theParams[c] = 0.0; // sigmoid(0) = 0.5
		} else {
			theParams[c * ppc + 0] = std::log(2.0);              // k = 2
			theParams[c * ppc + 1] = std::log(theLags / 3.0 + 1e-12); // scale = L/3
		}
	}
	theKernelsFresh = false;
}

// Per-channel kernel weights + derivatives w.r.t. the unconstrained
// parameters. Weights are normalized to sum to 1 over the window, so
// dw_l/dtheta = w_l * (dlogg_l/dtheta - sum_j w_j dlogg_j/dtheta).
void Adstock::channelKernel(uint c, double* w, double* dw) const {
	const uint L = theLags;
	const uint ppc = nParamsPerChannel();

	if (theKernel == Kernel::Geometric) {
		const double rho = theParams[c];
		const double lambda = 1.0 / (1.0 + std::exp(-rho));
		// g_l = lambda^l; S = sum g; S' = sum l lambda^(l-1)
		double S = 0.0, Sp = 0.0, pl = 1.0 /* lambda^l */, plm = 0.0 /* l*lambda^(l-1) */;
		for (uint l = 0; l < L; ++l) {
			w[l] = pl;
			dw[l] = plm; // temporarily dg_l/dlambda
			S += pl;
			Sp += plm;
			plm = (l + 1) * pl;
			pl *= lambda;
		}
		const double sig = lambda * (1.0 - lambda); // dlambda/drho
		for (uint l = 0; l < L; ++l) {
			const double g = w[l], dg = dw[l];
			dw[l] = sig * (dg * S - g * Sp) / (S * S);
			w[l] = g / S;
		}
		return;
	}

	// Weibull: k = exp(kappa), s = exp(sigma); z_l = (l+1)/s, u_l = log z_l
	// log g_l = (k-1) u_l - z_l^k
	// dlogg/dkappa = k * u_l * (1 - z_l^k)
	// dlogg/dsigma = -(k-1) + k z_l^k
	const double kappa = theParams[c * ppc + 0];
	const double sigma = theParams[c * ppc + 1];
	const double k = std::exp(kappa);
	vector<double> logg(L), dk(L), ds(L);
	double maxlg = -1e300;
	for (uint l = 0; l < L; ++l) {
		const double u = std::log(static_cast<double>(l + 1)) - sigma;
		const double zk = std::exp(k * u); // z^k
		logg[l] = (k - 1.0) * u - zk;
		dk[l] = k * u * (1.0 - zk);
		ds[l] = -(k - 1.0) + k * zk;
		if (logg[l] > maxlg) maxlg = logg[l];
	}
	double S = 0.0;
	for (uint l = 0; l < L; ++l) {
		w[l] = std::exp(logg[l] - maxlg);
		S += w[l];
	}
	double mk = 0.0, ms = 0.0; // sum_j w_j dlogg_j
	for (uint l = 0; l < L; ++l) {
		w[l] /= S;
		mk += w[l] * dk[l];
		ms += w[l] * ds[l];
	}
	double* dwk = dw;
	double* dws = dw + L;
	for (uint l = 0; l < L; ++l) {
		dwk[l] = w[l] * (dk[l] - mk);
		dws[l] = w[l] * (ds[l] - ms);
	}
}

void Adstock::computeKernels() const {
	// Trainers mutate params() in place, so freshness = "kernels were
	// computed from exactly these parameter values".
	if (theKernelsFresh && theCachedParams == theParams) return;
	const uint L = theLags, ppc = nParamsPerChannel();
	for (uint c = 0; c < theChannels; ++c)
		channelKernel(c, theW.data() + c * L, theDw.data() + static_cast<size_t>(c) * ppc * L);
	theCachedParams = theParams;
	theKernelsFresh = true;
}

vector<double> Adstock::kernelWeights(uint c) const {
	assert(c < theChannels);
	vector<double> w(theLags), dw(static_cast<size_t>(nParamsPerChannel()) * theLags);
	channelKernel(c, w.data(), dw.data());
	return w;
}

void Adstock::transform(const double* in, double* out) const {
	computeKernels();
	const uint L = theLags;
	for (uint c = 0; c < theChannels; ++c) {
		const double* w = theW.data() + c * L;
		const double* x = in + c * L;
		double s = 0.0;
		for (uint l = 0; l < L; ++l)
			s += w[l] * x[l];
		out[c] = s;
	}
	for (uint p = 0; p < thePassthrough; ++p)
		out[theChannels + p] = in[theChannels * L + p];
}

const double* Adstock::transformBatch(const double* in, uint B) {
	computeKernels();
	const uint din = inputDim(), dout = outputDim();
	theOutputs.resize(static_cast<size_t>(B) * dout);
	for (uint b = 0; b < B; ++b)
		transform(in + static_cast<size_t>(b) * din, theOutputs.data() + static_cast<size_t>(b) * dout);
	return theOutputs.data();
}

void Adstock::accumulateGradients(const double* rawIn, const double* outDelta, uint B) {
	computeKernels();
	const uint L = theLags, ppc = nParamsPerChannel(), din = inputDim(), dout = outputDim();
	for (uint c = 0; c < theChannels; ++c) {
		const double* dwc = theDw.data() + static_cast<size_t>(c) * ppc * L;
		for (uint p = 0; p < ppc; ++p) {
			const double* dw = dwc + p * L;
			double acc = 0.0;
			for (uint b = 0; b < B; ++b) {
				const double* x = rawIn + static_cast<size_t>(b) * din + c * L;
				double s = 0.0;
				for (uint l = 0; l < L; ++l)
					s += dw[l] * x[l];
				acc += outDelta[static_cast<size_t>(b) * dout + c] * s;
			}
			theGradients[c * ppc + p] += acc;
		}
	}
}

std::string MultiLayerPerceptron::kernelToTag(Adstock::Kernel k) {
	return k == Adstock::Kernel::Geometric ? "geometric" : "weibull";
}

Adstock::Kernel MultiLayerPerceptron::kernelFromTag(const std::string& tag) {
	if (tag == "geometric") return Adstock::Kernel::Geometric;
	if (tag == "weibull") return Adstock::Kernel::Weibull;
	throw std::invalid_argument("unknown adstock kernel: " + tag);
}
