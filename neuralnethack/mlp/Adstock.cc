#include "Adstock.hh"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace MultiLayerPerceptron;
using std::vector;

Adstock::Adstock(uint channels, uint lags, uint passthrough, Kernel k)
    : theChannels(channels), theLags(lags), thePassthrough(passthrough), theKernel(k), theNBoxes(0),
      theSaturation(Saturation::None) {
	assert(channels > 0 && lags > 0);
	theParams.assign(nParams(), 0.0);
	theGradients.assign(nParams(), 0.0);
	theUpdates.assign(nParams(), 0.0);
	theW.assign(static_cast<size_t>(channels) * lags, 0.0);
	theDw.assign(static_cast<size_t>(channels) * nParamsPerChannel() * lags, 0.0);
	initParams();
}

Adstock::Adstock(uint channels, uint lags, uint passthrough, Kernel k, uint nBoxes, Saturation sat)
    : theChannels(channels), theLags(lags), thePassthrough(passthrough), theKernel(k),
      theNBoxes(nBoxes), theSaturation(sat) {
	assert(channels > 0 && lags > 0 && nBoxes > 0);
	theParams.assign(nParams(), 0.0);
	theGradients.assign(nParams(), 0.0);
	theUpdates.assign(nParams(), 0.0);
	theW.assign(static_cast<size_t>(nBoxes) * lags, 0.0);
	theDw.assign(static_cast<size_t>(nBoxes) * nParamsPerChannel() * lags, 0.0);
	thePi.assign(static_cast<size_t>(channels) * nBoxes, 0.0);
	initParams();
}

uint Adstock::nParams() const {
	const uint ppk = nParamsPerChannel();
	if (!boxed()) return theChannels * ppk;
	uint n = theNBoxes * ppk + theChannels * theNBoxes;
	if (theSaturation == Saturation::Hill) n += 2 * theNBoxes;
	return n;
}

void Adstock::killGradients() {
	theGradients.assign(theGradients.size(), 0.0);
}

void Adstock::initParams() {
	const uint ppk = nParamsPerChannel();
	theParams.assign(nParams(), 0.0);
	if (!boxed()) {
		for (uint c = 0; c < theChannels; ++c) {
			if (theKernel == Kernel::Geometric) {
				theParams[c] = 0.0; // sigmoid(0) = 0.5
			} else {
				theParams[c * ppk + 0] = std::log(2.0);                   // k = 2
				theParams[c * ppk + 1] = std::log(theLags / 3.0 + 1e-12); // scale = L/3
			}
		}
	} else {
		// Stagger the boxes fast -> slow so gradients can separate them
		// (identical boxes would receive identical gradients forever).
		for (uint k = 0; k < theNBoxes; ++k) {
			const double frac = (k + 1.0) / (theNBoxes + 1.0); // (0,1)
			if (theKernel == Kernel::Geometric) {
				// lambda_k spread over (0,1): rho = logit(frac)
				theParams[kernOff() + k] = std::log(frac / (1.0 - frac));
			} else {
				theParams[kernOff() + k * ppk + 0] = std::log(2.0); // shape 2
				// scale spread over the window
				theParams[kernOff() + k * ppk + 1] = std::log(theLags * frac / 2.0 + 1e-12);
			}
		}
		if (theSaturation == Saturation::Hill) {
			for (uint k = 0; k < theNBoxes; ++k) {
				theParams[sigOff() + k] = 0.0; // half-saturation 1
				theParams[nuOff() + k] = 0.0;  // exponent 1
			}
		}
		// logits stay 0: uniform routing over already-distinct boxes
	}
	theKernelsFresh = false;
}

// Normalized kernel weights + derivatives w.r.t. the unconstrained
// parameters, from one parameter set p (length ppk). Weights sum to 1
// over the window, so dw_l/dtheta = w_l (dlogg_l - sum_j w_j dlogg_j).
void Adstock::kernelFromParams(const double* p, double* w, double* dw) const {
	const uint L = theLags;

	if (theKernel == Kernel::Geometric) {
		const double rho = p[0];
		const double lambda = 1.0 / (1.0 + std::exp(-rho));
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
	const double kappa = p[0];
	const double sigma = p[1];
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

void Adstock::softmaxRouting(uint c, double* pi) const {
	const uint K = theNBoxes;
	const double* lg = theParams.data() + logitOff() + static_cast<size_t>(c) * K;
	double m = lg[0];
	for (uint k = 1; k < K; ++k)
		if (lg[k] > m) m = lg[k];
	double S = 0;
	for (uint k = 0; k < K; ++k) {
		pi[k] = std::exp((lg[k] - m) / theTau);
		S += pi[k];
	}
	for (uint k = 0; k < K; ++k)
		pi[k] /= S;
}

void Adstock::computeKernels() const {
	// Trainers mutate params() in place, so freshness = "caches were
	// computed from exactly these parameter values".
	if (theKernelsFresh && theCachedParams == theParams) return;
	const uint L = theLags, ppk = nParamsPerChannel();
	if (!boxed()) {
		for (uint c = 0; c < theChannels; ++c)
			kernelFromParams(theParams.data() + c * ppk, theW.data() + c * L,
			                 theDw.data() + static_cast<size_t>(c) * ppk * L);
	} else {
		for (uint k = 0; k < theNBoxes; ++k)
			kernelFromParams(theParams.data() + kernOff() + k * ppk, theW.data() + k * L,
			                 theDw.data() + static_cast<size_t>(k) * ppk * L);
		for (uint c = 0; c < theChannels; ++c)
			softmaxRouting(c, thePi.data() + static_cast<size_t>(c) * theNBoxes);
	}
	theCachedParams = theParams;
	theKernelsFresh = true;
}

vector<double> Adstock::kernelWeights(uint c) const {
	assert(c < theChannels);
	computeKernels();
	const uint L = theLags;
	if (!boxed()) return vector<double>(theW.begin() + c * L, theW.begin() + (c + 1) * L);
	vector<double> w(L, 0.0);
	const double* pi = thePi.data() + static_cast<size_t>(c) * theNBoxes;
	for (uint k = 0; k < theNBoxes; ++k)
		for (uint l = 0; l < L; ++l)
			w[l] += pi[k] * theW[k * L + l];
	return w;
}

vector<double> Adstock::naturalParams() const {
	const uint ppk = nParamsPerChannel();
	vector<double> out;
	auto pushKernel = [&](const double* p) {
		if (theKernel == Kernel::Geometric) {
			out.push_back(1.0 / (1.0 + std::exp(-p[0])));
		} else {
			out.push_back(std::exp(p[0]));
			out.push_back(std::exp(p[1]));
		}
	};
	if (!boxed()) {
		for (uint c = 0; c < theChannels; ++c)
			pushKernel(theParams.data() + c * ppk);
		return out;
	}
	for (uint k = 0; k < theNBoxes; ++k)
		pushKernel(theParams.data() + kernOff() + k * ppk);
	if (theSaturation == Saturation::Hill) {
		for (uint k = 0; k < theNBoxes; ++k)
			out.push_back(std::exp(theParams[sigOff() + k]));
		for (uint k = 0; k < theNBoxes; ++k)
			out.push_back(std::exp(theParams[nuOff() + k]));
	}
	return out;
}

vector<double> Adstock::routingProbs(uint c) const {
	assert(boxed() && c < theChannels);
	computeKernels();
	const double* pi = thePi.data() + static_cast<size_t>(c) * theNBoxes;
	return vector<double>(pi, pi + theNBoxes);
}

vector<uint> Adstock::boxAssignments() const {
	assert(boxed());
	computeKernels();
	vector<uint> a(theChannels, 0);
	for (uint c = 0; c < theChannels; ++c) {
		const double* pi = thePi.data() + static_cast<size_t>(c) * theNBoxes;
		for (uint k = 1; k < theNBoxes; ++k)
			if (pi[k] > pi[a[c]]) a[c] = k;
	}
	return a;
}

vector<double> Adstock::boxKernel(uint k) const {
	assert(boxed() && k < theNBoxes);
	computeKernels();
	return vector<double>(theW.begin() + k * theLags, theW.begin() + (k + 1) * theLags);
}

double Adstock::boxSaturation(uint k) const {
	assert(boxed() && theSaturation == Saturation::Hill && k < theNBoxes);
	return std::exp(theParams[sigOff() + k]);
}

double Adstock::boxHillExponent(uint k) const {
	assert(boxed() && theSaturation == Saturation::Hill && k < theNBoxes);
	return std::exp(theParams[nuOff() + k]);
}

// hill(a; s, n) = a^n / (a^n + s^n) for a >= 0.
double Adstock::hill(double a, double s, double n) {
	if (a <= 0.0) return 0.0;
	const double an = std::pow(a, n), sn = std::pow(s, n);
	return an / (an + sn);
}

// Partials on the natural scale; all zero at a = 0 by continuity of the
// gradient contributions we need (a > 0 in practice: spend is
// non-negative and kernels are positive).
void Adstock::hillPartials(double a, double s, double n, double& dha, double& dhs, double& dhn) {
	if (a <= 0.0) {
		dha = dhs = dhn = 0.0;
		return;
	}
	const double an = std::pow(a, n), sn = std::pow(s, n);
	const double D = an + sn, D2 = D * D;
	dha = n * std::pow(a, n - 1.0) * sn / D2;
	dhs = -an * n * std::pow(s, n - 1.0) / D2;
	dhn = an * sn * (std::log(a) - std::log(s)) / D2;
}

void Adstock::transform(const double* in, double* out) const {
	computeKernels();
	const uint L = theLags;
	if (!boxed()) {
		for (uint c = 0; c < theChannels; ++c) {
			const double* w = theW.data() + c * L;
			const double* x = in + c * L;
			double s = 0.0;
			for (uint l = 0; l < L; ++l)
				s += w[l] * x[l];
			out[c] = s;
		}
	} else {
		const uint K = theNBoxes;
		const bool hillOn = theSaturation == Saturation::Hill;
		for (uint c = 0; c < theChannels; ++c) {
			const double* x = in + c * L;
			const double* pi = thePi.data() + static_cast<size_t>(c) * K;
			double a = 0.0;
			for (uint k = 0; k < K; ++k) {
				const double* w = theW.data() + k * L;
				double dot = 0.0;
				for (uint l = 0; l < L; ++l)
					dot += w[l] * x[l];
				a += pi[k] * dot;
			}
			if (hillOn) {
				double h = 0.0;
				for (uint k = 0; k < K; ++k)
					h += pi[k] * hill(a, std::exp(theParams[sigOff() + k]),
					                  std::exp(theParams[nuOff() + k]));
				out[c] = h;
			} else {
				out[c] = a;
			}
		}
	}
	for (uint p = 0; p < thePassthrough; ++p)
		out[theChannels + p] = in[theChannels * L + p];
}

const double* Adstock::transformBatch(const double* in, uint B) {
	computeKernels();
	const uint din = inputDim(), dout = outputDim(), L = theLags;
	theOutputs.resize(static_cast<size_t>(B) * dout);
	if (!boxed()) {
		for (uint b = 0; b < B; ++b)
			transform(in + static_cast<size_t>(b) * din,
			          theOutputs.data() + static_cast<size_t>(b) * dout);
		return theOutputs.data();
	}
	// Boxed: also cache per-box dots and the mixed dot a_c for backward.
	const uint K = theNBoxes;
	const bool hillOn = theSaturation == Saturation::Hill;
	theA.resize(static_cast<size_t>(B) * theChannels);
	theDot.resize(static_cast<size_t>(B) * theChannels * K);
	for (uint b = 0; b < B; ++b) {
		const double* row = in + static_cast<size_t>(b) * din;
		double* orow = theOutputs.data() + static_cast<size_t>(b) * dout;
		for (uint c = 0; c < theChannels; ++c) {
			const double* x = row + c * L;
			const double* pi = thePi.data() + static_cast<size_t>(c) * K;
			double* dot = theDot.data() + (static_cast<size_t>(b) * theChannels + c) * K;
			double a = 0.0;
			for (uint k = 0; k < K; ++k) {
				const double* w = theW.data() + k * L;
				double d = 0.0;
				for (uint l = 0; l < L; ++l)
					d += w[l] * x[l];
				dot[k] = d;
				a += pi[k] * d;
			}
			theA[static_cast<size_t>(b) * theChannels + c] = a;
			if (hillOn) {
				double h = 0.0;
				for (uint k = 0; k < K; ++k)
					h += pi[k] * hill(a, std::exp(theParams[sigOff() + k]),
					                  std::exp(theParams[nuOff() + k]));
				orow[c] = h;
			} else {
				orow[c] = a;
			}
		}
		for (uint p = 0; p < thePassthrough; ++p)
			orow[theChannels + p] = row[theChannels * L + p];
	}
	return theOutputs.data();
}

void Adstock::accumulateGradients(const double* rawIn, const double* outDelta, uint B) {
	computeKernels();
	const uint L = theLags, ppk = nParamsPerChannel(), din = inputDim(), dout = outputDim();

	if (!boxed()) {
		for (uint c = 0; c < theChannels; ++c) {
			const double* dwc = theDw.data() + static_cast<size_t>(c) * ppk * L;
			for (uint p = 0; p < ppk; ++p) {
				const double* dw = dwc + p * L;
				double acc = 0.0;
				for (uint b = 0; b < B; ++b) {
					const double* x = rawIn + static_cast<size_t>(b) * din + c * L;
					double s = 0.0;
					for (uint l = 0; l < L; ++l)
						s += dw[l] * x[l];
					acc += outDelta[static_cast<size_t>(b) * dout + c] * s;
				}
				theGradients[c * ppk + p] += acc;
			}
		}
		return;
	}

	// Boxed backward. Uses the transformBatch caches (a_c, per-box dots).
	const uint K = theNBoxes;
	const bool hillOn = theSaturation == Saturation::Hill;
	assert(theA.size() == static_cast<size_t>(B) * theChannels);

	// Natural-scale hill params, hoisted
	vector<double> sNat(K), nNat(K);
	if (hillOn)
		for (uint k = 0; k < K; ++k) {
			sNat[k] = std::exp(theParams[sigOff() + k]);
			nNat[k] = std::exp(theParams[nuOff() + k]);
		}

	vector<double> hk(K), dha_k(K), dhs_k(K), dhn_k(K), contrib(K);
	for (uint b = 0; b < B; ++b) {
		const double* row = rawIn + static_cast<size_t>(b) * din;
		for (uint c = 0; c < theChannels; ++c) {
			const double g = outDelta[static_cast<size_t>(b) * dout + c];
			if (g == 0.0) continue;
			const double* x = row + c * L;
			const double* pi = thePi.data() + static_cast<size_t>(c) * K;
			const double* dot = theDot.data() + (static_cast<size_t>(b) * theChannels + c) * K;
			const double a = theA[static_cast<size_t>(b) * theChannels + c];

			// d out / d a  and per-box hill terms
			double dha = 1.0;
			if (hillOn) {
				dha = 0.0;
				for (uint k = 0; k < K; ++k) {
					hillPartials(a, sNat[k], nNat[k], dha_k[k], dhs_k[k], dhn_k[k]);
					hk[k] = hill(a, sNat[k], nNat[k]);
					dha += pi[k] * dha_k[k];
					// hill param grads (chain exp): sigma, nu
					theGradients[sigOff() + k] += g * pi[k] * dhs_k[k] * sNat[k];
					theGradients[nuOff() + k] += g * pi[k] * dhn_k[k] * nNat[k];
				}
			}

			// box kernel params: d a / d theta_kp = pi_k * (dw_kp . x)
			const double gd = g * dha;
			for (uint k = 0; k < K; ++k) {
				if (pi[k] == 0.0) continue;
				const double* dwk = theDw.data() + static_cast<size_t>(k) * ppk * L;
				for (uint p = 0; p < ppk; ++p) {
					const double* dw = dwk + p * L;
					double s = 0.0;
					for (uint l = 0; l < L; ++l)
						s += dw[l] * x[l];
					theGradients[kernOff() + k * ppk + p] += gd * pi[k] * s;
				}
			}

			// routing logits: d out / d pi_k = [hill_k(a)] + dha * dot_k,
			// then the softmax Jacobian with temperature.
			double mean = 0.0;
			for (uint k = 0; k < K; ++k) {
				contrib[k] = (hillOn ? hk[k] : 0.0) + dha * dot[k];
				mean += contrib[k] * pi[k];
			}
			double* glog = theGradients.data() + logitOff() + static_cast<size_t>(c) * K;
			for (uint m = 0; m < K; ++m)
				glog[m] += g * (pi[m] / theTau) * (contrib[m] - mean);
		}
	}
}

void Adstock::applyEntropyPenaltyGradient() {
	if (!boxed() || theEntropyBeta <= 0.0) return;
	computeKernels();
	const uint K = theNBoxes;
	// d H(pi_c) / d logit_cm = -(pi_m / tau) (log pi_m + H_c)
	for (uint c = 0; c < theChannels; ++c) {
		const double* pi = thePi.data() + static_cast<size_t>(c) * K;
		double H = 0.0;
		for (uint k = 0; k < K; ++k)
			if (pi[k] > 0.0) H -= pi[k] * std::log(pi[k]);
		double* glog = theGradients.data() + logitOff() + static_cast<size_t>(c) * K;
		for (uint m = 0; m < K; ++m) {
			const double lp = pi[m] > 0.0 ? std::log(pi[m]) : 0.0;
			glog[m] += theEntropyBeta * (-(pi[m] / theTau) * (lp + H));
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
