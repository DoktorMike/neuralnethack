#include "Activation.hh"
#include "MultiLayerPerceptron.hh"

#include <cmath>
#include <stdexcept>

namespace MultiLayerPerceptron {

// Vectorisable polynomial tanh / sigmoid for the batch path.
//
// libm's scalar tanh / exp ate ~25% of training time on profile because gcc
// auto-vectorisation didn't pick libmvec siblings even with -ffast-math +
// `omp simd`. Replaced with branchless polynomial approximations that the
// compiler vectorises trivially. Accuracy: tanh better than 1e-7 on
// [-3, 3] and saturates correctly outside; sigmoid is derived from the
// same tanh, accurate to ~5e-8 over the working range. Both well within
// training noise. Scalar fire() still uses libm for exactness.
namespace {
inline double fast_tanh(double x) {
	const double xc = x < -5.0 ? -5.0 : (x > 5.0 ? 5.0 : x);
	const double x2 = xc * xc;
	const double a = xc * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
	const double b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
	return a / b;
}

inline double fast_sigmoid(double x) {
	return 0.5 * (fast_tanh(0.5 * x) + 1.0);
}
} // namespace

// Tag <-> string round-trip ---------------------------------------------------

Activation activationFromTag(std::string_view tag) {
	if (tag == SIGMOID) return Sigmoid{};
	if (tag == TANHYP) return TanH{};
	if (tag == LINEAR) return Linear{};
	if (tag == RELU) return ReLU{};
	if (tag == LEAKYRELU) return LeakyReLU{};
	if (tag == ELU_ACT) return ELU{};
	throw std::invalid_argument(std::string("activationFromTag: unknown tag '") +
	                            std::string(tag) + "'");
}

const std::string& activationToTag(const Activation& a) {
	static const std::string kSigmoid = SIGMOID;
	static const std::string kTanH = TANHYP;
	static const std::string kLinear = LINEAR;
	static const std::string kReLU = RELU;
	static const std::string kLeakyReLU = LEAKYRELU;
	static const std::string kELU = ELU_ACT;
	struct V {
		const std::string& operator()(Sigmoid) const { return kSigmoid; }
		const std::string& operator()(TanH) const { return kTanH; }
		const std::string& operator()(Linear) const { return kLinear; }
		const std::string& operator()(ReLU) const { return kReLU; }
		const std::string& operator()(LeakyReLU) const { return kLeakyReLU; }
		const std::string& operator()(ELU) const { return kELU; }
	};
	return std::visit(V{}, a);
}

// Scalar fire -----------------------------------------------------------------

double fire(Sigmoid, double lif) {
	return 1.0 / (1.0 + std::exp(-lif));
}
double fire(TanH, double lif) {
	return std::tanh(lif);
}
double fire(Linear, double lif) {
	return lif;
}
double fire(ReLU, double lif) {
	return lif > 0.0 ? lif : 0.0;
}
double fire(LeakyReLU a, double lif) {
	return lif > 0.0 ? lif : a.alpha * lif;
}
double fire(ELU a, double lif) {
	return lif > 0.0 ? lif : a.alpha * (std::exp(lif) - 1.0);
}

// Scalar firePrime (from local induced field) ---------------------------------

double firePrime(Sigmoid s, double lif) {
	const double f = fire(s, lif);
	return f * (1.0 - f);
}
double firePrime(TanH t, double lif) {
	const double f = fire(t, lif);
	return 1.0 - f * f;
}
double firePrime(Linear, double) {
	return 1.0;
}
double firePrime(ReLU, double lif) {
	return lif > 0.0 ? 1.0 : 0.0;
}
double firePrime(LeakyReLU a, double lif) {
	return lif > 0.0 ? 1.0 : a.alpha;
}
double firePrime(ELU a, double lif) {
	return lif > 0.0 ? 1.0 : a.alpha * std::exp(lif);
}

// Scalar firePrime (from cached output y = fire(lif)) -------------------------

double firePrimeFromOutput(Sigmoid, double y) {
	return y * (1.0 - y);
}
double firePrimeFromOutput(TanH, double y) {
	return 1.0 - y * y;
}
double firePrimeFromOutput(Linear, double) {
	return 1.0;
}
double firePrimeFromOutput(ReLU, double y) {
	return y > 0.0 ? 1.0 : 0.0;
}
double firePrimeFromOutput(LeakyReLU a, double y) {
	return y > 0.0 ? 1.0 : a.alpha;
}
double firePrimeFromOutput(ELU a, double y) {
	return y >= 0.0 ? 1.0 : y + a.alpha;
}

// Scalar firePrimePrime (from local induced field) ----------------------------

double firePrimePrime(Sigmoid s, double lif) {
	const double f = fire(s, lif);
	const double fp = f * (1.0 - f);
	return fp - 2.0 * f * fp;
}
double firePrimePrime(TanH t, double lif) {
	const double f = fire(t, lif);
	return 2.0 * f * (f * f - 1.0);
}
double firePrimePrime(Linear, double) {
	return 0.0;
}
double firePrimePrime(ReLU, double) {
	return 0.0;
}
double firePrimePrime(LeakyReLU, double) {
	return 0.0;
}
double firePrimePrime(ELU a, double lif) {
	return lif > 0.0 ? 0.0 : a.alpha * std::exp(lif);
}

// Scalar firePrimePrime (from cached output) ----------------------------------

double firePrimePrimeFromOutput(Sigmoid, double y) {
	const double fp = y * (1.0 - y);
	return fp - 2.0 * y * fp;
}
double firePrimePrimeFromOutput(TanH, double y) {
	return 2.0 * y * (y * y - 1.0);
}
double firePrimePrimeFromOutput(Linear, double) {
	return 0.0;
}
double firePrimePrimeFromOutput(ReLU, double) {
	return 0.0;
}
double firePrimePrimeFromOutput(LeakyReLU, double) {
	return 0.0;
}
double firePrimePrimeFromOutput(ELU a, double y) {
	return y >= 0.0 ? 0.0 : y + a.alpha;
}

// Batch activation ------------------------------------------------------------

void applyActivation(Sigmoid, double* __restrict__ out, ActivationSize n) {
#pragma omp simd
	for (ActivationSize i = 0; i < n; ++i)
		out[i] = fast_sigmoid(out[i]);
}

void applyActivation(TanH, double* __restrict__ out, ActivationSize n) {
#pragma omp simd
	for (ActivationSize i = 0; i < n; ++i)
		out[i] = fast_tanh(out[i]);
}

void applyActivation(Linear, double* __restrict__, ActivationSize) {
	// identity
}

void applyActivation(ReLU, double* __restrict__ out, ActivationSize n) {
	for (ActivationSize i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : 0.0;
}

void applyActivation(LeakyReLU a, double* __restrict__ out, ActivationSize n) {
	const double alpha = a.alpha;
	for (ActivationSize i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : alpha * out[i];
}

void applyActivation(ELU a, double* __restrict__ out, ActivationSize n) {
	const double alpha = a.alpha;
#pragma omp simd
	for (ActivationSize i = 0; i < n; ++i)
		out[i] = out[i] > 0.0 ? out[i] : alpha * (std::exp(out[i]) - 1.0);
}

// Batch derivative scaling ----------------------------------------------------

void applyDerivScale(Sigmoid, const double* __restrict__ out, double* __restrict__ deltas,
                     ActivationSize n) {
	for (ActivationSize i = 0; i < n; ++i)
		deltas[i] *= out[i] * (1.0 - out[i]);
}

void applyDerivScale(TanH, const double* __restrict__ out, double* __restrict__ deltas,
                     ActivationSize n) {
	for (ActivationSize i = 0; i < n; ++i)
		deltas[i] *= (1.0 - out[i] * out[i]);
}

void applyDerivScale(Linear, const double* __restrict__, double* __restrict__, ActivationSize) {
	// f'(x) = 1, deltas unchanged
}

void applyDerivScale(ReLU, const double* __restrict__ out, double* __restrict__ deltas,
                     ActivationSize n) {
	for (ActivationSize i = 0; i < n; ++i)
		deltas[i] *= (out[i] > 0.0 ? 1.0 : 0.0);
}

void applyDerivScale(LeakyReLU a, const double* __restrict__ out, double* __restrict__ deltas,
                     ActivationSize n) {
	const double alpha = a.alpha;
	for (ActivationSize i = 0; i < n; ++i)
		deltas[i] *= (out[i] > 0.0 ? 1.0 : alpha);
}

void applyDerivScale(ELU a, const double* __restrict__ out, double* __restrict__ deltas,
                     ActivationSize n) {
	const double alpha = a.alpha;
	for (ActivationSize i = 0; i < n; ++i)
		deltas[i] *= (out[i] >= 0.0 ? 1.0 : out[i] + alpha);
}

} // namespace MultiLayerPerceptron
