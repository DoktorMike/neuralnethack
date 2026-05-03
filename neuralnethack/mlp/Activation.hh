#ifndef __Activation_hh__
#define __Activation_hh__

#include <string>
#include <string_view>
#include <variant>

namespace MultiLayerPerceptron {

// Activation tag types. Replace the old Layer-subclass hierarchy: each tag
// carries only the parameters specific to that activation (LeakyReLU/ELU
// alpha), and dispatch is done via std::visit on Activation rather than
// virtual calls. The compiler can inline applyActivation per call site.
struct Sigmoid {};
struct TanH {};
struct Linear {};
struct ReLU {};
struct LeakyReLU {
	double alpha = 0.01;
};
struct ELU {
	double alpha = 1.0;
};

using Activation = std::variant<Sigmoid, TanH, Linear, ReLU, LeakyReLU, ELU>;

// String tag (logsig / tansig / purelin / relu / leakyrelu / elu) round-trip.
// activationFromTag throws std::invalid_argument on unknown tags.
Activation activationFromTag(std::string_view tag);
const std::string& activationToTag(const Activation& a);

// Scalar API. fire/firePrime/firePrimePrime take the local induced field;
// the *FromOutput variants take the cached activation y = fire(lif) and
// compute the derivative analytically (cheaper than recomputing fire).

double fire(Sigmoid, double lif);
double fire(TanH, double lif);
double fire(Linear, double lif);
double fire(ReLU, double lif);
double fire(LeakyReLU a, double lif);
double fire(ELU a, double lif);

double firePrime(Sigmoid, double lif);
double firePrime(TanH, double lif);
double firePrime(Linear, double lif);
double firePrime(ReLU, double lif);
double firePrime(LeakyReLU a, double lif);
double firePrime(ELU a, double lif);

double firePrimeFromOutput(Sigmoid, double y);
double firePrimeFromOutput(TanH, double y);
double firePrimeFromOutput(Linear, double y);
double firePrimeFromOutput(ReLU, double y);
double firePrimeFromOutput(LeakyReLU a, double y);
double firePrimeFromOutput(ELU a, double y);

double firePrimePrime(Sigmoid, double lif);
double firePrimePrime(TanH, double lif);
double firePrimePrime(Linear, double lif);
double firePrimePrime(ReLU, double lif);
double firePrimePrime(LeakyReLU a, double lif);
double firePrimePrime(ELU a, double lif);

double firePrimePrimeFromOutput(Sigmoid, double y);
double firePrimePrimeFromOutput(TanH, double y);
double firePrimePrimeFromOutput(Linear, double y);
double firePrimePrimeFromOutput(ReLU, double y);
double firePrimePrimeFromOutput(LeakyReLU a, double y);
double firePrimePrimeFromOutput(ELU a, double y);

// Batch API. applyActivation maps lif -> output in place (used after the
// linear+norm step in Layer::propagateBatch). applyDerivScale multiplies
// deltas[i] by f'(output[i]) (used in Layer::applyDerivativeBatch).
using ActivationSize = unsigned int;

void applyActivation(Sigmoid, double* out, ActivationSize n);
void applyActivation(TanH, double* out, ActivationSize n);
void applyActivation(Linear, double* out, ActivationSize n);
void applyActivation(ReLU, double* out, ActivationSize n);
void applyActivation(LeakyReLU a, double* out, ActivationSize n);
void applyActivation(ELU a, double* out, ActivationSize n);

void applyDerivScale(Sigmoid, const double* out, double* deltas, ActivationSize n);
void applyDerivScale(TanH, const double* out, double* deltas, ActivationSize n);
void applyDerivScale(Linear, const double* out, double* deltas, ActivationSize n);
void applyDerivScale(ReLU, const double* out, double* deltas, ActivationSize n);
void applyDerivScale(LeakyReLU a, const double* out, double* deltas, ActivationSize n);
void applyDerivScale(ELU a, const double* out, double* deltas, ActivationSize n);

} // namespace MultiLayerPerceptron

#endif
