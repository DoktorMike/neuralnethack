#ifndef __Layer_hh__
#define __Layer_hh__

#include "../Random.hh"
#include "Activation.hh"
#include "MultiLayerPerceptron.hh"

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

namespace MultiLayerPerceptron {
/**A class representing a layer in an Mlp.
 * A Layer has a number of neurons with an activation function held as a
 * std::variant tag (see Activation.hh). The Layer knows the number of
 * neurons contained in itself and its predecessor.
 * \sa Mlp, Activation.
 */
class Layer {
  public:
	/**Constructor from a string activation tag (logsig / tansig / ...). */
	Layer(const uint nc, const uint np, std::string t);

	/**Constructor from an Activation variant directly. */
	Layer(const uint nc, const uint np, Activation act);

	// ACCESSOR AND MUTATOR FUNCTIONS

	/**Index operator: weight at index i. */
	double& operator[](const uint i);

	/**Get the weights leading into this layer.
	 * \return the weight vector.
	 */
	std::vector<double>& weights();
	const std::vector<double>& weights() const;

	/**Set the weights to those in w.
	 * \param w the weights to assign from.
	 */
	void weights(const std::vector<double>& w);

	/**Returns the output from this layer.
	 * Not including the bias!
	 * \return the outputs from this layer.
	 */
	std::vector<double>& outputs();
	const std::vector<double>& outputs() const;

	/**Returns the local gradients in this layer.
	 * \return the local gradients in this layer.
	 */
	std::vector<double>& localGradients();

	/**Return the gradients leading into this layer.
	 * \return the gradients leading into this layer.
	 */
	std::vector<double>& gradients();
	const std::vector<double>& gradients() const;

	/**Return the weight updates leading into this layer.
	 * \return the wight updates leading into this layer.
	 */
	std::vector<double>& weightUpdates();

	// ACCESSOR FUNCTIONS

	double& weights(const uint i, const uint j);
	double& weights(const uint i);
	double& outputs(const uint i);
	double& localGradients(const uint i);
	double& gradients(const uint i, const uint j);
	double& gradients(const uint i);
	double& weightUpdates(const uint i, const uint j);
	double& weightUpdates(const uint i);

	// COUNTS AND CRAP

	uint nWeights() const;
	uint nNeurons() const;
	uint nPrevious() const;
	uint size() const;

	/**The activation tag for this layer (variant over Sigmoid, TanH, ...). */
	const Activation& activation() const { return theAct; }
	Activation& activation() { return theAct; }

	/**The string identifier for this layer's activation (used for serialisation). */
	const std::string& type() const { return theType; }

	// PRINTS

	void printWeights(std::ostream& os) const;
	void printLocalGradients(std::ostream& os) const;
	void printGradients(std::ostream& os) const;

	// UTILS

	/**Weight initialisation scheme.
	 *  - LegacyUniform: U(-0.5, 0.5) for all weights and biases. Pre-Glorot
	 *    behaviour, kept for back-compat with serialised models / older
	 *    benchmarks.
	 *  - Glorot: Glorot/Xavier uniform, U(-a, a) with a = sqrt(6/(n_in+n_out)).
	 *    Biases initialised to zero. Default for new Mlp construction.
	 */
	enum class InitScheme { LegacyUniform, Glorot };

	void initScheme(InitScheme s) { theInitScheme = s; }
	InitScheme initScheme() const { return theInitScheme; }

	/**Assign new random weights to the weight vector. */
	void regenerateWeights();

	// Activation API. Scalar overloads dispatch via std::visit on theAct.
	// fire(uint i) returns the cached output directly (identical across all
	// activations, since outputs[] stores the activation result post-propagate).

	double fire(double lif) const;
	double fire(const uint i) const;
	double firePrime(double lif) const;
	double firePrime(const uint i) const;
	double firePrimePrime(double lif) const;
	double firePrimePrime(const uint i) const;

	/**Propagates an input pattern through this layer.
	 * Note that the bias should not be included in the parameter
	 * since it is explicitly included later.
	 * \param input the input to propagate.
	 * \param preactSkip optional pre-activation skip-add buffer of size
	 *        nNeurons(). Added between the linear+norm step and the
	 *        activation. Pass nullptr (default) for no skip.
	 * \return the outputs from this Layer.
	 */
	std::vector<double>& propagate(const std::vector<double>& input,
	                               const double* preactSkip = nullptr);

	/**Apply the activation derivative to a vector of deltas in batch.
	 * Computes deltas[i] *= f'(outputs[i]) for all neurons.
	 * \param deltas the vector to scale by the derivative.
	 */
	void applyDerivative(std::vector<double>& deltas);

	/**Propagate a batch of inputs through this layer using GEMM. */
	const double* propagateBatch(const double* input, uint B, uint n_in,
	                             const double* preactSkip = nullptr);

	void applyDerivativeBatch(uint B);
	void accumulateGradientsBatch(const double* input, uint B);

	std::vector<double>& batchOutputs();
	std::vector<double>& batchLocalGradients();

	void dropoutRate(double rate);
	double dropoutRate() const;
	void training(bool t);
	bool isTraining() const;

	void normType(NormType nt);
	NormType normType() const;

	std::vector<double>& gamma();
	std::vector<double>& beta();
	std::vector<double>& gammaGradients();
	std::vector<double>& betaGradients();
	std::vector<double>& gammaUpdates();
	std::vector<double>& betaUpdates();

	uint nNormParams() const;

	void applyNormBackwardBatch(uint B);

	std::vector<double> calcLifs(const std::vector<double>& input);

  protected:
	uint index(const uint i, const uint j) const;

	uint ncurr;
	uint nprev;

	/**Type tag (logsig / tansig / ...). Held alongside theAct because
	 * serialisation writes the string and several callers query it. */
	std::string theType;

	/**Activation variant. Replaces the old per-subclass virtual dispatch. */
	Activation theAct;

	std::vector<double> theWeights;
	std::vector<double> theOutputs;
	std::vector<double> theLocalGradients;
	std::vector<double> theGradients;
	std::vector<double> theWeightUpdates;

	double theDropoutRate;
	bool theTraining;
	std::vector<double> theDropoutMask;
	std::vector<double> theBatchDropoutMask;

	std::vector<double> theBatchOutputs;
	std::vector<double> theBatchLocalGradients;

	mutable std::vector<double> theBiasBuf;

	// --- Normalization state ---
	NormType theNormType;
	std::vector<double> theGamma;
	std::vector<double> theBeta;
	std::vector<double> theGammaGrad;
	std::vector<double> theBetaGrad;
	std::vector<double> theGammaUpdate;
	std::vector<double> theBetaUpdate;
	std::vector<double> theRunningMean;
	std::vector<double> theRunningVar;
	double theBNMomentum;
	std::vector<double> theBatchZHat;
	std::vector<double> theBatchNormMean;
	std::vector<double> theBatchNormVar;
	static constexpr double NORM_EPS = 1e-5;

	InitScheme theInitScheme = InitScheme::Glorot;
};

// ACCESSOR AND MUTATOR FUNCTIONS

inline std::vector<double>& Layer::weights() {
	return theWeights;
}

inline const std::vector<double>& Layer::weights() const {
	return theWeights;
}

inline const std::vector<double>& Layer::outputs() const {
	return theOutputs;
}

inline const std::vector<double>& Layer::gradients() const {
	return theGradients;
}

inline void Layer::weights(const std::vector<double>& w) {
	theWeights = w;
}

inline std::vector<double>& Layer::outputs() {
	return theOutputs;
}

inline std::vector<double>& Layer::localGradients() {
	return theLocalGradients;
}

inline std::vector<double>& Layer::gradients() {
	return theGradients;
}

inline std::vector<double>& Layer::weightUpdates() {
	return theWeightUpdates;
}

inline double& Layer::weights(const uint i, const uint j) {
	return theWeights[index(i, j)];
}

inline double& Layer::weights(const uint i) {
	assert(i < theWeights.size());
	return theWeights[i];
}

inline double& Layer::outputs(const uint i) {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

inline double& Layer::localGradients(const uint i) {
	assert(i < theLocalGradients.size());
	return theLocalGradients[i];
}

inline double& Layer::gradients(const uint i, const uint j) {
	return theGradients[index(i, j)];
}

inline double& Layer::gradients(const uint i) {
	assert(i < theGradients.size());
	return theGradients[i];
}

inline double& Layer::weightUpdates(const uint i, const uint j) {
	return theWeightUpdates[index(i, j)];
}

inline double& Layer::weightUpdates(const uint i) {
	assert(i < theWeightUpdates.size());
	return theWeightUpdates[i];
}

// COUNTS AND CRAP

inline uint Layer::nWeights() const {
	return theWeights.size();
}

inline uint Layer::nNeurons() const {
	return ncurr;
}

inline uint Layer::nPrevious() const {
	return nprev;
}

inline uint Layer::size() const {
	return nNeurons();
}

inline std::vector<double>& Layer::batchOutputs() {
	return theBatchOutputs;
}
inline std::vector<double>& Layer::batchLocalGradients() {
	return theBatchLocalGradients;
}
inline void Layer::normType(NormType nt) {
	theNormType = nt;
}
inline NormType Layer::normType() const {
	return theNormType;
}
inline std::vector<double>& Layer::gamma() {
	return theGamma;
}
inline std::vector<double>& Layer::beta() {
	return theBeta;
}
inline std::vector<double>& Layer::gammaGradients() {
	return theGammaGrad;
}
inline std::vector<double>& Layer::betaGradients() {
	return theBetaGrad;
}
inline std::vector<double>& Layer::gammaUpdates() {
	return theGammaUpdate;
}
inline std::vector<double>& Layer::betaUpdates() {
	return theBetaUpdate;
}
inline uint Layer::nNormParams() const {
	return theNormType != NormType::None ? 2 * ncurr : 0;
}
inline void Layer::dropoutRate(double rate) {
	theDropoutRate = rate;
}
inline double Layer::dropoutRate() const {
	return theDropoutRate;
}
inline void Layer::training(bool t) {
	theTraining = t;
}
inline bool Layer::isTraining() const {
	return theTraining;
}

inline double Layer::fire(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

// PRIVATE--------------------------------------------------------------------//

inline uint Layer::index(const uint i, const uint j) const {
	assert(i < ncurr && j < nprev + 1);
	return i * (nprev + 1) + j;
}

} // namespace MultiLayerPerceptron

#endif
