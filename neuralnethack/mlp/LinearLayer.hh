#ifndef __LinearLayer_hh__
#define __LinearLayer_hh__

#include "Layer.hh"

#include <string>
#include <cassert>

namespace MultiLayerPerceptron {
/**A class representing a linear implementation of the layer interface.
 * It knows the number of neurons contained in itself and its
 * predecessor.
 * \f[\varphi (v)=v\f]
 * \sa Layer, Mlp.
 */
class LinearLayer : public Layer {
  public:
	/**Constructor.
	 * \param nc the number of neurons in this layer.
	 * \param np the number of neurons in the previous layer.
	 */
	LinearLayer(uint nc, uint np);

	/**Destructor.
	 */
	virtual ~LinearLayer();

	std::unique_ptr<Layer> clone() const override { return std::make_unique<LinearLayer>(*this); }

	// ACCESSOR AND MUTATOR FUNCTIONS

	// ACCESSOR FUNCTIONS

	// COUNTS AND CRAP

	// PRINTS

	// UTILS

	double fire(const double lif) const;

	double fire(const uint i) const;

	double firePrime(const double lif) const;

	double firePrime(const uint i) const;

	double firePrimePrime(const double lif) const;

	double firePrimePrime(const uint i) const;
};

inline double LinearLayer::fire(const double lif) const {
	return lif;
}

inline double LinearLayer::fire(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

inline double LinearLayer::firePrime(const double lif) const {
	return 1.0;
}

inline double LinearLayer::firePrime(const uint i) const {
	assert(i < theOutputs.size());
	return 1.0;
}

inline double LinearLayer::firePrimePrime(const double lif) const {
	return 0.0;
}

inline double LinearLayer::firePrimePrime(const uint i) const {
	assert(i < theOutputs.size());
	return 0.0;
}
} // namespace MultiLayerPerceptron
#endif
