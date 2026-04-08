#ifndef __ReLULayer_hh__
#define __ReLULayer_hh__

#include "Layer.hh"
#include <cassert>

namespace MultiLayerPerceptron {
class ReLULayer : public Layer {
  public:
	ReLULayer(uint nc, uint np);
	virtual ~ReLULayer();
	std::unique_ptr<Layer> clone() const override { return std::make_unique<ReLULayer>(*this); }

	double fire(const double lif) const;
	double fire(const uint i) const;
	double firePrime(const double lif) const;
	double firePrime(const uint i) const;
	double firePrimePrime(const double lif) const;
	double firePrimePrime(const uint i) const;
};

inline double ReLULayer::fire(const double lif) const {
	return lif > 0.0 ? lif : 0.0;
}

inline double ReLULayer::fire(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

inline double ReLULayer::firePrime(const double lif) const {
	return lif > 0.0 ? 1.0 : 0.0;
}

inline double ReLULayer::firePrime(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i] > 0.0 ? 1.0 : 0.0;
}

inline double ReLULayer::firePrimePrime(const double) const {
	return 0.0;
}

inline double ReLULayer::firePrimePrime(const uint i) const {
	assert(i < theOutputs.size());
	return 0.0;
}
} // namespace MultiLayerPerceptron
#endif
