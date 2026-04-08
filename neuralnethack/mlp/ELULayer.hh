#ifndef __ELULayer_hh__
#define __ELULayer_hh__

#include "Layer.hh"
#include <cmath>
#include <cassert>

namespace MultiLayerPerceptron {
class ELULayer : public Layer {
  public:
	ELULayer(uint nc, uint np);
	virtual ~ELULayer();
	std::unique_ptr<Layer> clone() const override { return std::make_unique<ELULayer>(*this); }

	double fire(const double lif) const;
	double fire(const uint i) const;
	double firePrime(const double lif) const;
	double firePrime(const uint i) const;
	double firePrimePrime(const double lif) const;
	double firePrimePrime(const uint i) const;

  private:
	static constexpr double ALPHA = 1.0;
};

inline double ELULayer::fire(const double lif) const {
	return lif > 0.0 ? lif : ALPHA * (exp(lif) - 1.0);
}

inline double ELULayer::fire(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i];
}

inline double ELULayer::firePrime(const double lif) const {
	return lif > 0.0 ? 1.0 : ALPHA * exp(lif);
}

inline double ELULayer::firePrime(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i] >= 0.0 ? 1.0 : theOutputs[i] + ALPHA;
}

inline double ELULayer::firePrimePrime(const double lif) const {
	return lif > 0.0 ? 0.0 : ALPHA * exp(lif);
}

inline double ELULayer::firePrimePrime(const uint i) const {
	assert(i < theOutputs.size());
	return theOutputs[i] >= 0.0 ? 0.0 : theOutputs[i] + ALPHA;
}
} // namespace MultiLayerPerceptron
#endif
