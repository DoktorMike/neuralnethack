#ifndef __LeakyReLULayer_hh__
#define __LeakyReLULayer_hh__

#include "Layer.hh"
#include <cassert>

namespace MultiLayerPerceptron
{
	class LeakyReLULayer: public Layer
	{
		public:
			LeakyReLULayer(uint nc, uint np);
			virtual ~LeakyReLULayer();
			std::unique_ptr<Layer> clone() const override
			{ return std::make_unique<LeakyReLULayer>(*this); }

			double fire(const double lif) const;
			double fire(const uint i) const;
			double firePrime(const double lif) const;
			double firePrime(const uint i) const;
			double firePrimePrime(const double lif) const;
			double firePrimePrime(const uint i) const;

		private:
			static constexpr double ALPHA = 0.01;
	};

	inline double LeakyReLULayer::fire(const double lif) const
	{ return lif > 0.0 ? lif : ALPHA * lif; }

	inline double LeakyReLULayer::fire(const uint i) const
	{ assert(i < theOutputs.size()); return theOutputs[i]; }

	inline double LeakyReLULayer::firePrime(const double lif) const
	{ return lif > 0.0 ? 1.0 : ALPHA; }

	inline double LeakyReLULayer::firePrime(const uint i) const
	{ assert(i < theOutputs.size()); return theOutputs[i] > 0.0 ? 1.0 : ALPHA; }

	inline double LeakyReLULayer::firePrimePrime(const double) const
	{ return 0.0; }

	inline double LeakyReLULayer::firePrimePrime(const uint i) const
	{ assert(i < theOutputs.size()); return 0.0; }
}
#endif
