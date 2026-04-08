#ifndef __TanHypLayer_hh__
#define __TanHypLayer_hh__

#include "Layer.hh"

#include <string>
#include <cmath>

namespace MultiLayerPerceptron
{
	/**A class representing a tanhyp implementation of the layer interface.
	 * It knows the number of neurons contained in itself and its
	 * predecessor.
	 * \f[\varphi (v)=\tanh{v}\f]
	 * \sa Layer, Mlp.
	 */
	class TanHypLayer: public Layer
	{
		public:
			/**Constructor.
			 * \param nc the number of neurons in this layer.
			 * \param np the number of neurons in the previous layer.
			 */
			TanHypLayer(uint nc, uint np);

			/**Destructor.
			 */
			virtual ~TanHypLayer();

			std::unique_ptr<Layer> clone() const override
			{ return std::make_unique<TanHypLayer>(*this); }

			//ACCESSOR AND MUTATOR FUNCTIONS

			//ACCESSOR FUNCTIONS

			//COUNTS AND CRAP

			//PRINTS

			//UTILS

			double fire(const double lif) const;
			double fire(const uint i) const;
			double firePrime(const double lif) const;
			double firePrime(const uint i) const;
			double firePrimePrime(const double lif) const;
			double firePrimePrime(const uint i) const;
	};

	inline double TanHypLayer::fire(const double lif) const { return tanh(lif); }

	inline double TanHypLayer::fire(const uint i) const
	{
		assert(i<theOutputs.size());
		return theOutputs[i];
	}

	inline double TanHypLayer::firePrime(const double lif) const
	{
		double tmp = fire(lif);
		return 1.0 - tmp * tmp;
	}

	inline double TanHypLayer::firePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		return 1.0 - theOutputs[i] * theOutputs[i];
	}

	inline double TanHypLayer::firePrimePrime(const double lif) const
	{
		double f = fire(lif);
		return 2*f*(f*f-1);
	}

	inline double TanHypLayer::firePrimePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		double f = theOutputs[i];
		return 2*f*(f*f-1);
	}
}
#endif
