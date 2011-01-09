#ifndef __SigmoidLayer_hh__
#define __SigmoidLayer_hh__

#include "Layer.hh"

#include <string>

namespace MultiLayerPerceptron
{
	/**A class representing a sigmoid implementation of the layer interface.
	 * It knows the number of neurons contained in itself and its
	 * predecessor.
	 * \f[\varphi (v)=\frac{1}{1+\exp^{-v}}\f]
	 * \sa Layer, Mlp.
	 */
	class SigmoidLayer: public Layer
	{
		public:
			/**Constructor.
			 * \param nc the number of neurons in this layer.
			 * \param np the number of neurons in the previous layer.
			 */
			SigmoidLayer(uint nc, uint np);

			/**Destructor.
			 */
			virtual ~SigmoidLayer();

			//ACCESSOR AND MUTATOR FUNCTIONS

			//ACCESSOR FUNCTIONS

			//COUNTS AND CRAP

			//PRINTS

			//UTILS

			double fire(double lif);

			double fire(uint i);

			double firePrime(double lif);

			double firePrime(uint i);
	};
}
#endif
