#ifndef __TanHypLayer_hh__
#define __TanHypLayer_hh__

#include "Layer.hh"

#include <string>

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
