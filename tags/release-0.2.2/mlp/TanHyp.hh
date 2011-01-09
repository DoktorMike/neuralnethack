#ifndef __TanHyp_hh__
#define __TanHyp_hh__

#include "Neuron.hh"

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Neuron interface. 
	 * This class implements the tanhyp activation function.
	 * \f[\varphi(v)=\tanh(v)\f]
	 */
	class TanHyp: public Neuron
	{
		public:
			/**Basic constructor.
			 * \param nprev the number of neurons in the previous layer.
			 */
			TanHyp(uint nprev);

			/**A copy constructor. 
			 * \param n the object to copy from.
			 */
			TanHyp(const TanHyp& n);

			/**The destructor. */
			virtual ~TanHyp();

			/**Assignment operator. 
			 * \param n the object to assign from.
			 */
			TanHyp& operator=(const TanHyp& n);

			/**The activation function.
			 * \param input the vector holding the input to the neuron.
			 */
			double fire(vector<double>& input);

			/**A derivative of the activation function.
			 * \param input the vector holding the input to the neuron.
			 */
			double firePrime(vector<double>& input);

			/**A derivative of the activation function. */
			double firePrime();
	};
}
#endif
