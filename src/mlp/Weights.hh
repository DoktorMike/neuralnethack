#ifndef __Weights_hh__
#define __Weights_hh__

#include "MultiLayerPerceptron.hh"
#include "NeuralNetHack.hh"

#include <vector>
#include <cstdlib>

namespace MultiLayerPerceptron
{
	/**A class holding all the weights in the network. 
	 * The input layer is not regarded as a layer in this class since there 
	 * are no weights going _into_ that layer. 
	 * Thus index 0 refers to the first hidden layer.
	 */
	class Weights
	{

		public:
			/**The basic constructor.
			 * \param a the architechture.
			 */
			Weights(std::vector<uint>& a);

			/**The copy constructor.
			 * \param w the Weights to copy from.
			 */
			Weights(const Weights& w);

			/**The destructor. */
			virtual ~Weights();

			/**The assignment operator.
			 * \param w the Weights to assign from.
			 */
			Weights& operator=(const Weights& w);

			/**Prints all the weights to stdout. */
			void print();

			/**Kill the specified weight.
			 * \param n the index of the weight to kill.
			 */
			void kill(uint n);

			/**Kill the specified weight.
			 * \param layer the layer to which the weight connects.
			 * \param ncurr the neuron in the layer that the the weight is connected to.
			 * \param nprev the neuron in the previous layer that the weight connects to.
			 */
			void kill(uint layer, uint ncurr, uint nprev);

			/**Update the specified weight.
			 * \param n the index of the weight to kill.
			 * \param w the new value for the weight.
			 */
			void update(uint n, double w);

			/**Update the specified weight.
			 * \param layer the layer to which the weight connects.
			 * \param ncurr the neuron in the layer that the the weight is connected to.
			 * \param nprev the neuron in the previous layer that the weight connects to.
			 * \param w the new value for the weight.
			 */
			void update(uint layer, uint ncurr, uint nprev, double w);

			/**Return the entire weight vector.
			 * \return the weight vector.
			 */
			std::vector<double>& weights();

			/**Return the interval representing the weights leading into a layer.
			 * \param layer the layer whos weights to get.
			 * \param first the first weight leading into this layer.
			 * \param last the last weight leading into this layer.
			 */
			void weights(uint layer, std::vector<double>::iterator& first, std::vector<double>::iterator& last);

			/**Return the interval representing the weights leading into a neuron.
			 * \param layer the layer we want to access.
			 * \param neuron the neuron in the layer whos weights to get.
			 * \param first the first weight leading into this neuron.
			 * \param last the last weight leading into this neuron.
			 */
			void weights(uint layer, uint neuron, 
					std::vector<double>::iterator& first,
					std::vector<double>::iterator& last);

			/**Return the interval representing the weight between two neurons.
			 * \param layer the layer to which the weight connects.
			 * \param ncurr the neuron in the layer that the the weight is connected to.
			 * \param nprev the neuron in the previous layer that the weight connects to.
			 * \param first the first weight leading into this neuron.
			 * \param last the last weight leading into this neuron.
			 */
			void weights(uint layer, uint ncurr, uint nprev, 
					std::vector<double>::iterator& first, 
					std::vector<double>::iterator& last);

			/**Return the size of the weightVector.
			 * \return the number of weights residing in the weight vector.
			 */
			uint size();

		private:
			/**Return the index representing the weight between two neurons.
			 * The first argument is the layer, the second is the neuron
			 * residing in that layer and the third is the neuron in the
			 * previous layer.
			 * \param layer the layer to which the weight connects.
			 * \param ncurr the neuron in the layer that the the weight is connected to.
			 * \param nprev the neuron in the previous layer that the weight connects to.
			 * \return the index representing this weight in the weight vector.
			 */
			uint index(uint layer, uint ncurr, uint nprev);

			/**Return the start index for the weights leading into a neuron in a layer.
			 * \param layer the layer to which the weight connects.
			 * \param neuron the neuron in the layer that the the weight is connected to.
			 * \return the index representing this weight in the weight vector.
			 */
			uint index(uint layer, uint neuron);

			/**Return the start index for the weights leading into the
			 * first neuron in a layer.
			 * \param layer the layer to which the weight connects.
			 * \return the index representing this weight in the weight vector.
			 */
			uint index(uint layer);

			/**Return the interval representing the weight between two neurons.
			 * \param layer the layer to which the weight connects.
			 * \param ncurr the neuron in the layer that the the weight is connected to.
			 * \param nprev the neuron in the previous layer that the weight connects to.
			 * \param first the first weight leading into this neuron.
			 * \param last the last weight leading into this neuron.
			 */
			void itor(uint layer, uint ncurr, uint nprev, 
					std::vector<double>::iterator& first, 
					std::vector<double>::iterator& last);

			/**Return the interval representing the weights leading into a neuron.
			 * \param layer the layer to which the weight connects.
			 * \param neuron the neuron in the layer that the the weight is connected to.
			 * \param first the first weight leading into this neuron.
			 * \param last the last weight leading into this neuron.
			 */
			void itor(uint layer, uint neuron, 
					std::vector<double>::iterator& first, 
					std::vector<double>::iterator& last);

			/**Return the interval representing the weights leading into a layer.
			 * \param layer the layer to which the weight connects.
			 * \param first the first weight leading into this neuron.
			 * \param last the last weight leading into this neuron.
			 */
			void itor(uint layer, std::vector<double>::iterator& first, 
					std::vector<double>::iterator& last);

			/**The weight vector. */
			std::vector<double>* theWeights;

			/**The architechture. */
			std::vector<uint> arch;

	};
}
#endif
