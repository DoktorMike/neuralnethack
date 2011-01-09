#ifndef __Layer_hh__
#define __Layer_hh__

#include "MultiLayerPerceptron.hh"

#include <string>

namespace MultiLayerPerceptron
{
	/**A class representing a layer in an Mlp.
	 * A Layer has a number of neurons which have an activation function that
	 * is implementation dependant. The Layer knows the number of neurons 
	 * contained in itself and its predecessor.
	 * \sa Mlp, SigmoidLayer, TanHypLayer, LinearLayer.
	 */
	class Layer
	{
		public:
			/**Constructor.
			 * \param nc the number of neurons in this layer.
			 * \param np the number of neurons in the previous layer.
			 * \param t the type of layer this is.
			 */
			Layer(uint nc, uint np, string t);

			/**Copy constructur.
			 * \param layer the Layer to copy from.
			 */
			Layer(const Layer& layer);

			/**Destructor.
			 */
			virtual ~Layer();

			/**Assignment operator.
			 * \param layer the Layer to assign from.
			 * \return the Layer assigned to.
			 */
			Layer& operator=(const Layer& layer);

			/**Index operator.
			 * Fetch the weight located at the specified index.
			 * \param i the index to return.
			 * \return the weight at index i.
			 */
			double& operator[](const uint i);

			//ACCESSOR AND MUTATOR FUNCTIONS
			/**Get the weights leading into this layer.
			 * \return the weight vector.
			 */
			vector<double>& weights();

			/**Set the weights to those in w.
			 * \param w the weights to assign from.
			 */
			void weights(vector<double>& w);

			/**Returns the output from this layer. 
			 * Not including the bias!
			 * \return the outputs from this layer.
			 */
			vector<double>& outputs();

			/**Returns the local gradients in this layer. 
			 * \return the local gradients in this layer.
			 */
			vector<double>& localGradients();

			/**Return the gradients leading into this layer.
			 * \return the gradients leading into this layer.
			 */
			vector<double>& gradients();

			/**Return the weight updates leading into this layer.
			 * \return the wight updates leading into this layer.
			 */
			vector<double>& weightUpdates();

			//ACCESSOR FUNCTIONS

			/**Get the specified weight.
			 * \param i the node in this layer that the weight is connected to.
			 * \param j the node in the previous layer that the weight is connected to.
			 * \return the weight.
			 */
			double& weights(uint i, uint j);
			
			/**Get the specified weight.
			 * \param i the index of the weight in the weight vector.
			 * \return the weight.
			 */
			double& weights(uint i);
			
			/**Get the specified output.
			 * \param i the node which output to return.
			 * \return the output.
			 */
			double& outputs(uint i);
			
			/**Get the specified local gradient.
			 * \param i the node which local gradient to return.
			 * \return the local gradient.
			 */
			double& localGradients(uint i);
			
			/**Get the specified gradient.
			 * \param i the node in this layer that the gradient is connected to.
			 * \param j the node in the previous layer that the gradient is connected to.
			 * \return the gradient.
			 */
			double& gradients(uint i, uint j);
			
			/**Get the specified gradient.
			 * \param i the index of the gradient in the gradient vector.
			 * \return the gradient.
			 */
			double& gradients(uint i);
			
			/**Get the specified weight update.
			 * \param i the node in this layer that the weight update is connected to.
			 * \param j the node in the previous layer that the weight update is connected to.
			 * \return the weight update.
			 */
			double& weightUpdates(uint i, uint j);
			
			/**Get the specified weight update.
			 * \param i the index of the weight update in the weight update vector.
			 * \return the weight update.
			 */
			double& weightUpdates(uint i);

			//COUNTS AND CRAP

			/**Get the number of weights contained in this layer.
			 * \return the number of weights contained in this layer.
			 */
			uint nWeights();

			/**Get the number of neurons contained in this layer.
			 * Not including the bias.
			 * \return the number of neurons contained in this layer.
			 */
			uint nNeurons();

			/**Get the number of neurons contained in this layer.
			 * Not including the bias.
			 * \return the number of neurons contained in this layer.
			 */
			uint size();

			//PRINTS

			/**Prints the weights leading into this layer.
			 */
			void printWeights();

			/**Prints the local gradients for the neurons in this layer.
			 */
			void printLocalGradients();

			/**Prints the gradients for the neurons in this layer.
			 */
			void printGradients();

			//UTILS

			/**Assign new random weights to the weight vector. */
			void regenerateWeights();

			/**Activation function for every neuron in this layer.
			 * \param lif the local induced field for a neuron.
			 * \return the activation for the neuron.
			 */
			virtual double fire(double lif) = 0;

			/**Get the activation value for the specified neuron.
			 * \param i the neuron in this layer which activation to return.
			 * \return the activation for the neuron.
			 */
			virtual double fire(uint i) = 0;

			/**The derivative of the activation function.
			 * \param lif the local induced field for a neuron.
			 * \return the derivative of the activation for the neuron.
			 */
			virtual double firePrime(double lif) = 0;

			/**Get the derivative of the activation value for the specified neuron.
			 * \param i the neuron in this layer which activation to return.
			 * \return the derivative of the activation for the neuron.
			 */
			virtual double firePrime(uint i) = 0;

			/**Propagates an input pattern through this layer. 
			 * Note that the bias should not be included in the parameter 
			 * since it is explicitly included later.
			 * \param input the input to propagate.
			 * \return the outputs from this Layer.
			 */
			vector<double>& propagate(vector<double>& input);

		protected:
			/**Convert a two index value to a one index value.
			 * \param i the row.
			 * \param j the column.
			 * \return the resulting index.
			 */
			uint index(uint i, uint j);

			/**Number of neurons in this layer. */
			uint ncurr;

			/**Number of neurons in previous layer. */
			uint nprev;

			/**Type of neurons in this layer. */
			string theType;

			/**The weights leading into this layer. */
			vector<double> theWeights;
			
			/**The output of this layer. */
			vector<double> theOutputs;

			/**The local gradients of this layers neurons. */
			vector<double> theLocalGradients;

			/**The weight gradients of this layers weights. */
			vector<double> theGradients;

			/**The update used for this layers weights. */
			vector<double> theWeightUpdates;
	};
}
#endif
