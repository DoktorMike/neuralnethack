/*$Id: Layer.hh 1622 2007-05-08 08:29:10Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#ifndef __Layer_hh__
#define __Layer_hh__

#include "MultiLayerPerceptron.hh"

#include <string>
#include <vector>
#include <cassert>

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
			Layer(const uint nc, const uint np, std::string t);

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
			std::vector<double>& weights();

			/**Set the weights to those in w.
			 * \param w the weights to assign from.
			 */
			void weights(const std::vector<double>& w);

			/**Returns the output from this layer. 
			 * Not including the bias!
			 * \return the outputs from this layer.
			 */
			std::vector<double>& outputs();

			/**Returns the local gradients in this layer. 
			 * \return the local gradients in this layer.
			 */
			std::vector<double>& localGradients();

			/**Return the gradients leading into this layer.
			 * \return the gradients leading into this layer.
			 */
			std::vector<double>& gradients();

			/**Return the weight updates leading into this layer.
			 * \return the wight updates leading into this layer.
			 */
			std::vector<double>& weightUpdates();

			//ACCESSOR FUNCTIONS

			/**Get the specified weight.
			 * \param i the node in this layer that the weight is connected to.
			 * \param j the node in the previous layer that the weight is connected to.
			 * \return the weight.
			 */
			double& weights(const uint i, const uint j);
			
			/**Get the specified weight.
			 * \param i the index of the weight in the weight vector.
			 * \return the weight.
			 */
			double& weights(const uint i);
			
			/**Get the specified output.
			 * \param i the node which output to return.
			 * \return the output.
			 */
			double& outputs(const uint i);
			
			/**Get the specified local gradient.
			 * \param i the node which local gradient to return.
			 * \return the local gradient.
			 */
			double& localGradients(const uint i);
			
			/**Get the specified gradient.
			 * \param i the node in this layer that the gradient is connected to.
			 * \param j the node in the previous layer that the gradient is connected to.
			 * \return the gradient.
			 */
			double& gradients(const uint i, const uint j);
			
			/**Get the specified gradient.
			 * \param i the index of the gradient in the gradient vector.
			 * \return the gradient.
			 */
			double& gradients(const uint i);
			
			/**Get the specified weight update.
			 * \param i the node in this layer that the weight update is connected to.
			 * \param j the node in the previous layer that the weight update is connected to.
			 * \return the weight update.
			 */
			double& weightUpdates(const uint i, const uint j);
			
			/**Get the specified weight update.
			 * \param i the index of the weight update in the weight update vector.
			 * \return the weight update.
			 */
			double& weightUpdates(const uint i);

			//COUNTS AND CRAP

			/**Get the number of weights contained in this layer.
			 * \return the number of weights contained in this layer.
			 */
			uint nWeights() const;

			/**Get the number of neurons contained in this layer.
			 * Not including the bias.
			 * \return the number of neurons contained in this layer.
			 */
			uint nNeurons() const;

			/**Get the number of neurons contained in the previous layer.
			 * Not including the bias.
			 * \return the number of neurons contained in the previous layer.
			 */
			uint nPrevious() const;

			/**Get the number of neurons contained in this layer.
			 * Not including the bias.
			 * \return the number of neurons contained in this layer.
			 */
			uint size() const;

			//PRINTS

			/**Prints the weights leading into this layer.
			 */
			void printWeights(std::ostream& os) const;

			/**Prints the local gradients for the neurons in this layer.
			 */
			void printLocalGradients(std::ostream& os) const;

			/**Prints the gradients for the neurons in this layer.
			 */
			void printGradients(std::ostream& os) const;

			//UTILS

			/**Assign new random weights to the weight vector. */
			void regenerateWeights();

			/**Activation function for every neuron in this layer.
			 * \param lif the local induced field for a neuron.
			 * \return the activation for the neuron.
			 */
			virtual double fire(double lif) const = 0;

			/**Get the activation value for the specified neuron.
			 * \param i the neuron in this layer which activation to return.
			 * \return the activation for the neuron.
			 */
			virtual double fire(const uint i) const = 0;

			/**The derivative of the activation function.
			 * \param lif the local induced field for a neuron.
			 * \return the derivative of the activation for the neuron.
			 */
			virtual double firePrime(double lif) const = 0;

			/**Get the derivative of the activation value for the specified neuron.
			 * \param i the neuron in this layer which activation to return.
			 * \return the derivative of the activation for the neuron.
			 */
			virtual double firePrime(const uint i) const = 0;

			/**Propagates an input pattern through this layer. 
			 * Note that the bias should not be included in the parameter 
			 * since it is explicitly included later.
			 * \param input the input to propagate.
			 * \return the outputs from this Layer.
			 */
			std::vector<double>& propagate(const std::vector<double>& input);

			/**Calculates the Local Induced Field for each node in this layer. 
			 * Note that the bias should not be included in the parameter 
			 * since it is explicitly included later.
			 * \param input the input to this Layer.
			 * \return the local induced fields.
			 */
			std::vector<double> calcLifs(const std::vector<double>& input);

		protected:
			/**Convert a two index value to a one index value.
			 * \param i the row.
			 * \param j the column.
			 * \return the resulting index.
			 */
			uint index(const uint i, const uint j) const;

			/**Number of neurons in this layer. */
			uint ncurr;

			/**Number of neurons in previous layer. */
			uint nprev;

			/**Type of neurons in this layer. */
			std::string theType;

			/**The weights leading into this layer. */
			std::vector<double> theWeights;
			
			/**The output of this layer. */
			std::vector<double> theOutputs;

			/**The local gradients of this layers neurons. */
			std::vector<double> theLocalGradients;

			/**The weight gradients of this layers weights. */
			std::vector<double> theGradients;

			/**The update used for this layers weights. */
			std::vector<double> theWeightUpdates;

			/**The functor for initialising the weights. */
			template <class T> struct newRand 
			{ 
				void operator()(T& a){ a = 0.5-drand48(); }
			};
	};

	//ACCESSOR AND MUTATOR FUNCTIONS

	inline std::vector<double>& Layer::weights() {return theWeights;}

	inline void Layer::weights(const std::vector<double>& w) {theWeights = w;}

	inline std::vector<double>& Layer::outputs() {return theOutputs;}

	inline std::vector<double>& Layer::localGradients() {return theLocalGradients;}

	inline std::vector<double>& Layer::gradients() {return theGradients;}

	inline std::vector<double>& Layer::weightUpdates() {return theWeightUpdates;}

	inline double& Layer::weights(const uint i, const uint j) {return theWeights[index(i,j)];}

	inline double& Layer::weights(const uint i)
	{
		assert(i < theWeights.size());
		return theWeights[i];
	}

	inline double& Layer::outputs(const uint i)
	{
		assert(i < theOutputs.size());
		return theOutputs[i];
	}

	inline double& Layer::localGradients(const uint i)
	{
		assert(i < theLocalGradients.size());
		return theLocalGradients[i];
	}

	inline double& Layer::gradients(const uint i, const uint j) {return theGradients[index(i,j)];}

	inline double& Layer::gradients(const uint i)
	{
		assert(i < theGradients.size());
		return theGradients[i];
	}

	inline double& Layer::weightUpdates(const uint i, const uint j)
	{return theWeightUpdates[index(i,j)];}

	inline double& Layer::weightUpdates(const uint i)
	{
		assert(i < theWeightUpdates.size());
		return theWeightUpdates[i];
	}


	//COUNTS AND CRAP

	inline uint Layer::nWeights() const	{return theWeights.size();}

	inline uint Layer::nNeurons() const {return ncurr;}

	inline uint Layer::nPrevious() const {return nprev;}

	inline uint Layer::size() const {return nNeurons();}

	//PRIVATE--------------------------------------------------------------------//

	inline uint Layer::index(const uint i, const uint j) const
	{
		assert(i < ncurr && j < nprev+1);
		return i*(nprev+1)+j;
	}

}

#endif
