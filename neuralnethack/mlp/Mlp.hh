/*$Id: Mlp.hh 1684 2007-10-12 15:55:07Z michael $*/

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


#ifndef __Mlp_hh__
#define __Mlp_hh__

#include "Layer.hh"

namespace MultiLayerPerceptron
{
	/**A struct representing the model for a multilayer perceptron. */
	struct MlpModel
	{
		std::vector<uint> architecture;
		std::vector<std::string> types;
		bool softmax;
	};
	
	/**A Class representing a multilayer perceptron. 
	 * \todo Implement softmax output.
	 */
	class Mlp
	{
		public:
			/**Base constructor.
			 * \param a the architecture of this MLP.
			 * \param t the types of neurons that are to be used.
			 * \param s the softmax output switch.
			 */
			Mlp(const std::vector<uint>& a, const std::vector<std::string>& t, bool s);

			/**Base constructor.
			 * \param mlpmodel the model to use for this Mlp.
			 */
			Mlp(const MlpModel& mlpmodel);

			/** Copy constructor.
			 * \param mlp the mlp to copy from.
			 */
			Mlp(const Mlp& mlp);

			/**The default destructor.
			 */
			~Mlp();

			/**Assignment operator.
			 * \param mlp the Mlp to assign from.
			 * \return the Mlp assigned to.
			 */
			Mlp& operator=(const Mlp& mlp);

			/**Index operator.
			 * \param index the index to return.
			 * \return the layer at position index.
			 */
			Layer& operator[](const uint index);

//ACCESSORS AND MUTATORS
			/**Return the architechture of this MLP.
			 * \return the architecture.
			 */
			std::vector<uint>& arch();

			/**Return the types of activation functions used in this Mlp.
			 * \return the types.
			 */
			std::vector<std::string>& types();

			/**Return the softmax switch for this Mlp.
			 * \return true if softmax is on, false otherwise.
			 */
			bool softmax();

			/**Return the layervector.
			 * \return the vector of layers.
			 */
			std::vector<Layer*>& layers();

			/**Returns the indexed layer.
			 * \param index the index to return.
			 * \return the layer at position index.
			 */
			Layer& layer(uint index);

			/**Returns the entire weight vector.
			 * \return the weight vector.
			 */
			std::vector<double> weights() const;

			/**Set the weightvector to the weights in w.
			 * \param w the weights to set the weightvector to.
			 */
			void weights(std::vector<double>& w);

			/**Returns the entire gradient vector.
			 * \return the gradient vector.
			 */
			std::vector<double> gradients() const;

			/**Set the gradient vector to the gradients in g.
			 * \param g the weights to set the gradient vector to.
			 */
			void gradients(std::vector<double>& g);

//UTILITY
			/**Randomize the weights between -1/2 and 1/2.
			 */
			void regenerateWeights();

			/**Pushes a pattern through this MLP.
			 * \param pattern the pattern to propagate.
			 * \return the output vector for this pattern.
			 */
			const std::vector<double>& propagate(const std::vector<double>& pattern);

//SIZE etc
			/**Return the number of layers contained in this MLP.
			 * \return the number of layers.
			 */
			uint nLayers() const;

			/**Return the number of weights contained in this MLP.
			 * \return the number of weights.
			 */
			uint nWeights() const;

			/**Alias for nLayers.
			 * \return the number of layers.
			 */
			uint size() const;
			
//PRINTS etc
			/**Print all the weights in the network.
			 * \param os the output stream to write to.
			 */
			void printWeights(std::ostream& os) const;

			/**Print all the gradients in the network.
			 * \param os the output stream to write to.
			 */
			void printGradients(std::ostream& os) const;

		private:
			/**Create all the layers in this MLP.
			 */
			void createLayers();

			/**The architecture for this MLP. */
			std::vector<uint> theArch;

			/**The type of neurons for each layer. */
			std::vector<std::string> theTypes;

			/**Softmax switch. */
			bool theSoftmax;

			/**The layers in this MLP. */
			std::vector<Layer*> theLayers;
	};

//INLINES

	inline std::vector<uint>& Mlp::arch()
	{return theArch;}

	inline std::vector<std::string>& Mlp::types()
	{return theTypes;}

	inline bool Mlp::softmax()
	{return theSoftmax;}

	inline std::vector<Layer*>& Mlp::layers()
	{return theLayers;}
}
#endif
