#ifndef __Mlp_hh__
#define __Mlp_hh__

#include "Layer.hh"

namespace MultiLayerPerceptron
{
	/**A Class representing a multilayer perceptron. 
	 * \todo Implement softmax output.
	 */
	class Mlp
	{
		public:
			///Base constructor.
			///\param a the architecture of this MLP.
			///\param t the types of neurons that are to be used.
			///\param s the softmax output switch.
			Mlp(vector<uint>& a, vector<string>& t, bool s);

			///Copy constructor.
			Mlp(const Mlp&);

			///The default destructor.
			~Mlp();

			///Assignment operator.
			Mlp& operator=(const Mlp&);

			///Index operator.
			Layer& operator[](const uint);

//ACCESSORS AND MUTATORS
			///Return the architechture of this MLP.
			vector<uint>& arch();

			vector<string>& types();

			///Return the layervector.
			vector<Layer*>& layers();

			///Returns the indexed layer.
			Layer& layer(uint index);

			///Returns the entire weight vector.
			vector<double> weights();

			///Set the weightvector to the weights in w.
			///\param w the weights to set the weightvector to.
			void weights(vector<double>& w);

			///Returns the entire gradient vector.
			vector<double> gradients();

//UTILITY
			void regenerateWeights();

			///Pushes a pattern through this MLP.
			vector<double>& propagate(vector<double>&);

//SIZE etc
			///Return the number of layers contained in this MLP.
			uint nLayers();

			///Return the number of weights contained in this MLP.
			uint nWeights();

			///Alias for nLayers.
			uint size();
			
//PRINTS etc
			///Print all the weights in the network.
			void printWeights();

			///Print all the gradients in the network.
			void printGradients();

		private:
			///Create all the layers in this MLP.
			void createLayers();

			///The architecture for this MLP.
			vector<uint> theArch;

			///The type of neurons for each layer.
			vector<string> theTypes;

			///Softmax switch.
			bool softmax;

			///The layers in this MLP.
			vector<Layer*> theLayers;
	};
}
#endif
