#ifndef __Mlp_hh__
#define __Mlp_hh__

#include "Layer.hh"
#include "NeuralNetHack.hh"

namespace MultiLayerPerceptron
{
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
			Mlp(std::vector<uint>& a, std::vector<std::string>& t, bool s);

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
			std::vector<double> weights();

			/**Set the weightvector to the weights in w.
			 * \param w the weights to set the weightvector to.
			 */
			void weights(std::vector<double>& w);

			/**Returns the entire gradient vector.
			 * \return the gradient vector.
			 */
			std::vector<double> gradients();

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
			std::vector<double>& propagate(std::vector<double>& pattern);

//SIZE etc
			/**Return the number of layers contained in this MLP.
			 * \return the number of layers.
			 */
			uint nLayers();

			/**Return the number of weights contained in this MLP.
			 * \return the number of weights.
			 */
			uint nWeights();

			/**Alias for nLayers.
			 * \return the number of layers.
			 */
			uint size();
			
//PRINTS etc
			/**Print all the weights in the network.
			 * \param os the output stream to write to.
			 */
			void printWeights(std::ostream& os);

			/**Print all the gradients in the network.
			 * \param os the output stream to write to.
			 */
			void printGradients(std::ostream& os);

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
