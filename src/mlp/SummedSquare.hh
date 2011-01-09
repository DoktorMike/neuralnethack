#ifndef __SummedSquare_hh__
#define __SummedSquare_hh__

#include "Error.hh"

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Error interface. 
	 * Only full batch updating is supported. This class represents 
	 * the Summed Square Error function.
	 * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
	 */
	class SummedSquare:public Error
	{
		public:
			/**Basic constructor. This constructor sets the mlp and the
			 * dataset pointers to 0.
			 */
			SummedSquare();

			/**Basic constructor.
			 * \param mlp the mlp to use.
			 * \param dset the dataset to use.
			 */
			SummedSquare(MultiLayerPerceptron::Mlp* mlp, DataTools::DataSet* dset);

			/**Basic destructor. */
			~SummedSquare();

			/**Return the gradient vector. This returns the gradient vector
			 * calculated from a block of patterns.
			 * \param mlp the MLP to calculate the gradient for.
			 * \param dset the data set to use.
			 * \return the gradient.
			 */
			double gradient(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& dset);

			/**Calculate the gradient vector, and return the error.
			 * This returns the error calculated from a block of patterns.
			 * \return the error.
			 */
			double gradient();

			/**Calculate the summed square error for this Mlp and DataSet.
			 * \param mlp the output from the MLP.
			 * \param dset the desired output.
			 * \return the error.
			 */
			double outputError(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& dset);

			/**Calculate the summed square error for this Mlp and DataSet.
			 * \return the error.
			 */
			double outputError();

		private:
			/**Copy constructor.
			 * \param sse the error to create this object from.
			 */
			SummedSquare(const SummedSquare& sse);

			/**Assignment operator.
			 * \param sse the error to copy from.
			 */
			SummedSquare& operator=(const SummedSquare& sse);

			/**Calculate the output error for every neuron in the output
			 * layer.
			 * \param ol the output layer.
			 * \param out the output from the MLP.
			 * \param dout the desired output.
			 */
			void localGradient(MultiLayerPerceptron::Layer& ol, std::vector<double>& out, 
					std::vector<double>& dout);

			/**Calculate the local gradient.
			 * \param curr the current layer.
			 * \param next the next layer.
			 */
			void localGradient(MultiLayerPerceptron::Layer& curr, MultiLayerPerceptron::Layer& next);

			/**Back-propagate the output error through the network.
			 * \deprecated Use the backpropagate() instead, since an Mlp is now a
			 * member of the Error interface.
			 * \param mlp the MLP to backpropagate the error through.
			 */
			void backpropagate(MultiLayerPerceptron::Mlp& mlp);

			/**Back-propagate the output error through the network. */
			void backpropagate();

			/**Calculate gradient for the weights in the first layer.
			 * \deprecated Use the gradientBatch instead since Error functions in
			 * NeuralNetHack should only do batch updating. It is up to the Trainer to
			 * divide up DataSet:s as to provide block or online updating.
			 * \param first the first hidden layer with weights.
			 * \param in the input vector.
			 */
			void gradient(MultiLayerPerceptron::Layer& first, std::vector<double>& in);

			/**Calculate gradient for the weights in the first layer. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * block version of gradient.
			 * \param first the first hidden layer with weights.
			 * \param in the input vector.
			 */
			void gradientBatch(MultiLayerPerceptron::Layer& first, std::vector<double>& in);

			/**Calculate the gradient between two layers.
			 * \deprecated Use the gradientBatch instead since Error functions in
			 * NeuralNetHack should only do batch updating. It is up to the Trainer to
			 * divide up DataSet:s as to provide block or online updating.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradient(MultiLayerPerceptron::Layer& curr, MultiLayerPerceptron::Layer& prev); 

			/**Calculate the gradient between two layers. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * batch version of gradient.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradientBatch(MultiLayerPerceptron::Layer& curr, MultiLayerPerceptron::Layer& prev); 

			/**Calculate the summed square error for this output.
			 * \param out the output from the MLP.
			 * \param dout the desired output.
			 * \return the error.
			 */
			double outputError(std::vector<double>& out, std::vector<double>& dout);

			/**Set all gradients to zero. */
			void killGradients();

	};
}
#endif
