#ifndef __CrossEntropy_hh__
#define __CrossEntropy_hh__

#include "Error.hh"

namespace NeuralNetHack
{
	/**A class representing the implementation of the Error interface. 
	 * Only full batch updating is supported. This class represents 
	 * the Cross Entropy Error function.
	 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
	 * \todo The summation should be over the number of classes in order to
	 * make this error function work.
	 */
	class CrossEntropy:public Error
	{
		public:
			/**Basic constructor. */
			CrossEntropy();

			/**Basic constructor.
			 * \param mlp the mlp to use.
			 * \param dset the dataset to use.
			 */
			CrossEntropy(Mlp* mlp, DataSet* dset);

			/**Basic destructor. */
			~CrossEntropy();

			/**Return the gradient vector. This returns the gradient vector
			 * calculated from a block of patterns.
			 * \param mlp the MLP to calculate the gradient for.
			 * \param dset the data set to use.
			 * \return the gradient.
			 */
			double gradient(Mlp& mlp, DataSet& dset);

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
			double outputError(Mlp& mlp, DataSet& dset);

			/**Calculate the summed square error for this Mlp and DataSet.
			 * \return the error.
			 */
			double outputError();

		private:
			/**Copy constructor.
			 * \param sse the error to create this object from.
			 */
			CrossEntropy(const CrossEntropy& sse);

			/**Assignment operator.
			 * \param sse the error to copy from.
			 */
			CrossEntropy& operator=(const CrossEntropy& sse);

			/**Calculate the output error for every neuron in the output
			 * layer.
			 * \param ol the output layer.
			 * \param out the output from the MLP.
			 * \param dout the desired output.
			 */
			void localGradient(Layer& ol, vector<double>& out, 
					vector<double>& dout);

			/**Calculate the local gradient.
			 * \param curr the current layer.
			 * \param next the next layer.
			 */
			void localGradient(Layer& curr, Layer& next);

			/**Back-propagate the output error through the network.
			 * \deprecated Use the backpropagate() instead, since an Mlp is now a
			 * member of the Error interface.
			 * \param mlp the MLP to backpropagate the error through.
			 */
			void backpropagate(Mlp& mlp);

			/**Back-propagate the output error through the network. */
			void backpropagate();

			/**Calculate gradient for the weights in the first layer.
			 * \deprecated Use the gradientBatch instead since Error functions in
			 * NeuralNetHack should only do batch updating. It is up to the Trainer to
			 * divide up DataSet:s as to provide block or online updating.
			 * \param first the first hidden layer with weights.
			 * \param in the input vector.
			 */
			void gradient(Layer& first, vector<double>& in);

			/**Calculate gradient for the weights in the first layer. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * block version of gradient.
			 * \param first the first hidden layer with weights.
			 * \param in the input vector.
			 */
			void gradientBatch(Layer& first, vector<double>& in);

			/**Calculate the gradient between two layers.
			 * \deprecated Use the gradientBatch instead since Error functions in
			 * NeuralNetHack should only do batch updating. It is up to the Trainer to
			 * divide up DataSet:s as to provide block or online updating.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradient(Layer& curr, Layer& prev); 

			/**Calculate the gradient between two layers. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * batch version of gradient.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradientBatch(Layer& curr, Layer& prev); 

			/**Calculate the summed square error for this output.
			 * \param out the output from the MLP.
			 * \param dout the desired output.
			 * \return the error.
			 * \bug This generates a NaN at some point when error is very low.
			 * It probably has something to do with the logarithm.
			 */
			double outputError(vector<double>& out, vector<double>& dout);

			/**Set all gradients to zero. */
			void killGradients();

	};
}
#endif
