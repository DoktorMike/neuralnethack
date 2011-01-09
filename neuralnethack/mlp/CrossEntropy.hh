/*$Id: CrossEntropy.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __CrossEntropy_hh__
#define __CrossEntropy_hh__

#include "Error.hh"

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Error interface. 
	 * Only full batch updating is supported. This class represents 
	 * the Kullback Leibler(Cross Entropy) Error function. Not that this Error function is
	 * supposed to be used with class discrimination problems ONLY! Thus in an
	 * architechture with only one output I still assume two classes. In the
	 * special case of one output and two classes the following will be used:
	 * \f[E=-\frac{1}{N}\sum_{n}\left(d_n\ln y_n + (1-d_n)\ln (1-y_n)\right)\f]
	 * Otherwise we use:
	 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
	 */
	class CrossEntropy:public Error
	{
		public:
			/**Basic constructor. This constructor sets the mlp and the
			 * dataset pointers to 0.
			 */
			CrossEntropy();

			/**Basic constructor.
			 * \param mlp the mlp to use.
			 * \param dset the dataset to use.
			 */
			CrossEntropy(MultiLayerPerceptron::Mlp* mlp, 
					DataTools::DataSet* dset);

			/**Basic destructor. */
			~CrossEntropy();

			double gradient(MultiLayerPerceptron::Mlp& mlp, 
					DataTools::DataSet& dset);

			double gradient();

			double outputError(MultiLayerPerceptron::Mlp& mlp, 
					DataTools::DataSet& dset);

			double outputError() const;

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
			void localGradient(MultiLayerPerceptron::Layer& ol, 
					std::vector<double>& out, std::vector<double>& dout);

			/**Calculate the local gradient.
			 * \param curr the current layer.
			 * \param next the next layer.
			 */
			void localGradient(MultiLayerPerceptron::Layer& curr, 
					MultiLayerPerceptron::Layer& next);

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
			void gradient(MultiLayerPerceptron::Layer& first, 
					std::vector<double>& in);

			/**Calculate gradient for the weights in the first layer. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * block version of gradient.
			 * \param first the first hidden layer with weights.
			 * \param in the input vector.
			 */
			void gradientBatch(MultiLayerPerceptron::Layer& first, 
					std::vector<double>& in);

			/**Calculate the gradient between two layers.
			 * \deprecated Use the gradientBatch instead since Error functions in
			 * NeuralNetHack should only do batch updating. It is up to the Trainer to
			 * divide up DataSet:s as to provide block or online updating.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradient(MultiLayerPerceptron::Layer& curr, 
					MultiLayerPerceptron::Layer& prev); 

			/**Calculate the gradient between two layers. This
			 * does not replace the previous gradients. Instead it adds to new
			 * gradient to the previous one. This is intended for use with the
			 * batch version of gradient.
			 * \param curr the current layer.
			 * \param prev the previous layer.
			 */
			void gradientBatch(MultiLayerPerceptron::Layer& curr, 
					MultiLayerPerceptron::Layer& prev); 

			double outputError(const std::vector<double>& out, 
					const std::vector<double>& dout) const;

			/**Set all gradients to zero. */
			void killGradients();

	};
}
#endif
