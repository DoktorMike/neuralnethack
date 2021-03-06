/*$Id: Error.hh 1654 2007-07-04 21:55:36Z michael $*/

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


#ifndef __Error_hh__
#define __Error_hh__

#include "Mlp.hh"
#include "../datatools/DataSet.hh"

#include <vector>

namespace MultiLayerPerceptron
{
	/**A base class representing the error functions. The error function is
	 * responsible for calculating an error and a gradient to update the Mlp
	 * weights by. Also every Error function use full batch mode i.e. it is up
	 * to the Trainer to divide it into block or online by dividing the
	 * DataSet that it receives into sub DataSet:s. The impementation specific 
	 * Error function \f[E\f] is extended with 
	 * \f[\hat{E} = E + \nu\sum_i\frac{\omega_i^2}{\omega_0^2 + \omega_i^2}\f]
	 * \sa SummedSquare, CrossEntropy, DataSet, Trainer.
	 */
	class Error
	{
		public:		
			/**Basic destructor. */
			virtual ~Error();

			/**Calculate the gradient vector, and return the error.
			 * This returns the error calculated from a block of patterns.
			 * \param mlp the MLP to calculate the gradient for.
			 * \param dset the data set to use.
			 * \return the error.
			 */
			virtual double gradient(MultiLayerPerceptron::Mlp& mlp, 
					DataTools::DataSet& dset) = 0;

			/**Calculate the gradient vector, and return the error.
			 * This returns the error calculated from a block of patterns.
			 * \return the error.
			 */
			virtual double gradient() = 0;

			/**Calculate the summed square error for this Mlp and DataSet.
			 * \param mlp the output from the MLP.
			 * \param dset the desired output.
			 * \return the error.
			 */
			virtual double outputError(MultiLayerPerceptron::Mlp& mlp, 
					DataTools::DataSet& dset) = 0;

			/**Calculate the error for this Mlp and DataSet.
			 * \return the error.
			 */
			virtual double outputError() const = 0;

			/**Accessor method for the Mlp.
			 * \return the Mlp belonging to the Error.
			 */
			MultiLayerPerceptron::Mlp& mlp();

			/**Mutator method for the Mlp.
			 * \param mlp the Mlp to assign to this Error.
			 */
			void mlp(MultiLayerPerceptron::Mlp& mlp);

			/**Accessor method for the DataSet.
			 * \return the DataSet belonging to the Error.
			 */
			DataTools::DataSet& dset();

			/**Mutator method for the DataSet.
			 * \param dset the DataSet to assign to this Error.
			 */
			void dset(DataTools::DataSet& dset);

			/**Accessor for theWeightElimOn.
			 * \return the theWeightElimOn.
			 */
			bool weightElimOn() const;

			/**Mutator for theWeightElimOn.
			 * \param on the value to set.
			 */
			void weightElimOn(bool on);

			/**Accessor for theWeightElimAlpha.
			 * \return the theWeightElimAlpha.
			 */
			double weightElimAlpha() const;

			/**Mutator for theWeightElimAlpha.
			 * \param alpha the value to set.
			 */
			void weightElimAlpha(double alpha);

			/**Accessor for theWeightElimW0.
			 * \return the theWeightElimW0.
			 */
			double weightElimW0() const;

			/**Mutator for theWeightElimW0.
			 * \param w0 the value to set.
			 */
			void weightElimW0(double w0);

		protected:	

			/**Basic constructor.
			 * \param mlp the mlp to use.
			 * \param dset the dataset to use.
			 */
			Error(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& dset);

			/**Calculate the error for this output.
			 * \param out the output from the MLP.
			 * \param dout the desired output.
			 * \return the error.
			 */
			virtual double outputError(const std::vector<double>& out, 
					const std::vector<double>& dout) const = 0;

			/**Calculates the gradient of the weight elimination term. 
			 * \param wi the weight to regularize.
			 * \return the weight elimination term.
			 */
			double weightElimGrad(double wi) const;

			/**Add the gradient of the weight elimination term to each weight 
			 * in the specified interval. 
			 * The offset defines the starting index while length
			 * defies the number of elements to update.
			 * \param gradients the gradients to regularize.
			 * \param offset the offset.
			 * \param length the length.
			 */
			void weightElimGrad(std::vector<double>& gradients, 
					const std::vector<double>& weights, 
					uint offset, uint length) const;

			/**Add the gradient of the weight elimination term to each 
			 * gradient except the bias.
			 * \param gradients the gradient vector for the entire Mlp.
			 * \param ncurr the number of neurons in current layer.
			 * \param nprev the number of neurons in previous layer.
			 */
			void weightElimGradLayer(std::vector<double>& gradients, 
					const std::vector<double>& weights, 
					uint ncurr, uint nprev) const;

			/**Add the gradient of the weight elimination term to each 
			 * gradient except the bias.
			 * \param gradients the gradient vector for the entire Mlp.
			 * \param arch the architecture for the Mlp.
			 */
			void weightElimGradMlp(std::vector<double>& gradients, 
					const std::vector<double>& weights, 
					const std::vector<uint>& arch) const;

			/**Add the gradient of the weight elimination term to each 
			 * gradient in the Mlp.
			 * The bias is skipped as usual.
			 */
			void weightElimGrad();

			/**Calculates the weight elimination term. 
			 * \return the weight elimination term.
			 */
			double weightElim() const;

			/**The Mlp associated with an Error.*/
			MultiLayerPerceptron::Mlp* theMlp;

			/**The DataSet associated with an Error.*/
			DataTools::DataSet* theDset;

			/**Controls whether to use weight elimination or not. */
			bool theWeightElimOn;

			/**The importance of the weight elimination term. */
			double theWeightElimAlpha;

			/**Scaling factor typically set to unity. */
			double theWeightElimW0;

		private:
			/**Copy constructor. */
			Error(const Error&);

			/**Assignment operator. */
			Error& operator=(const Error&);

	};
}
#endif
