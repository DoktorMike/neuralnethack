/*$Id: Saliency.hh 1616 2007-04-17 12:09:41Z michael $*/

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


#ifndef __Saliency_hh__
#define __Saliency_hh__

#include "mlp/Mlp.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "Ensemble.hh"

#include <vector>
#include <ostream>

namespace NeuralNetHack
{
	/**This namespace encloses functions for calculating salinecies for a
	 * given datapoint for a given Mlp or Ensemble.
	 */
	namespace Saliency
	{

		/**Propagates an input through an Mlp up to the last layer. This seems
		 * very similar to the propagate function in the mlp, but they differ
		 * in that this propagate does not activate the last layer, but thus
		 * only calculates the induced local field.
		 * \param mlp the mlp to propagate the input through.
		 * \param input the input to propagate.
		 * \return the local induced field for the output neuron.
		 */
		double propagate(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input);

		/**Calculate the saliency for each variable averaged over all
		 * datapoints and committee members.
		 * \param committee the committee to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(Ensemble& committee, DataTools::DataSet& data);

		/**Calculate the saliency for each variable averaged over all
		 * datapoints and committee members.
		 * \param committee the committee to use as a predictor
		 * \param pattern the pattern containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(Ensemble& committee, DataTools::Pattern& pattern);

		/**Calculate the saliency for each variable averaged over all
		 * datapoints.
		 * \param mlp the mlp to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& data);

		/**Calculate the saliency for each variables in the pattern.
		 * \param mlp the mlp to use as a predictor
		 * \param pattern the pattern containing all variables
		 * \return a vector of variable saliences
		 */
		std::vector<double> saliency(MultiLayerPerceptron::Mlp& mlp, DataTools::Pattern& pattern);

		/**Calculate the partial derivative of the output with respect to an
		 * input variable.
		 * \param mlp the mlp to use as a predictor
		 * \param input the input vector containing all variables
		 * \param index the index of the varible to use for the partial derivative
		 * \return the value of the partial derivative
		 */
		double derivative(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input, uint index);

		/**Calculate the partial derivative of the output with respect to an
		 * input variable. Note that the last activation function is not used
		 * in this derivative. Thus what we use is the linear combination of
		 * the activation of each hidden node plus a bias.
		 * \param mlp the mlp to use as a predictor
		 * \param input the input vector containing all variables
		 * \param index the index of the varible to use for the partial derivative
		 * \return the value of the partial derivative
		 */
		double derivative_inner(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input, uint index);

		/**Print the result of an Saliency calculation for all Inputs in the
		 * DataSet.
		 * \param os the output stream to write to.
		 * \param saliency the saliencies to write.
		 */
		void print(std::ostream& os, std::vector<double>& saliency);
	}
}

#endif
