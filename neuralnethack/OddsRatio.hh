/*$Id: OddsRatio.hh 1595 2007-01-12 16:24:32Z michael $*/

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


#ifndef __OddsRatio_hh__
#define __OddsRatio_hh__

#include "mlp/Mlp.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "Ensemble.hh"

#include <vector>
#include <ostream>

namespace NeuralNetHack
{
	/**This namespace encloses functions for calculating effective odds
	 * ratios. It is currently limited to single output networks using a
	 * sigmoidal activation function in the output layer. We define the Odds
	 * for an input to be
	 * \f[Odds = \frac{y^{present}}{1-y^{present}}\f]
	 * where \f$y^{present}\f$ is the output from the Mlp given that the input
	 * is present. Thus, the odds ratio is defined as
	 * \f[OR = \frac{ y^{present}(1-y^{absent}) }{ y^{absent}(1-y^{present}) }\f]
	 */
	namespace OddsRatio
	{

		/**Calculates the odds ratios for each data point and mlp and return the
		 * average. This performs an extra sum where we simply take the
		 * average over Mlp:s and data points.
		 * \f[\sum_c^C\sum_n^N \overline{OR_n}\f] where \f$N\f$ is the total number 
		 * of data points, and \f$C\f$ is the total number of committee members.
		 * \param mlp the Mlp to use.
		 * \param pattern the DataSet to calculate OR:s for each data point on.
		 * \return the average OR for each Mlp and input in the DataSet.
		 */
		std::vector<double> oddsRatio(Ensemble& committee, DataTools::DataSet& data);

		/**Calculates the odds ratios for each variable in the pattern. 
		 * This performs an extra sum where we simply take the
		 * average over Mlp:s. \f[\sum_c^C \overline{OR_c}\f] where \f$C\f$ 
		 * is the total number of committee members.
		 * \param mlp the Mlp to use.
		 * \param pattern the DataSet to calculate OR:s for each data point on.
		 * \return the average OR for each Mlp and input in the DataSet.
		 */
		std::vector<double> oddsRatio(Ensemble& committee, DataTools::Pattern& pattern);

		/**Calculates the odds ratios for each data point and return the
		 * average.
		 * \f[\sum_n^N \overline{OR_n}\f] where \f$N\f$ is the total number of data
		 * points.
		 * \param mlp the Mlp to use.
		 * \param pattern the DataSet to calculate OR:s for each data point on.
		 * \return the average OR for each input in the DataSet.
		 */
		std::vector<double> oddsRatio(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& data);

		/**Calculates the odds ratio for each input value. The odds ratio for
		 * each input is calculated by setting it to zero and
		 * propagate it through the Mlp. The comparison is made by propagating
		 * the original input pattern through the mlp and take the ratio
		 * between them. After a few simplifications it falls out as
		 * \f[\exp(g(\overline{\omega}, \overline{x^{absent}})-g(\overline{\omega}, \overline{x^{present}}))\f]
		 * where \f$g\f$ is the local induced field for the ouput neuron given
		 * input and weights.
		 * \param mlp the Mlp to use.
		 * \param pattern the Pattern to calculate OR:s for each input on.
		 * \return the OR for each input in pattern.
		 */
		std::vector<double> oddsRatioComplex(MultiLayerPerceptron::Mlp& mlp, DataTools::Pattern& pattern);

		/**Calculates the odds ratio for each input value. The odds ratio for
		 * each input is calculated by setting it to zero and
		 * propagate it through the Mlp. The comparison is made by propagating
		 * the original input pattern through the mlp and take the ratio
		 * between them. This is slightly more computationally demanding than
		 * the Complex version.
		 * \param mlp the Mlp to use.
		 * \param pattern the Pattern to calculate OR:s for each input on.
		 * \return the OR for each input in pattern.
		 */
		std::vector<double> oddsRatioSimple(MultiLayerPerceptron::Mlp& mlp, DataTools::Pattern& pattern);

		/**Propagates an input through an Mlp up to the last layer. This seems
		 * very similar to the propagate function in the mlp, but they differ
		 * in that this propagate does not activate the last layer, but thus
		 * only calculates the induced local field.
		 * \param mlp the mlp to propagate the input through.
		 * \param input the input to propagate.
		 * \return the local induced field for the output neuron.
		 */
		double propagate(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input);

		/**Print the result of an OddsRatio calculation for all Inputs in the
		 * DataSet.
		 * \param os the output stream to write to.
		 * \param oddsrat the oddsratios to write.
		 */
		void print(std::ostream& os, std::vector<double>& oddsrat);
	}
}

#endif
