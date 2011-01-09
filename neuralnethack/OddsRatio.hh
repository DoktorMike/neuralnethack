/*$Id: OddsRatio.hh 1676 2007-09-12 14:36:58Z michael $*/

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
#include <functional>

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
		 * of data points, and \f$C\f$ is the total number of ensemble members.
		 * \param mlp the Mlp to use.
		 * \param pattern the DataSet to calculate OR:s for each data point on.
		 * \return the average OR for each Mlp and input in the DataSet.
		 */
		std::vector<double> oddsRatio(Ensemble& ensemble, DataTools::DataSet& data);

		/**Calculates the odds ratios for each variable in the pattern. 
		 * This performs an extra sum where we simply take the
		 * average over Mlp:s. \f[\sum_c^C \overline{OR_c}\f] where \f$C\f$ 
		 * is the total number of ensemble members.
		 * \param mlp the Mlp to use.
		 * \param pattern the DataSet to calculate OR:s for each data point on.
		 * \return the average OR for each Mlp and input in the DataSet.
		 */
		std::vector<double> oddsRatio(Ensemble& ensemble, DataTools::Pattern& pattern);

		/**Print the result of an OddsRatio calculation for all Inputs in the
		 * DataSet.
		 * \param os the output stream to write to.
		 * \param oddsrat the oddsratios to write.
		 */
		void print(std::ostream& os, std::vector<double>& oddsrat);

	}
}

#endif
