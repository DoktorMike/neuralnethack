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
