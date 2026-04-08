#ifndef __Gof_hh__
#define __Gof_hh__

#include <vector>
#include <utility>
#include <cstdlib>

namespace EvalTools
{
	/**A class representing the Hosmer-Lemeshow goodness of fit test.
	 * \f[\chi ^2 = \sum_{j=1}^{G}\frac{(o_j - n_j \bar{\pi}_j)^2}{ n_j \bar{\pi}_j (1 - \bar{\pi}_j) }\f]
	 * Where \f$o_j\f$ is the number of observed positives in bin j, and 
	 * \f$\bar{\pi}_j\f$ is the mean average predicted value in bin j. G is
	 * the number of bins meanwhile \f$n_j\f$ is the number of samples in the
	 * bin.This test statistics follow the chi square statistics with a (G-2)
	 * deegree of freedom.
	 */
	class Gof
	{
		public:
			/**Basic constructor.
			 * \param nb the number of bins to split data into.
			 */
			Gof(uint nb);

			/**Calculate the goodness of fit measure using hosmer lemeshow
			 * test statistics.
			 * \param output the output from our model.
			 * \param target the target for the output.
			 * \return the goodness of fit measure.
			 */
			double goodnessOfFit(const std::vector<double>& output, 
					const std::vector<uint>& target);

		private:

			/**Convert and sort a pair of vectors into a vector of pairs.
			 * \param output the predictors output.
			 * \param target the predictors target.
			 * \return the output and target in one vector of pairs.
			 */
			std::vector< std::pair<double, uint> > doSort(const std::vector<double>& output, 
					const std::vector<uint>& target);

			/**The number of bins to split the data into. */
			uint numBins;
	};
}

#endif
