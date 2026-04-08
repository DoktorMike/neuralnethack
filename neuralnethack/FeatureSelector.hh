/*$Id: FeatureSelector.hh 1701 2008-02-02 23:25:34Z michael $*/

/*
Neuralnethack - A program for training and validating ensembles of MLPs
Copyright © 2007  Michael Green

This file is part of Neuralnethack.

Neuralnethack is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

Neuralnethack is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Neuralnethack; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

Michael Green <michael@thep.lu.se>
*/

#ifndef __FeatureSelector_hh__
#define __FeatureSelector_hh__

#include <ostream>

#include "Config.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"

namespace NeuralNetHack
{
	class FeatureSelector
	{
		public:

			/**Basic constructor for the FeatureSelector class.
			 * \param minf the minimum number of features to keep
			 * \param maxf the maximum number of features to keep
			 * \param maxf the maximum number of features to remove in each iteration
			 */
			FeatureSelector(uint minf, uint maxf, uint maxr);

			/**Runs the feature selction process.
			 * \param config the Config object to read configuration from
			 * \param f the evaluator function for assessing the quality of a feature set
			 * \return the new Config object with the reduced feature set.
			 */
			Config run(Config& config, double (*f)(Ensemble&, DataTools::DataSet&));

		private:

			/**Parses the data files given in config using the input columns
			 * given in config.
			 * \param config the Config to use.
			 * \param trnData the training DataSet to fill up.
			 * \param tstData the test DataSet to fill.
			 */
			void parseData(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData);

			/**Selects the features to remove based on the given values in effects.
			 * \param effects the effects removing a feature had on the performance measure.
			 */
			std::vector<uint> removeFeatures(std::vector<double>& effects, const std::vector<uint>& inputs);

			void storeClampingEffect(Ensemble& e, DataTools::DataSet& d, double auc, 
					std::vector<double>& aucImpact, double (*f)(Ensemble&, DataTools::DataSet&));

			/**Calculates the mean value for a specific variable in the given DataSet.
			 * \param d the DataSet to find the variable in.
			 * \param index the index of the variable to calculate the mean for.
			 * \return the mean for the variable with the given index.
			 */
			double mean(DataTools::DataSet& d, uint index);

			/** The minimum number of features to keep in the model. */
			uint minFeatures;
			/** The maximum number of features to keep in the model. */
			uint maxFeatures;
			/** The maximum number of features to remove in each iteration from the model. */
			uint maxRemove;
			/** The best Config setting so far. */
			Config best;
	};
}
#endif
