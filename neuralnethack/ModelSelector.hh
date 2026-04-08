/*$Id: ModelSelector.hh 1648 2007-06-01 15:39:10Z michael $*/

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

#ifndef __ModelSelector_hh__
#define __ModelSelector_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"

#include <vector>
#include <utility>

namespace NeuralNetHack
{
/**Selecting the optimal model by performing a grid search over the chosen
 * parameters.
 */
class ModelSelector
{
	public:
		ModelSelector();
		virtual ~ModelSelector();
		std::pair<Config, double> findBestModel(DataTools::DataSet& trnData, Config& config);

	private:
		/** Method for constructing a full sequence from a vector seq.
		 * \param seq vector containing the start stop and increment value.
		 */
		std::vector<double> sequence(const std::vector<double>& seq);

		std::pair<double, double>* trainAndValidateModel(DataTools::DataSet& trnData, const Config& config);

		/** Calculate the .632+ rule.
		 * \param gamma The no-information error rate. For AUC it's 0.5.
		 * \param meanTrn The mean training performance over the bootstrap samples.
		 * \param meanTst The mean testing performance over the bootstrap samples.
		 */
		double Auc632PlusRule(double meanTrn, double meanTst);
};
}
#endif
