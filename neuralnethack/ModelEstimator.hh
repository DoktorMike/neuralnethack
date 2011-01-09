/*$Id: ModelEstimator.hh 1627 2007-05-08 16:40:20Z michael $*/

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


#ifndef __ModelEstimator_hh__
#define __ModelEstimator_hh__

#include "EnsembleBuilder.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"

#include <vector>
#include <utility>
#include <iostream>

namespace NeuralNetHack
{
	/**A base class representing the different model estimation methods available.
	 * \sa CrossValidator, Bootstrapper.
	 */
	class ModelEstimator
	{
		public:
			/**Basic constructor. */
			ModelEstimator();

			/**Constructor taking the mandatory arguments for this class to
			 * function properly. This is probably what you want to use.
			 * \param eb the EnsembleBuilder to use to build the Ensemble.
			 * \param s the Sampler to use to sample the DataSet.
			 */
			ModelEstimator(EnsembleBuilder& eb, DataTools::Sampler& s);

			/**Basic destructor. */
			virtual ~ModelEstimator();

			/**The pointer to the pair of estimation values. The first element
			 * of the pair is the estimation value for the training run and
			 * the second is for the validation.
			 */
			std::pair<double, double>* estimateModel(
					double (*errorFunc)(Ensemble& committee, DataTools::DataSet& data));

			/**Virtual method that will estimate the model given the DataSet.
			 * \return the estimation value of the training and the validation.
			 */
			std::pair<double, double>* runAndEstimateModel(
					double (*errorFunc)(Ensemble& committee, DataTools::DataSet& data));

			/**Accessor for the EnsembleBuilder pointer.
			 * \return the pointer to the EnsembleBuilder.
			 */
			EnsembleBuilder* ensembleBuilder();

			/**Mutator for the EnsembleBuilder pointer.
			 * \param eb the pointer to the EnsembleBuilder to set.
			 */
			void ensembleBuilder(EnsembleBuilder* eb);

			/**Accessor for the Sampler pointer.
			 * \return the pointer to the Sampler.
			 */
			DataTools::Sampler* sampler();

			/**Mutator for the Data pointer.
			 * \param s the pointer to the Data to set.
			 */
			void sampler(DataTools::Sampler* s);

			/**Accessor for the Session vector.
			 * \return the Session vector.
			 */
			std::vector<Session>& sessions();

			/**Mutator for the Session vector.
			 * \param s the Session vector to set.
			 */
			void sessions(std::vector<Session>& s);

		private:
			/**Pointer to the EnsembleBuilder we will use for estimating. */
			EnsembleBuilder* theEnsembleBuilder;

			/**Pointer to the Sampler we will use to estimate the model with. */
			DataTools::Sampler* theSampler;

			/**A vector of Estimations. */
			std::vector<Session> theSessions;


	};
}
#endif
