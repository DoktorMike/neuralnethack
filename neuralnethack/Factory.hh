/*$Id: Factory.hh 1546 2006-04-18 08:38:01Z michael $*/

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


#ifndef __Factory_hh__
#define __Factory_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Mlp.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "ModelEstimator.hh"

namespace NeuralNetHack
{
	/**This namespace encloses a number of functions for creating various
	 * objects needed in the NeuralNetHack project. Most of these functions
	 * need the configuration class Config. Note that the destruction of the
	 * created objects are left to the user.
	 * \sa Config.
	 */
	namespace Factory
	{
		/**Creates an Mlp object.
		 * \param config the config options.
		 * \return the created Mlp.
		 */
		MultiLayerPerceptron::Mlp* createMlp(const Config& config);

		/**Creates an Error object.
		 * \param config the config options.
		 * \param data the DataSet to give the Error.
		 * \return the created Error.
		 */
		MultiLayerPerceptron::Error* createError(const Config& config, DataTools::DataSet& data);

		/**Creates a Trainer object.
		 * \param config the config options.
		 * \param data the DataSet to give the Trainer.
		 * \return the created Trainer.
		 */
		MultiLayerPerceptron::Trainer* createTrainer(const Config& config, DataTools::DataSet& data);

		/**Creates a Sampler object.
		 * \param config the config options.
		 * \param data the DataSet to give the Sampler.
		 * \return the created Sampler.
		 */
		DataTools::Sampler* createSampler(const Config& config, DataTools::DataSet& data);

		/**Creates an EnsembleBuilder object.
		 * \param config the config options.
		 * \param data the DataSet to give the EnsembleBuilder.
		 * \return the created EnsembleBuilder.
		 */
		EnsembleBuilder* createEnsembleBuilder(const Config& config, DataTools::DataSet& data);


		/**Creates a ModelEstimator object.
		 * \param config the config options.
		 * \param data the DataSet to give the ModelEstimator.
		 * \return the created ModelEstimator.
		 */
		ModelEstimator* createModelEstimator(const Config& config, DataTools::DataSet& data);
	}
}

#endif
