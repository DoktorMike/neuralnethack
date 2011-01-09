/*$Id: GradientDescent.hh 1684 2007-10-12 15:55:07Z michael $*/

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


#ifndef __GradientDescent_hh__
#define __GradientDescent_hh__

#include "Trainer.hh"

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Trainer interface.
	 * This learning algorithm is called Gradient Descent and
	 * uses only first order gradients of the error function.
	 * The weight update rule:
	 * \f[\omega_{t+1}=\omega_t - \eta_t\cdot\frac{\partial E}
	 * {\partial\omega_t} + \alpha\cdot\Delta\omega_t\f] where 
	 * \f[ \eta_{t+1} = \left\{\begin{array}{ll}
	 * \eta_t * \gamma, & E_{t+1}>E_t;\\ 
	 * \eta_t * \left(1+\frac{1-\gamma}{10}\right), & otherwise;
	 * \end{array}\right. \f]
	 */
	class GradientDescent: public Trainer
	{

		public:
			/**Basic constructor.
			 * \param mlp the Mlp to train.
			 * \param data the DataSet to use.
			 * \param error the Error function to use.
			 * \param te the training error at which to stop training.
			 * \param bs the batch size.
			 * \param lr the learning rate.
			 * \param dlr the decrease of learning rate.
			 * \param m the momentum term.
			 */
			GradientDescent(Mlp& mlp, DataTools::DataSet& data, Error& error, 
					double te, uint bs, double lr, double dlr, double m);

			/**Basic destructor. */
			~GradientDescent();

			/**Method used to train an MLP. This uses the Mlp and the DataSet
			 * in the Trainer.
			 */
			void train(std::ostream& os);

			/**Set the learning rate for this trainer.
			 * \param lr the learning rate.
			 */
			void learningRate(double lr);

			/**Set the decrease of learning rate for this trainer.
			 * \param dlr the learning rate.
			 */
			void decLearningRate(double dlr);

			/**Set the momentum for this trainer.
			 * \param m the momentum term.
			 */
			void momentum(double m);

			/**Return the batch size for this trainer. */
			uint batchSize();

			/**Return the learning rate this trainer is using. */
			double learningRate() const;

			/**Return the decrease of learning rate this trainer is using. */
			double decLearningRate() const;

			/**Return the momentum term this trainer is using. */
			double momentum() const;

			Trainer* clone() const;

		private:
			/**Copy constructor.
			 * \param gd the object to copy from.
			 */
			GradientDescent(const GradientDescent& gd);

			/**Assignment operator.
			 * \param gd the object to assign from.
			 */
			GradientDescent& operator=(const GradientDescent& gd);

			/**Method used to train an MLP during one epoch.
			 * \param dset the data set to train the MLP on.
			 */
			double train(DataTools::DataSet& dset);

			/**Updates the learning rate.
			 * \f[ \eta_{t+1} = \left\{\begin{array}{ll}
			 * \eta_t * \gamma, & E_{t+1}>E_t;\\ 
			 * \eta_t * \left(1+\frac{1-\gamma}{10}\right), & otherwise;
			 * \end{array}\right. \f]
			 */
			void updateLearningRate(double err, double prevErr);

			/**Build the block update DataSet.
			 * \param blockData the DataSet that will contain the new block.
			 * \param cntr the cntr of where we last left off in the original DataSet.
			 * \return true if cntr was resest, false otherwise.
			 */
			bool buildBlock(DataTools::DataSet& blockData, uint& cntr) const;

			/**The learning rate. */
			double theLearningRate;

			/**The decrease of the learning rate.
			 * This governs how much the learning rate should be decreased
			 * when training error goes up. The value is typically close to
			 * but always less than 1.0.
			 */
			double theDecLearningRate;

			/**The momentum term.
			 * This governs how much of the previous weight update should be
			 * included in the current update. This value is typically set
			 * between 0.8 and 0.99. Never more than 1.0.
			 */
			double theMomentum;

	};
}
#endif
