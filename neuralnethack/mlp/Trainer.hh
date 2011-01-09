/*$Id: Trainer.hh 1627 2007-05-08 16:40:20Z michael $*/

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


#ifndef __Trainer_hh__
#define __Trainer_hh__

#include "Error.hh"
#include "../datatools/DataSet.hh"

#include <string>
#include <ostream>

namespace MultiLayerPerceptron
{
	/**A base class representing the training of an MLP. */
	class Trainer
	{
		public:
			/**Basic destructor. */
			virtual ~Trainer();

			/**Return a pointer to the Mlp being trained.
			 * \return the mlp pointer.
			 */
			Mlp* mlp();
			
			/**Set the mlp pointer that shall be trained.
			 * \param mlp the pointer to the mlp that shall be trained.
			 */
			void mlp(Mlp* mlp);
			
			/**Get the DataSet this Trainer should use for training. 
			 * \return the pointer to the DataSet.
			 */
			DataTools::DataSet* data();

			/**Set the DataSet this Trainer should use for training. 
			 * \param d the pointer to the DataSet.
			 */
			void data(DataTools::DataSet* d);

			/**Return the error function this Trainer is using.
			 * \return the error function.
			 */
			Error* error();

			/**Set the error function to train this committee with.
			 * \param e the error function.
			 */
			void error(Error* e);

			/**Return the number of epochs to train for.
			 * \return the number of epochs.
			 */
			uint numEpochs() const;

			/**Set the number of epochs to train for.
			 * \param ne the number of epochs.
			 */
			void numEpochs(uint ne);
			
			/**Return the training error for this trainer. */
			double trainingError() const;

			/**Set the training error for this trainer.
			 * \param te the training error.
			 */
			void trainingError(double te);
			
			/**Get the batch size for this trainer.
			 * \return the batch size.
			 */
			uint batchSize() const;

			/**Set the batch size for this trainer.
			 * \param bs the batch size.
			 */
			void batchSize(uint bs);

			/**Tells wether this trainer has everything it needs in order to
			 * perform training.
			 * \return true if everything is ready, false if something is missing.
			 */
			bool isValid() const;

			/**Method used to train an MLP. This method sets the member
			 * attributes Mlp and DataSet and then performs training.
			 * \param mlp the MLP to train.
			 * \param dset the data set to train the MLP on.
			 * \param os the ostream to print the training process to.
			 */
			 void train(Mlp& mlp, DataTools::DataSet& dset, std::ostream& os);
			
			/**Method used to train a copy of the current MLP. 
			 * This creates a copy of the current 
			 * Mlp, trains it and returns it to the caller.
			 * The Error function in the Trainer is the same.
			 * \param dset the data set to train the MLP on.
			 * \param os the ostream to print the training process to.
			 * \return a pointer to the newly trained Mlp.
			 */
			Mlp* trainNew(DataTools::DataSet& dset, std::ostream& os);

			/**Method used to train a copy of the current MLP. 
			 * This creates a copy of the current 
			 * Mlp, trains it and returns it to the caller.
			 * The DataSet and the Error function in the Trainer is the same.
			 * \param os the ostream to print the training process to.
			 * \return a pointer to the newly trained Mlp.
			 */
			Mlp* trainNew(std::ostream& os);

			/**Method used to train an MLP. This uses the Mlp the DataSet
			 * and the Error function in the Trainer.
			 * \param os the ostream to print the training process to.
			 */
			virtual void train(std::ostream& os) = 0;

			/**Method that clones the Trainer and returns a copy of it.
			 * \return the clone of this Trainer.
			 */
			virtual Trainer* clone() const = 0;

		protected:
			/**Basic constructor.
			 * \param mlp the Mlp to train.
			 * \param data the DataSet to use.
			 * \param error the Error function to use.
			 * \param te the maximum training error allowed.
			 * \param bs the batch size.
			 */
			Trainer(Mlp& mlp, DataTools::DataSet& data, Error& error, double te, uint bs);

			/**Copy constructor. 
			 * \param trainer the Trainer object to copy.
			 */
			Trainer(const Trainer& trainer);

			/**Assignment operator.
			 * \param trainer the Trainer object to copy.
			 */
			Trainer& operator=(const Trainer& trainer);

			/**Check if convergence criterion is reached.
			 * \param ecurr the error in the current epoch.
			 * \param eprev the error from the previous epoch.
			 * \return true if criterion is met, false otherwise.
			 */
			bool hasConverged(double ecurr, double eprev) const;

			/**The Mlp this Trainer is using. */
			Mlp* theMlp;
			
			/**The DataSet this Trainer will be using. */
			DataTools::DataSet* theData;

			/**The error function. */
			Error* theError;

			/**The number of epochs to train for. */
			uint theNumEpochs;

			/**The error required to stop training. */
			double theTrainingError;

			/**The number of patterns to use every epoch. */
			uint theBatchSize;

		private:

	};
}
#endif
