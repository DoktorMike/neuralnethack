#ifndef __Trainer_hh__
#define __Trainer_hh__

#include "datatools/DataSet.hh"
#include "Error.hh"

#include <string>

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
			 * This method deletes eventual previous error function
			 * and creates a new one as specified by the string e.
			 * \param e the error function.
			 */
			void error(Error* e);

			/**Return the number of epochs to train for.
			 * \return the number of epochs.
			 */
			uint numEpochs();

			/**Set the number of epochs to train for.
			 * \param ne the number of epochs.
			 */
			void numEpochs(uint ne);
			
			/**Return the training error for this trainer. */
			double trainingError();

			/**Set the training error for this trainer.
			 * \param te the training error.
			 */
			void trainingError(double te);
			
			/**Get the batch size for this trainer.
			 * \return the batch size.
			 */
			uint batchSize();

			/**Set the batch size for this trainer.
			 * \param bs the batch size.
			 */
			void batchSize(uint bs);

			/**Tells wether this trainer has everything it needs in order to
			 * perform training.
			 * \return true if everything is ready, false if something is missing.
			 */
			bool isValid();

			/**Method used to train an MLP. This method sets the member
			 * attributes Mlp and DataSet and then performs training.
			 * \param mlp the MLP to train.
			 * \param dset the data set to train the MLP on.
			 */
			 void train(Mlp& mlp, DataTools::DataSet& dset);
			
			/**Method used to train an MLP. This uses the Mlp and the DataSet
			 * in the Trainer.
			 */
			virtual void train()=0;

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
			bool hasConverged(double ecurr, double eprev);

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
