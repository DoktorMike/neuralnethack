#ifndef __Trainer_hh__
#define __Trainer_hh__

#include "datatools/DataSet.hh"
#include "Error.hh"

#include <string>

namespace NeuralNetHack
{
	using DataTools::DataSet;
	using DataTools::Pattern;
	using std::string;

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

			/**Accessor for theWeightElimOn.
			 * \return the theWeightElimOn.
			 */
			bool weightElimOn();

			/**Mutator for theWeightElimOn.
			 * \param on the value to set.
			 */
			void weightElimOn(bool on);

			/**Accessor for theWeightElimAlpha.
			 * \return the theWeightElimAlpha.
			 */
			double weightElimAlpha();

			/**Mutator for theWeightElimAlpha.
			 * \param alpha the value to set.
			 */
			void weightElimAlpha(double alpha);

			/**Accessor for theWeightElimW0.
			 * \return the theWeightElimW0.
			 */
			double weightElimW0();

			/**Mutator for theWeightElimW0.
			 * \param w0 the value to set.
			 */
			void weightElimW0(double w0);

			/**Tells wether this trainer has everything it needs in order to
			 * perform training.
			 * \return true if everything is ready, false if something is missing.
			 */
			bool validate();

			/**Method used to train an MLP.
			 * \param mlp the MLP to train.
			 * \param dset the data set to train the MLP on.
			 */
			virtual void train(Mlp& mlp, DataSet& dset)=0;

		protected:
			/**Basic constructor.
			 * \param te the maximum training error allowed.
			 * \param bs the batch size.
			 * \param we toggle weight elimination.
			 * \param alpha the importance of the weight elimination term.
			 * \param w0 the scaling factor of the weight elimination.
			 */
			Trainer(double te, uint bs, bool we, double alpha, double w0);

			/**Basic constructor. */
			Trainer();

			/**Calculates the weight elimination term. 
			 * \param wi the weight to regularize.
			 * \return the weight elimination term.
			 */
			double weightElimination(double wi);

			/**Check if convergence criterion is reached.
			 * \param ecurr the error in the current epoch.
			 * \param eprev the error from the previous epoch.
			 * \return true if criterion is met, false otherwise.
			 */
			bool hasConverged(double ecurr, double eprev);

			/**The error function. */
			Error* theError;

			/**The number of epochs to train for. */
			uint theNumEpochs;

			/**The error required to stop training. */
			double theTrainingError;

			/**The number of patterns to use every epoch. */
			uint theBatchSize;

			/**The vector representing the weight update.
			 * \deprecated This is no longer used in GradientDescent or
			 * QuasiNewton learning.
			 */
			vector<double> theWeightUpdate;

			/**Controls whether to use weight elimination or not. */
			bool theWeightElimOn;

			/**The importance of the weight elimination term. */
			double theWeightElimAlpha;

			/**Scaling factor typically set to unity. */
			double theWeightElimW0;

		private:

			/**Copy constructor. 
			 * \param trainer the Trainer object to copy.
			 */
			Trainer(const Trainer& trainer);

			/**Assignment operator.
			 * \param trainer the Trainer object to copy.
			 */
			Trainer& operator=(const Trainer& trainer);

	};
}
#endif
