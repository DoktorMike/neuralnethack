#ifndef __CrossValidator_hh__
#define __CrossValidator_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Mlp.hh"
#include "Trainer.hh"
#include "Committee.hh"
#include <iostream>

namespace NeuralNetHack
{
	using DataTools::DataSet;
	using MultiLayerPerceptron::Mlp;

	/**A class representing N runs of K-fold cross validation. 
	 * A DataSet is divided into K parts. Each part is then used as a 
	 * validation set e.g. choose part 1 as validation set and train an 
	 * mlp on parts 2-K, then choose part 2 as validation set and train 
	 * another mlp on parts 1,3-K etc. This procedure is repeated N 
	 * times yielding a total of N*K models. Note that this is currently
	 * limited to single output binary class problems. Another thing to note
	 * is that the AUC given from this cross validation scheme is the ensemble
	 * of N models on the validation part and ensamble of (K-1)*N on the
	 * training part. This yields a much better AUC than just averaging over
	 * the N*K seperately calculated AUC.
	 * \todo Implement the posibility of getting an AUC based on the average
	 * of N*K separately calculated AUC(s).
	 */
	class CrossValidator
	{
		public:
			/**Basic contructor. */
			CrossValidator();
			
			/**Copy constructor. 
			 * \param cv the CrossValidator to copy from.
			 */
			CrossValidator(const CrossValidator& cv);
			
			/**Basic destructor. */
			virtual ~CrossValidator();
			
			/**Assignment operator.
			 * \param cv the CrossValidator to assign from.
			 * \return the CrossValidator assigned to.
			 */
			CrossValidator& operator=(const CrossValidator& cv);

			/**Return the output from the Mlp(s) on the validation set(s).
			 * \return the validation output.
			 */
			vector<double>& validateOutput();

			/**Set the output from the Mlp(s) on the validation set(s).
			 * \param vo the validation output to set.
			 */
			void validateOutput(vector<double>& vo);

			/**Return the targets from the validation set(s).
			 * \return the validation targets.
			 */
			vector<uint>& validateTarget();

			/**Set the targets from the validation set(s).
			 * \param vt the validation targets to set.
			 */
			void validateTarget(vector<uint>& vt);

			/**Get the AUC for the validation or the training.
			 * \param validation flag whether to give auc for training or for
			 * validation.
			 * \return the AUC.
			 */
			double auc(bool validation=true);

			/**Return the committee created by this CrossValidator.
			 * \return the committee.
			 */
			Committee& committee();

			/**Set the committee.
			 * This has very little meaning in the current setup.
			 * \param c the committee to set.
			 */
			void committee(Committee& c);
		
			/**Perform a N cross validations consisting of only 2 parts.
			 * There are a training part and a validation part.
			 * \param trainer the minimisation method to train the models
			 * with.
			 * \param ds the DataSet to perform the cross validation on.
			 * \param n the number of cross validations to perform.
			 * \param ratio the trainingsize/validationsize ratio.
			 */
			void crossValidate(Trainer& trainer, DataSet& ds, uint n, double ratio);

			/**Perform N runs of K-fold cross validations.
			 * \param trainer the minimisation method to train the models
			 * with.
			 * \param dataSet the DataSet to perform the cross validation on.
			 * \param n the number of cross validations to perform.
			 * \param k the number of parts to split the DataSet into.
			 */
			void crossValidate(Trainer& trainer, DataSet& dataSet, uint n, uint k);

		private:
			/**Performs a standard K-fold cross validation.
			 * \param trainer the minimisation method to train the models
			 * with.
			 * \param dataSet the DataSet to split up in K equal parts.
			 * \param k the number of parts to slit the data set into.
			 */
			void crossValidate(Trainer& trainer, DataSet& dataSet, uint k);

			/**Validates the model on the given validation set.
			 * Adds the output from the mlp on the validation set to the
			 * valOutput. Also adds the target for each point to valTarget.
			 * \param mlp the model to validate.
			 * \param validationSet the validation set to use.
			 * \return the calculated AUC.
			 */
			double validate(Mlp& mlp, DataSet& validationSet);

			/**Add outputs for each data point to valOutput and valTarget.
			 * Thus for each data point in the data set the output from the
			 * model is added to the valOutput. Resulting in valOutput
			 * containing N outputs for each data point. Note that this is
			 * only valid for the validation part.
			 * \param mlp the model to which output to add.
			 * \param ds the data set containing the data points to add output
			 * for.
			 * \param validation flag whether to add to validation or
			 * training.
			 */
			void addToCommitteeOutput(Mlp& mlp, DataSet& ds, bool validation=true);
			
			/**Convert run n and part k to an index.
			 * \param n the wanted run.
			 * \param k the wanted part.
			 * \return the corresponding index.
			 */
			uint nkToIndex(uint n, uint k);
			
			/**The number of runs this crossvalidator will perform. */
			uint numRuns;

			/**The number of parts this crossvalidator will split the data in. */
			uint numParts;

			/**The output on the validation set from a trained model.
			 * Note that the output from K*N models evaluate every data point
			 * here and thus we get a committee output by averaging their
			 * output.
			 */
			vector<double> valOutput;

			/**The target for the validation set. */
			vector<uint> valTarget;

			/**The output on the training set from a trained model.
			 * Note that the output from (K-1)*N models evaluate every data point
			 * here and thus we get a committee output by averaging their
			 * output.
			 */
			vector<double> trnOutput;

			/**The target for the training set. */
			vector<uint> trnTarget;

			/**The total AUC for the training. */
			double aucTraining;
			
			/**The total AUC for the validation. */
			double aucValidation;

			/**The committee consisting of N*K models. */
			Committee* theCommittee;

			//vector<uint> comSizeVal;
			//vector<uint> comSizeTrn;

	};
}
#endif
