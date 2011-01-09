#ifndef __CrossValidator_hh__
#define __CrossValidator_hh__

#include "Config.hh"
#include "datatools/DataSet.hh"
#include "mlp/Mlp.hh"
#include "Trainer.hh"
#include <iostream>

namespace NeuralNetHack
{
	using DataTools::DataSet;
	using MultiLayerPerceptron::Mlp;

	/**A class representing N K-fold cross validation runs. A DataSet is divided
	 * into K parts. Each part is then used as a validation set e.g. choose
	 * part 1 as validation set and train an mlp on parts 2-K, then choose
	 * part 2 as validation set and train another mlp on parts 1,3-K etc.
	 * This procedure is repeated N times yielding a total of N*K models.
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
			 * \return a copy of the CrossValidator.
			 */
			CrossValidator& operator=(const CrossValidator& cv);

			vector<double>& validateOutput();

			void validateOutput(vector<double>& vo);

			vector<uint>& validateTarget();

			void validateTarget(vector<uint>& vt);

			void crossValidate(Trainer& trainer, DataSet& dataSet, uint n, uint k);

			/**Print the statistics for one or all of the n*k runs.
			 * \param os the output stream to write to.
			 * \param together flag whether to print statistics for the total
			 * N*K cross validations or print the statistics individually.
			 * \param plot toggle ROC plot in the output.
			 */
			void printValidationStats(ostream& os, uint n, uint k, bool plot=false);

			/**Print the statistics for one or all of the n*k runs.
			 * \param os the output stream to write to.
			 * \param together flag whether to print statistics for the total
			 * N*K cross validations or print the statistics individually.
			 * \param plot toggle ROC plot in the output.
			 */
			void printTrainingStats(ostream& os, uint n, uint k, bool plot=false);

			/**Print the statistics for one or all of the n*k runs.
			 * \param os the output stream to write to.
			 * \param together flag whether to print statistics for the total
			 * N*K cross validations or print the statistics individually.
			 * \param validation flag whether to print statistics for
			 * validation set or training set.
			 * \param plot toggle ROC plot in the output.
			 */
			void printStats(ostream& os, uint n, uint k, bool validation=true, bool plot=false);

			class CrossResult
			{
				public:
					CrossResult():mlp(0){}
					CrossResult(Mlp* m, DataSet& ts, DataSet& vs):
						mlp(m),trainingSet(ts),validationSet(vs){}
					CrossResult(const CrossResult& cr){*this = cr;}
					~CrossResult(){}
					CrossResult& operator=(const CrossResult& cr)
					{
						if(this != &cr){
							mlp = cr.mlp;
							trainingSet = cr.trainingSet;
							validationSet = cr.validationSet;
						}
						return *this;
					}
					Mlp* mlp;
					DataSet trainingSet;
					DataSet validationSet;
			};

		private:
			void crossValidate(Trainer& trainer, DataSet& dataSet, uint k);

			void validate(Mlp& mlp, DataSet& validationSet);

			uint numRuns;

			uint numParts;

			vector<double> valOutput;

			vector<uint> valTarget;

			vector<CrossResult> crossResults;

	};
}
#endif
