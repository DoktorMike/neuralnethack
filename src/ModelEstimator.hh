#ifndef __ModelEstimator_hh__
#define __ModelEstimator_hh__

#include "EnsembleBuilder.hh"
#include "Committee.hh"
#include "datatools/DataSet.hh"

#include <vector>
#include <utility>
#include <iostream>

namespace NeuralNetHack
{
	class ModelEstimator
	{
		public:
			virtual ~ModelEstimator();
			virtual std::pair<double, double>* estimateModel() = 0;

			EnsembleBuilder* ensembleBuilder();
			void ensembleBuilder(EnsembleBuilder* eb);
			DataTools::DataSet* data();
			void data(DataTools::DataSet* d);

			/**Print the output and target for each data point in the DataSet.
			 * \param os the output stream to write to.
			 * \todo The function shouldn't assume single output.
			 */
			virtual void printOutputTargetList(std::ostream& os);

		protected:
			ModelEstimator();
			ModelEstimator(const ModelEstimator& me);
			ModelEstimator& operator=(const ModelEstimator& me);

			struct Estimation
			{
				Committee* committee;
				DataTools::DataSet* trnData;
				DataTools::DataSet* valData;
			};

			std::pair<double, double>* calcMeanTrnValAuc();

			EnsembleBuilder* theEnsembleBuilder;
			DataTools::DataSet* theData;
			std::vector<Estimation> theEstimations;

		private:

	};
}
#endif
