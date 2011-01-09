#ifndef __EnsembleBuilder_hh__
#define __EnsembleBuilder_hh__

#include "mlp/Trainer.hh"
#include "Committee.hh"
#include "datatools/DataSet.hh"
#include "datatools/DataManager.hh"

namespace NeuralNetHack
{

	class EnsembleBuilder
	{
		public:
			virtual ~EnsembleBuilder();
			virtual Committee* buildEnsemble() = 0;

			DataTools::DataManager* dataManager();
			void dataManager(DataTools::DataManager* dm);
			MultiLayerPerceptron::Trainer* trainer();
			void trainer(MultiLayerPerceptron::Trainer* t);
			DataTools::DataSet* data();
			void data(DataTools::DataSet* d);
			bool randomSampling();
			void randomSampling(bool rs);

		protected:
			EnsembleBuilder();
			EnsembleBuilder(const EnsembleBuilder& eb);
			virtual EnsembleBuilder& operator=(const EnsembleBuilder& eb);

			bool isValid();

			DataTools::DataManager* theDataManager;
			MultiLayerPerceptron::Trainer* theTrainer;
			DataTools::DataSet* theData;
			//bool theRandomSampling;

		private:
	};
}
#endif
