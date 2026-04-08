#ifndef __Adam_hh__
#define __Adam_hh__

#include "Trainer.hh"
#include <vector>

namespace MultiLayerPerceptron
{
	/**Adam/AdamW optimizer.
	 * Uses per-weight adaptive learning rates with first and second moment
	 * estimates. When weightDecay > 0, implements decoupled weight decay (AdamW).
	 */
	class Adam : public Trainer
	{
		public:
			Adam(Mlp& mlp, DataTools::DataSet& data, Error& error,
				 double te, uint bs,
				 double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
				 double eps = 1e-8, double weightDecay = 0.0);
			~Adam();

			void train(std::ostream& os) override;
			std::unique_ptr<Trainer> clone() const override;

			double learningRate() const;
			void learningRate(double lr);

		private:
			Adam(const Adam& a);
			Adam& operator=(const Adam& a);

			double trainEpoch(DataTools::DataSet& dset);
			bool buildBlock(DataTools::DataSet& blockData, uint& cntr) const;

			double theLearningRate;
			double theBeta1;
			double theBeta2;
			double theEpsilon;
			double theWeightDecay;

			std::vector<double> theM;  // first moment
			std::vector<double> theV;  // second moment
			uint theTimestep;
	};
}
#endif
