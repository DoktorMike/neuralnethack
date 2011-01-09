#ifndef __Config_hh__
#define __Config_hh__

#include "NeuralNetHack.hh"
#include <string>

namespace NeuralNetHack
{
	/**A class representing all the needed configuration. Every outside 
	 * configuration is to be put in this file via the Parser.
	 */
	class Config
	{
		public:
			/**Basic constructor. */
			Config();

			/**Copy constructor.
			 * \param c the Config object to copy.
			 */
			Config(const Config& c);

			/**Basic destructor. */
			virtual ~Config();

			/**Assignment operator.
			 * \param c the Config object to assign from.
			 */
			Config& operator=(const Config& c);

			/**Print out every bit of information contained in this class. */
			void print();

			/**The suffix to use for all produced files. */
			std::string suffix;
			/**The file where the data is located. */
			std::string fileName;
			/**The input columns. */
			std::vector<uint> inCols;
			/**The output columns. */
			std::vector<uint> outCols;
			/**The file where the test data is located. */
			std::string fileNameT;
			/**The input columns for the test file. */
			std::vector<uint> inColsT;
			/**The output columns for the test file. */
			std::vector<uint> outColsT;
			/**The problem type. false means class and true means
			 * regression.
			 */
			bool problemType;
			/**The number of layers. */
			uint nLayers;
			/**The architecture. */
			std::vector<uint> arch;
			/**The activation functions. */
			std::vector<std::string> actFcn;
			/**The error function. */
			std::string errFcn;
			/**The minimisation algorithm. */
			std::string minMethod;
			/**The maximum number of epochs to train. */
			uint maxEpochs;
			/**The number of trainingsamples to use per epoch. */
			uint batchSize;
			/**The rate at which to train the MLP. */
			double learningRate;
			/**The decrease of learning rate. */
			double decLearningRate;
			/**The momentum term used in "poor mans conjugate gradient". */
			double momentum;
			/**Controls whether to use weight elimination or not. */
			bool weightElimOn;
			/**The importance of the weight elimination term. */
			double weightElimAlpha;
			/**Scaling factor typically set to unity. */
			double weightElimW0;
			/**The method for selecting the data. cs, bagg. */
			std::string ensParamDataSelection;
			/**The number of independent runs. */
			uint ensParamN;
			/**The number of parts to split the dataset into. */
			uint ensParamK;
			/**Randomised split mode for ensemble building. True means random,
			 * false means serial.
			 */
			bool ensParamSplitMode;
			/**Toggle wheather to use different weights for each mlp. */
			bool ensParamNewWeights;
			/**The method for selecting the data. cv, boot. */
			std::string msParamDataSelection;
			/**Number of independant cross validation runs for model selection. */
			uint msParamN;
			/**The number of parts to split the dataset into. */
			uint msParamK;
			/**Randomised split mode for model selection. True means random,
			 * false means serial.
			 */
			bool msParamSplitMode;
			/**Number of training data for model selection if msParamK==1. */
			double msParamNumTrainingData;
			/**Number of independant cross testing runs for model selection. */
			uint msgParamN;
			/**Number of subsets (K) in K-fold cross testing for model selection. */
			uint msgParamK;
			/**Randomised split mode for cross testing. True means random,
			 * false means serial.
			 */
			bool msgParamSplitMode;
			/**Number of training data for cross testing if msParamK==1. */
			double msgParamNumTrainingData;
			/**Toggle wheather to save all networks in the session. */
			bool saveSession;
			/**The amount of information to print. */
			uint info;
			/**Toggle whether to save all outputs from the model on each data
			 * point.
			 */
			bool saveOutputList;
			/**The seed to pass to srand. */
			uint seed;
		


		private:
	};
}
#endif
