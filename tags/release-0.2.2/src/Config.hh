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
			/**Basic destructor. */
			virtual ~Config();

			/**Print out every bit of information contained in this class. */
			void print();

			/**The file where the data is located. */
			string fileName;
			/**The input columns. */
			vector<uint> inCols;
			/**The output columns. */
			vector<uint> outCols;
			/**The number of layers. */
			uint nLayers;
			/**The architecture. */
			vector<uint> arch;
			/**The activation functions. */
			vector<string> actFcn;
			/**The error function. */
			string errFcn;
			/**The minimisation algorithm. */
			string minMethod;
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
			/**Number of independant cross validation runs for model selection. */
			uint msParamN;
			/**Number of subsets (K) in K-fold cross validation for model selection. */
			uint msParamK;
			/**Randomised split mode for model selection. True means random,
			 * false means serial.
			 */
			bool msParamSplitMode;
			/**Number of training data for model selection if msParamK==1. */
			uint msParamNumTrainingData;
			/**Controls whether to use weight elimination or not. */
			bool weightElimOn;
			/**The importance of the weight elimination term. */
			double weightElimAlpha;
			/**Scaling factor typically set to unity. */
			double weightElimW0;


		private:
			/**Copy constructor.
			 * \param c the Config object to copy.
			 */
			Config(const Config& c);

			/**Assignment operator.
			 * \param c the Config object to assign from.
			 */
			Config& operator=(const Config& c);
	};
}
#endif
