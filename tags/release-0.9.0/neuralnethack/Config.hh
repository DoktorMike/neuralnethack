/*$Id: Config.hh 1627 2007-05-08 16:40:20Z michael $*/

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


#ifndef __Config_hh__
#define __Config_hh__

#include <string>
#include <vector>
#include <map>
#include <ostream>

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
			void print(std::ostream& os);

			const std::string& suffix() const {return theSuffix;}
			void suffix(const std::string& suffix){this->theSuffix = suffix;}

			const std::string& fileName() const {return theFileName; }
			void fileName(const std::string& fileName) {this->theFileName = fileName; }
			
			const int idColumn() const {return theIdCol;}
			void idColumn(const int idColumn) {this->theIdCol = idColumn;}
			
			const std::vector<uint>& inputColumns() const {return theInCols;}
			void inputColumns(const std::vector<uint>& inputColumns) {this->theInCols = inputColumns;}
			
			const std::vector<uint>& outputColumns() const {return theOutCols;}
			void outputColumns(const std::vector<uint>& outputColumns) {this->theOutCols = outputColumns;}
			
			const std::vector<uint>& rowRange() const {return theRowRange;}
			void rowRange(const std::vector<uint>& rowRange) {this->theRowRange = rowRange;}

			const std::string& fileNameT() const {return theFileNameT; }
			void fileNameT(const std::string& fileName) {this->theFileNameT = fileName; }

			const int idColumnT() const {return theIdColT;}
			void idColumnT(const int idColumnT) {this->theIdColT = idColumnT;}
			
			const std::vector<uint>& inputColumnsT() const {return theInColsT;}
			void inputColumnsT(const std::vector<uint>& inputColumns) {this->theInColsT = inputColumns;}
			
			const std::vector<uint>& outputColumnsT() const {return theOutColsT;}
			void outputColumnsT(const std::vector<uint>& outputColumns) {this->theOutColsT = outputColumns;}
			
			const std::vector<uint>& rowRangeT() const {return theRowRangeT;}
			void rowRangeT(const std::vector<uint>& rowRangeT) {this->theRowRangeT = rowRangeT;}

			bool problemType() const {return theProblemType;}
			void problemType(bool problemType) {this->theProblemType = problemType;}

			uint numLayers() const {return theNumLayers;}
			void numLayers(const uint& theNumLayers) { this->theNumLayers = theNumLayers; }
			
			const std::vector<uint>& architecture() const {return theArch;}
			void architecture(const std::vector<uint>& architecture) 
			{
				this->theArch = architecture; 
				this->theNumLayers = architecture.size();
			}

			const std::vector<std::string>& actFcn() const {return theActFcn;}
			void actFcn(const std::vector<std::string>& theActFcn) {this->theActFcn = theActFcn;}

			const std::string& errFcn() const {return theErrFcn;}
			void errFcn(const std::string& theErrFcn) {this->theErrFcn = theErrFcn;}

			const std::string& minMethod() const {return theMinMethod;}
			void minMethod(const std::string& theMinMethod) {this->theMinMethod = theMinMethod;}

			const uint& maxEpochs() const {return theMaxEpochs;}
			void maxEpochs(const uint& theMaxEpochs) {this->theMaxEpochs = theMaxEpochs;}

			const uint& batchSize() const {return gdParam.theBatchSize;}
			void batchSize(const uint& theBatchSize) {this->gdParam.theBatchSize = theBatchSize;}

			const double& learningRate() const {return gdParam.theLearningRate;}
			void learningRate(const double& theLearningRate) {this->gdParam.theLearningRate = theLearningRate;}

			const double& decLearningRate() const {return gdParam.theDecLearningRate;}
			void decLearningRate(const double& theDecLearningRate) {this->gdParam.theDecLearningRate = theDecLearningRate;}

			const double& momentum() const {return gdParam.theMomentum;}
			void momentum(const double& theMomentum) {this->gdParam.theMomentum = theMomentum;}

			bool weightElimOn() const {return theWeightElimOn;}
			void weightElimOn(const bool& theWeightElimOn) {this->theWeightElimOn = theWeightElimOn;}

			double weightElimAlpha() const {return theWeightElimAlpha;}
			void weightElimAlpha(double theWeightElimAlpha) {this->theWeightElimAlpha = theWeightElimAlpha;}

			double weightElimW0() const {return theWeightElimW0;}
			void weightElimW0(double theWeightElimW0) {this->theWeightElimW0 = theWeightElimW0;}

			const std::string& ensParamDataSelection() const {return theEnsParamDataSelection;}
			void ensParamDataSelection(const std::string& theEnsParamDataSelection) {this->theEnsParamDataSelection = theEnsParamDataSelection;}

			const uint& ensParamN() const {return theEnsParamN;}
			void ensParamN(const uint& theEnsParamN) {this->theEnsParamN = theEnsParamN;}

			const uint& ensParamK() const {return theEnsParamK;}
			void ensParamK(const uint& theEnsParamK) {this->theEnsParamK = theEnsParamK;}

			const bool& ensParamSplitMode() const {return theEnsParamSplitMode;}
			void ensParamSplitMode(const bool& theEnsParamSplitMode) {this->theEnsParamSplitMode = theEnsParamSplitMode;}

			const bool& ensParamNewWeights() const {return theEnsParamNewWeights;}
			void ensParamNewWeights(const bool& theEnsParamNewWeights) {this->theEnsParamNewWeights = theEnsParamNewWeights;}

			const std::string& msParamDataSelection() const {return theMsParamDataSelection;}
			void msParamDataSelection(const std::string& theMsParamDataSelection) {this->theMsParamDataSelection = theMsParamDataSelection;}

			const uint& msParamN() const {return theMsParamN;}
			void msParamN(const uint& theMsParamN) {this->theMsParamN = theMsParamN;}

			const uint& msParamK() const {return theMsParamK;}
			void msParamK(const uint& theMsParamK) {this->theMsParamK = theMsParamK;}

			const bool& msParamSplitMode() const {return theMsParamSplitMode;}
			void msParamSplitMode(const bool& theMsParamSplitMode) {this->theMsParamSplitMode = theMsParamSplitMode;}

			const double& msParamNumTrainingData() const {return theMsParamNumTrainingData;}
			void msParamNumTrainingData(const double& theMsParamNumTrainingData) {this->theMsParamNumTrainingData = theMsParamNumTrainingData;}

			const uint& msgParamN() const {return theMsgParamN;}
			void msgParamN(const uint& theMsgParamN) {this->theMsgParamN = theMsgParamN;}

			const uint& msgParamK() const {return theMsgParamK;}
			void msgParamK(const uint& theMsgParamK) {this->theMsgParamK = theMsgParamK;}

			const bool& msgParamSplitMode() const {return theMsgParamSplitMode;}
			void msgParamSplitMode(const bool& theMsgParamSplitMode) {this->theMsgParamSplitMode = theMsgParamSplitMode;}

			const double& msgParamNumTrainingData() const {return theMsgParamNumTrainingData;}
			void msgParamNumTrainingData(const double& theMsgParamNumTrainingData) {this->theMsgParamNumTrainingData = theMsgParamNumTrainingData;}

			std::map<std::string, std::vector<double> >& vary() {return theVary;}

			const bool& saveSession() const {return theSaveSession;}
			void saveSession(const bool& theSaveSession) {this->theSaveSession = theSaveSession;}

			const uint& info() const {return theInfo;}
			void info(const uint& theInfo) {this->theInfo = theInfo;}

			const bool& saveOutputList() const {return theSaveOutputList;}
			void saveOutputList(const bool& theSaveOutputList) {this->theSaveOutputList = theSaveOutputList;}

			const uint& seed() const {return theSeed;}
			void seed(const uint& theSeed) {this->theSeed = theSeed;}

			const std::string& normalization() const {return theNormalization;}
			void normalization(const std::string& theNormalization) {this->theNormalization = theNormalization;}

		private:
			/**The suffix to use for all produced files. */
			std::string theSuffix;
			/**The file where the training data is located. */
			std::string theFileName;
			/**The id column. */
			int theIdCol;
			/**The input columns. */
			std::vector<uint> theInCols;
			/**The output columns. */
			std::vector<uint> theOutCols;
			/**The rows to use for the training file. */
			std::vector<uint> theRowRange;
			/**The file where the test data is located. */
			std::string theFileNameT;
			/**The id column for the test data. */
			int theIdColT;
			/**The input columns for the test file. */
			std::vector<uint> theInColsT;
			/**The output columns for the test file. */
			std::vector<uint> theOutColsT;
			/**The rows to use for the test file. */
			std::vector<uint> theRowRangeT;
			/**The problem type. false means class and true means
			 * regression.
			 */
			bool theProblemType;
			/**The number of layers. */
			uint theNumLayers;
			/**The architecture. */
			std::vector<uint> theArch;
			/**The activation functions. */
			std::vector<std::string> theActFcn;
			/**The error function. */
			std::string theErrFcn;
			/**The minimisation algorithm. */
			std::string theMinMethod;
			/**The maximum number of epochs to train. */
			uint theMaxEpochs;
			struct gdParam_t {
				/**The number of trainingsamples to use per epoch. */
				uint theBatchSize;
				/**The rate at which to train the MLP. */
				double theLearningRate;
				/**The decrease of learning rate. */
				double theDecLearningRate;
				/**The momentum term used in "poor mans conjugate gradient". */
				double theMomentum;
			}gdParam;
			/**Controls whether to use weight elimination or not. */
			bool theWeightElimOn;
			/**The importance of the weight elimination term. */
			double theWeightElimAlpha;
			/**Scaling factor typically set to unity. */
			double theWeightElimW0;
			/**The method for selecting the data. cs, bagg, none */
			std::string theEnsParamDataSelection;
			/**The number of independent runs. */
			uint theEnsParamN;
			/**The number of parts to split the dataset into. */
			uint theEnsParamK;
			/**Randomised split mode for ensemble building. True means random,
			 * false means serial.
			 */
			bool theEnsParamSplitMode;
			/**Toggle wheather to use different weights for each mlp. */
			bool theEnsParamNewWeights;
			/**The method for selecting the data. cv, boot, none */
			std::string theMsParamDataSelection;
			/**Number of independant cross validation runs for model selection. */
			uint theMsParamN;
			/**The number of parts to split the dataset into. */
			uint theMsParamK;
			/**Randomised split mode for model selection. True means random,
			 * false means serial.
			 */
			bool theMsParamSplitMode;
			/**Number of training data for model selection if msParamK==1. */
			double theMsParamNumTrainingData;
			/**Number of independant cross testing runs for model selection. */
			uint theMsgParamN;
			/**Number of subsets (K) in K-fold cross testing for model selection. */
			uint theMsgParamK;
			/**Randomised split mode for cross testing. True means random,
			 * false means serial.
			 */
			bool theMsgParamSplitMode;
			/**Number of training data for cross testing if msParamK==1. */
			double theMsgParamNumTrainingData;
			/**The parameters to vary. The key is the name of the variable to
			 * vary. The vector contains the start, stop and step value.
			 */
			std::map<std::string, std::vector<double> > theVary;
			/**Toggle wheather to save all networks in the session. */
			bool theSaveSession;
			/**The amount of information to print. */
			uint theInfo;
			/**Toggle whether to save all outputs from the model on each data
			 * point.
			 */
			bool theSaveOutputList;
			/**The seed to pass to srand. */
			uint theSeed;
			/**The normalization to perform on the data. */
			std::string theNormalization;

	};
}
#endif
