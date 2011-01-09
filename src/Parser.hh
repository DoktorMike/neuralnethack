#ifndef __Parser_hh__
#define __Parser_hh__

#include "datatools/CoreDataSet.hh"
#include "Config.hh"

#include <iostream>
#include <vector>
#include <string>

namespace NeuralNetHack
{
	/**This class encloses the file parsing.
	 * It contains parsing methods for configuration and data files.
	 */
	class Parser
	{
		public:
			/**Parses a data file for the training data.
			 * \deprecated no column selection available.
			 * \param in the stream containing the data.
			 * \param nInput the number of inputs to read.
			 * \param nOutput the number of outputs to read.
			 * \param dataSet the class to store the data in.
			 */
			static void readDataFile(std::ifstream& in, const int nInput, 
					const int nOutput, DataTools::CoreDataSet& dataSet);
			/**Parses a data file for the training data.
			 * \param in the stream containing the data.
			 * \param inCols the columns representing the input data.
			 * \param outCols the columns representing the output data.
			 * \param dataSet the class to store the data in.
			 */
			static void readDataFile(std::ifstream& in, std::vector<uint> inCols,
					std::vector<uint> outCols, DataTools::CoreDataSet& dataSet);
			/**Reads and parses the configuration file.
			 * \param in the stream containing the configuration.
			 * \param config the class to store the configuration in.
			 */
			static void readConfigurationFile(std::ifstream& in, Config& config);

		private:
			/**Investigates the stream for errors.
			 * If an error is found abort is called.
			 * \param in the input stream to check.
			 */
			static void checkStream(std::ifstream& in);

			/**Throws away every whitespace.
			 * \param in the input stream to read from.
			 */
			static void whitespace(std::ifstream& in);

			/**Throws away everything after '%' on a line.
			 * \param in the input stream to read from.
			 */
			static void comment(std::ifstream& in);

			/**Parses the filename part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static std::string parseFileName(std::ifstream& in);

			/**Parses the data columns part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static std::vector<uint> parseCol(std::ifstream& in);

			/**Parses the PType part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static bool parsePType(std::ifstream& in);

			/**Parses the number of layers part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static uint parseNLay(std::ifstream& in);

			/**Parses the MLP architecture part of the configuration file.
			 * \param in the input stream to read from.
			 * \param nLayers the number of layers in this architecture.
			 */
			static std::vector<uint> parseSize(std::ifstream& in, uint nLayers);

			/**Parses the Activation function part of the configuration file.
			 * \param in the input stream to read from.
			 * \param nLayers the number of layers in this architecture.
			 */
			static std::vector<std::string> parseActFcn(std::ifstream& in, uint nLayers);

			/**Parses the Error function part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static std::string parseErrFcn(std::ifstream& in);

			/**Parses the learning algorithm part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static std::string parseMinMethod(std::ifstream& in);

			/**Parses the epoch part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static uint parseMaxEpochs(std::ifstream& in);

			/**Parses the Gradient descent part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseGDParam(std::ifstream& in, Config& config);

			/**Parses the weight elimination part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseWeightElim(std::ifstream& in, Config& config);

			/**Parses the ensemble part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseEnsParam(std::ifstream& in, Config& config);

			/**Parses the cross validation part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseMSParam(std::ifstream& in, Config& config);

			/**Parses the cross testing part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseMSGParam(std::ifstream& in, Config& config);

			/**Reads one row of double from the stream.
			 * \param in the input stream to read from.
			 */
			static std::vector<double> readRow(std::ifstream& in);

			/**Reads and discards tab signs and space.
			 * \param in the input stream to read from.
			 */
			static void tabspace(std::ifstream& in);
	};
}
#endif
