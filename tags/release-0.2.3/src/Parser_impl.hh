#ifndef __Parser_impl_hh__
#define __Parser_impl_hh__

#include "Parser.hh"

namespace NeuralNetHack
{
	namespace Parser
	{
		/**Investigates the stream for errors.
		 * If an error is found abort is called.
		 * \param in the input stream to check.
		 */
		void checkStream(ifstream& in);
		
		/**Throws away every whitespace.
		 * \param in the input stream to read from.
		 */
		void whitespace(ifstream& in);
		
		/**Throws away everything after '%' on a line.
		 * \param in the input stream to read from.
		 */
		void comment(ifstream& in);
		
		/**Parses the filename part of the configuration file.
		 * \param in the input stream to read from.
		 */
		string parseFileName(ifstream& in);
		
		/**Parses the data columns part of the configuration file.
		 * \param in the input stream to read from.
		 */
		vector<uint> parseCol(ifstream& in);
		
		/**Parses the PType part of the configuration file.
		 * \param in the input stream to read from.
		 */
		string parsePType(ifstream& in);
		
		/**Parses the number of layers part of the configuration file.
		 * \param in the input stream to read from.
		 */
		uint parseNLay(ifstream& in);
		
		/**Parses the MLP architecture part of the configuration file.
		 * \param in the input stream to read from.
		 * \param nLayers the number of layers in this architecture.
		 */
		vector<uint> parseSize(ifstream& in, uint nLayers);
		
		/**Parses the Activation function part of the configuration file.
		 * \param in the input stream to read from.
		 * \param nLayers the number of layers in this architecture.
		 */
		vector<string> parseActFcn(ifstream& in, uint nLayers);

		/**Parses the Error function part of the configuration file.
		 * \param in the input stream to read from.
		 */
		string parseErrFcn(ifstream& in);

		/**Parses the learning algorithm part of the configuration file.
		 * \param in the input stream to read from.
		 */
		string parseMinMethod(ifstream& in);
		
		/**Parses the epoch part of the configuration file.
		 * \param in the input stream to read from.
		 */
		uint parseMaxEpochs(ifstream& in);
		
		/**Parses the Gradient descent part of the configuration file.
		 * \param in the input stream to read from.
		 * \param config the place to store configuration.
		 */
		void parseGDParam(ifstream& in, Config& config);
		
		/**Parses the model selection part of the configuration file.
		 * \param in the input stream to read from.
		 * \param config the place to store configuration.
		 */
		void parseMSParam(ifstream& in, Config& config);
		
		/**Parses the weight elimination part of the configuration file.
		 * \param in the input stream to read from.
		 * \param config the place to store configuration.
		 */
		void parseWeightElim(ifstream& in, Config& config);
		
		/**Reads one row of double from the stream.
		 * \param in the input stream to read from.
		 */
		vector<double> readRow(ifstream& in);
		
		/**Reads and discards tab signs and space.
		 * \param in the input stream to read from.
		 */
		void tabspace(ifstream& in);
	}
}
#endif
