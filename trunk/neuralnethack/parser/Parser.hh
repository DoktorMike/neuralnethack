/*$Id: Parser.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __Parser_hh__
#define __Parser_hh__

#include "../datatools/CoreDataSet.hh"
#include "../Config.hh"

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

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
			static void readDataFile(std::istream& in, const int nInput, 
					const int nOutput, DataTools::CoreDataSet& dataSet);

			/**Parses a data file for the training data.
			 * \param in the stream containing the data.
			 * \param idCol the column representing the identity of the Pattern.
			 * \param inCols the columns representing the input data.
			 * \param outCols the columns representing the output data.
			 * \param dataSet the class to store the data in.
			 */
			static void readDataFile(std::istream& in, const int idCol, 
					std::vector<uint> inCols, std::vector<uint> outCols,
					std::vector<uint> rowRange,	DataTools::CoreDataSet& dataSet);

			/**Reads and parses the configuration file.
			 * \param in the stream containing the configuration.
			 * \param config the class to store the configuration in.
			 */
			static void readConfigurationFile(std::istream& in, Config& config);

			/**Parses a line looking like "2-4,6,7-9" and returns a vector.
			 * The vector will contain "2,3,4,6,7,8,9".
			 * \param in the input stream to read from.
			 */
			static std::vector<uint> parseColumns(std::istream& in);

		private:
			/**Investigates the stream for errors.
			 * If an error is found abort is called.
			 * \param in the input stream to check.
			 */
			static void checkStream(std::istream& in);

			/**Throws away every whitespace.
			 * \param in the input stream to read from.
			 */
			static void whitespace(std::istream& in);

			/**Throws away everything after '%' on a line.
			 * \param in the input stream to read from.
			 */
			static void comment(std::istream& in);

			/**Parses a string in the configuration file.
			 * It breakes on newline and whitespace characters.
			 * \param in the input stream to read from.
			 */
			static std::string parseString(std::istream& in);

			/**Parses the data columns part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static std::vector<uint> parseCol(std::istream& in);

			/**Parses the PType part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static bool parsePType(std::istream& in);

			/**Parses the number of layers part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static uint parseNLay(std::istream& in);

			/**Parses the MLP architecture part of the configuration file.
			 * \param in the input stream to read from.
			 * \param nLayers the number of layers in this architecture.
			 */
			static std::vector<uint> parseSize(std::istream& in, uint nLayers);

			/**Parses the Activation function part of the configuration file.
			 * \param in the input stream to read from.
			 * \param nLayers the number of layers in this architecture.
			 */
			static void parseActFcn(std::istream& in, Config& config);

			/**Parses the Error function part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static void parseErrFcn(std::istream& in, Config& config);

			/**Parses the learning algorithm part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static void parseMinMethod(std::istream& in, Config& config);

			/**Parses the epoch part of the configuration file.
			 * \param in the input stream to read from.
			 */
			static void parseMaxEpochs(std::istream& in, Config& config);

			/**Parses the Gradient descent part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseGDParam(std::istream& in, Config& config);

			/**Parses the weight elimination part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseWeightElim(std::istream& in, Config& config);

			/**Parses the ensemble part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseEnsParam(std::istream& in, Config& config);

			/**Parses the cross validation part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseMSParam(std::istream& in, Config& config);

			/**Parses the cross testing part of the configuration file.
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseMSGParam(std::istream& in, Config& config);

			/**Parse the Vary part of the configuration file. 
			 * \param in the input stream to read from.
			 * \param config the place to store configuration.
			 */
			static void parseVary(std::istream& in, Config& config);

			/**Reads one row of double from the stream.
			 * \param in the input stream to read from.
			 */
			static std::vector<double> readRow(std::istream& in);

			/**Reads and discards tab signs and space.
			 * \param in the input stream to read from.
			 */
			static void tabspace(std::istream& in);

			/**Insert the elements found in row at location x into vec. */
			struct selectInserter{
				selectInserter(std::vector<std::string>& r):row(r) {	vec.reserve(r.size()); }
				void operator()(uint x){ char* end = new char; vec.push_back(strtod(row[x-1].c_str(), &end)); }
				std::vector<std::string>& row;
				std::vector<double> vec;
			};
	};
}
#endif
