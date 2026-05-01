#ifndef __Parser_hh__
#define __Parser_hh__

#include "../datatools/CoreDataSet.hh"
#include "../Config.hh"

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

namespace NeuralNetHack {
/**Data-file parsing and a thin wrapper around TomlParser for configs.
 *
 * Configuration files have moved to TOML; readConfigurationFile here just
 * delegates to TomlParser so existing callers keep working.
 */
class Parser {
  public:
	/**Parses a tabular data file into a CoreDataSet.
	 * \param in the stream containing the data.
	 * \param idCol the column representing the identity of the Pattern.
	 * \param inCols the columns representing the input data (1-indexed).
	 * \param outCols the columns representing the output data.
	 * \param rowRange the rows to use (1-indexed); {0} means all rows.
	 * \param dataSet the class to store the data in.
	 */
	static void readDataFile(std::istream& in, const int idCol, std::vector<uint> inCols,
	                         std::vector<uint> outCols, std::vector<uint> rowRange,
	                         DataTools::CoreDataSet& dataSet);

	/**Reads a TOML configuration stream into a Config. */
	static void readConfigurationFile(std::istream& in, Config& config);

  private:
	static void checkStream(std::istream& in);

	/**Insert the elements found in row at location x into vec. */
	struct selectInserter {
		selectInserter(std::vector<std::string>& r) : row(r) { vec.reserve(r.size()); }
		void operator()(uint x) {
			char* end = new char;
			vec.push_back(strtod(row[x - 1].c_str(), &end));
		}
		std::vector<std::string>& row;
		std::vector<double> vec;
	};
};
} // namespace NeuralNetHack
#endif
