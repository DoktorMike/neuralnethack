#ifndef __Parser_hh__
#define __Parser_hh__

#include "datatools/DataSet.hh"
#include "Config.hh"

using DataTools::DataSet;
using DataTools::Pattern;

namespace NetHack
{
    ///This namespace encloses the file parsing.
    ///It contains parsing methods for configuration and data files.
    namespace Parser
    {
	///Parses a data file for the training data.
	///\deprecated no column selection available.
	///\param in the stream containing the data.
	///\param nInput the number of inputs to read.
	///\param nOutput the number of outputs to read.
	///\param dataSet the class to store the data in.
	void readDataFile(ifstream& in, const int nInput, 
		const int nOutput, DataSet& dataSet);
	///Parses a data file for the training data.
	///\param in the stream containing the data.
	///\param inCols the columns representing the input data.
	///\param outCols the columns representing the output data.
	///\param dataSet the class to store the data in.
	void readDataFile(ifstream& in, vector<uint> inCols,
		vector<uint> outCols, DataSet& dataSet);
	///Reads and parses the configuration file.
	///\param in the stream containing the configuration.
	///\param config the class to store the configuration in.
	void readConfigurationFile(ifstream& in, Config& config);
    }
}
#endif
