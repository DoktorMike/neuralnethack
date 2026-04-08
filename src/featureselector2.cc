#include "neuralnethack/Config.hh"
#include "neuralnethack/parser/Parser.hh"
#include "neuralnethack/evaltools/Roc.hh"
#include "neuralnethack/NeuralNetHack.hh"
#include "neuralnethack/FeatureSelector.hh"
#include "neuralnethack/evaltools/EvalTools.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <functional>
#include <cmath>

using NeuralNetHack::Config;
using NeuralNetHack::FeatureSelector;
using NeuralNetHack::Parser;
using namespace EvalTools;

using std::abs;
using std::advance;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::make_pair;
using std::map;
using std::ofstream;
using std::ostream_iterator;
using std::ostringstream;
using std::pair;
using std::string;
using std::vector;

template <class T> struct mapValueIndex {
	mapValueIndex() : index(0) {}
	void operator()(T& x) { sals.insert(make_pair(abs(x), ++index)); }
	map<T, uint> sals;
	uint index;
};

void parseConf(string fname, Config& config) {
	ifstream confStream;

	cout << "Parsing and storing Configuration." << endl << endl;
	confStream.open(fname.c_str(), ios::in);
	if (!confStream) {
		cerr << "Could not open configuration file: " << fname << endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
}

int main(int argc, char* argv[]) {
	Config config;
	uint minF = 1, maxF = 10, maxR = 1;
	if (argc == 5) {
		minF = atoi(argv[1]);
		maxF = atoi(argv[2]);
		maxR = atoi(argv[3]);
		parseConf(argv[4], config);
	} else {
		cerr << "Usage: " << endl
		     << argv[0] << " MinFeatures MaxFeatures MaxRemove configfile" << endl;
		exit(0);
	}

	srand48(config.seed() == 0 ? time(0)
	                           : config.seed()); // This is the ONLY place one may set the seed!

	FeatureSelector fs(minF, maxF, maxR);
	fs.run(config, ErrorMeasures::auc);

	return 0;
}
