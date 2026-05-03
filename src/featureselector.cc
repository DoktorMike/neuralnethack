#include "neuralnethack/Random.hh"
#include "neuralnethack/parser/Parser.hh"
#include "neuralnethack/datatools/Normaliser.hh"
#include "neuralnethack/datatools/DataManager.hh"
#include "neuralnethack/datatools/CoreDataSet.hh"
#include "neuralnethack/evaltools/Roc.hh"
#include "neuralnethack/evaltools/EvalTools.hh"
#include "neuralnethack/Factory.hh"
#include "neuralnethack/PrintUtils.hh"
#include "neuralnethack/NeuralNetHack.hh"
#include "neuralnethack/ModelEstimator.hh"
#include "neuralnethack/ModelSelector.hh"
#include "neuralnethack/Saliency.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cmath>
#include <algorithm>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
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

void saveSaliencies(vector<Session>& sessions, Config& config) {
	if (config.saveSession() == false) return;
	ofstream os;
	string fname = "saliencies." + config.suffix() + ".txt";
	os.open(fname.c_str(), ios::out);
	if (!os) {
		cerr << "Could not open output file: " << fname << endl;
		abort();
	}
	PrintUtils::printSaliencies(os, sessions, config);
	os.close();
}

pair<Ensemble, double> validateFeatures(DataSet& trn, Config& config) {
	Ensemble ensemble;
	pair<Config, double> best;

	if (!config.vary().empty()) { // Model selection, training and testing
		ModelSelector ms;
		cout << "Finding the best model" << endl;
		best = ms.findBestModel(trn, config);
		cout << "Building the ensemble from best model" << endl;
		auto eb = Factory::createEnsembleBuilder(best.first, trn);
		std::unique_ptr<Ensemble> tmp(eb->buildEnsemble());
		ensemble = *tmp;
	}
	return make_pair(ensemble, best.second);
}

template <class T> struct mapValueIndex {
	mapValueIndex() : index(0) {}
	void operator()(T& x) { sals.insert(make_pair(abs(x), ++index)); }
	map<T, uint> sals;
	uint index;
};

void updateColumns(vector<uint>& inputs, vector<uint>& indices) {
	sort(indices.begin(), indices.end());
	vector<uint> tmp;
	for (uint i = 1; i <= inputs.size(); ++i)
		if (find(indices.begin(), indices.end(), i) == indices.end()) tmp.push_back(inputs[i - 1]);
	inputs = tmp;
}

Config& excludeFeatures(Config& config, vector<double>& saliencies, uint n) {
	// Build a map from the saliencies
	mapValueIndex<double> result =
	    for_each(saliencies.begin(), saliencies.end(), mapValueIndex<double>());
	map<double, uint> sals = result.sals;
	// Find the indices to remove
	uint cntr = 1;
	vector<uint> indices;
	for (map<double, uint>::iterator it = sals.begin(); it != sals.end(); ++it) {
		// cout<<it->second<<" "<<it->first<<endl;
		if (cntr++ <= n) indices.push_back(it->second);
	}
	// Update the input columns of the config.
	vector<uint> vec = config.inputColumns();
	updateColumns(vec, indices);
	config.inputColumns(vec);
	vec = config.inputColumnsT();
	updateColumns(vec, indices);
	config.inputColumnsT(vec);
	vec = config.architecture();
	vec[0] -= n;
	config.architecture(vec);
	ostringstream oss;
	oss << config.suffix() << "." << config.architecture().front();
	config.suffix(oss.str());

	return config;
}

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

void parseData(Config& config, DataSet& trnData, DataSet& valData, DataSet& tstData) {
	ifstream trnStream;
	ifstream tstStream;
	auto trnCoreData = std::make_shared<CoreDataSet>();
	auto tstCoreData = std::make_shared<CoreDataSet>();

	trnStream.open(config.fileName().c_str(), ios::in);
	if (!trnStream) {
		cerr << "Could not open data file: " << config.fileName() << endl;
		abort();
	}
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(),
	                     config.outputColumns(), config.rowRange(), *trnCoreData);
	trnStream.close();
	trnData.coreDataSet(trnCoreData);

	tstStream.open(config.fileNameT().c_str(), ios::in);
	if (!tstStream) {
		cerr << "Could not open data file: " << config.fileNameT() << endl;
		abort();
	}
	Parser::readDataFile(tstStream, config.idColumnT(), config.inputColumnsT(),
	                     config.outputColumnsT(), config.rowRangeT(), *tstCoreData);
	tstStream.close();
	tstData.coreDataSet(tstCoreData);

	DataManager dm;
	auto data = dm.split(trnData, 0.75);
	trnData = std::move(data.first);
	valData = std::move(data.second);
}

void featureSelect(Config& config) {
	Config conf = config, origConf = config;
	vector<double> saliencies;
	pair<Ensemble, double> result;
	uint n = 10;

	string fname = "backward_fs." + config.suffix() + ".txt";
	ofstream os(fname.c_str());
	while (conf.inputColumns().size() > n) {
		// Parse data based on input columns.
		DataSet trnData, valData, tstData;
		Normaliser norm;
		parseData(conf, trnData, valData, tstData);
		if (config.normalization() == "Z") {
			norm.calcAndNormalise(trnData, true);
			norm.normalise(tstData);
		}
		// Check result
		result = validateFeatures(trnData, conf);
		os << "Inputs: ";
		copy(conf.inputColumns().begin(), conf.inputColumns().end(),
		     ostream_iterator<uint>(os, ","));
		os << endl << "AUC: " << result.second << endl;
		saliencies = Saliency::saliencyMagnitude(result.first, valData, false);
		conf.suffix(origConf.suffix());
		conf = excludeFeatures(conf, saliencies, n);
	}
	os.close();
}

void parseCmdLine(Config& config, int argc, char* argv[]) {
	string filename(argv[1]);
	parseConf(filename, config);
	// config.print(cout);
}

int main(int argc, char* argv[]) {
	Config config;
	if (argc == 2) {
		parseConf(argv[1], config);
	} else if (argc == 1) {
		cout << "Usage: " << endl << "neuralnethack configfile" << endl;
		exit(0);
	} else {
		parseCmdLine(config, argc, argv);
	}

	nnh::rand::seed(config.seed() == 0
	                    ? time(0)
	                    : config.seed()); // This is the ONLY place one may set the seed!

	featureSelect(config);

	return 0;
}
