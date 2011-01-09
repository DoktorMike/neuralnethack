/*$Id: featureselector.cc 3344 2009-03-13 00:04:02Z michael $*/

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


#include "neuralnethack/parser/Parser.hh"
#include "neuralnethack/datatools/Normaliser.hh"
#include "neuralnethack/datatools/DataManager.hh"
#include "neuralnethack/datatools/CoreDataSet.hh"
#include "neuralnethack/evaltools/Roc.hh"
#include "neuralnethack/Factory.hh"
#include "neuralnethack/PrintUtils.hh"
#include "neuralnethack/NeuralNetHack.hh"
#include "neuralnethack/ModelEstimator.hh"
#include "neuralnethack/ModelSelector.hh"
#include "neuralnethack/Saliency.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <functional>
#include <cmath>
#include <algorithm>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace EvalTools;
using std::vector;
using std::pair;
using std::map;
using std::string;
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::ostringstream;
using std::ostream_iterator;
using std::make_pair;
using std::unary_function;
using std::abs;
using std::advance;

void saveSaliencies(vector<Session>& sessions, Config& config)
{
	if(config.saveSession() == false) return;
	ofstream os;
	string fname = "saliencies."+config.suffix()+".txt";
	os.open(fname.c_str(), ios::out);
	if(!os){
		cerr<<"Could not open output file: "<<fname<<endl;
		abort();
	}
	PrintUtils::printSaliencies(os, sessions, config);
	os.close();
}

pair<Ensemble, double> validateFeatures(DataSet& trn, Config& config)
{
	Ensemble ensemble;
	pair<Config, double> best;

	if(!config.vary().empty()){ //Model selection, training and testing
		ModelSelector ms;
		cout<<"Finding the best model"<<endl;
		best = ms.findBestModel(trn, config);
		cout<<"Building the ensemble from best model"<<endl;
		EnsembleBuilder* eb = Factory::createEnsembleBuilder(best.first, trn);
		Ensemble* tmp = eb->buildEnsemble();
		ensemble = *tmp;
		delete eb;
		delete tmp;
	}
	return make_pair<Ensemble, double>(ensemble, best.second);
}

template<class T> struct mapValueIndex : public unary_function<T, void>
{
	mapValueIndex():index(0){}
	void operator() (T& x){	sals.insert(make_pair(abs(x), ++index)); }
	map<T, uint> sals;
	uint index;
};

void updateColumns(vector<uint>& inputs, vector<uint>& indices)
{
	sort(indices.begin(), indices.end());
	vector<uint> tmp;
	for(uint i=1; i<=inputs.size(); ++i)
		if(find(indices.begin(), indices.end(), i) == indices.end()) tmp.push_back(inputs[i-1]);
	inputs = tmp;
}

Config& excludeFeatures(Config& config, vector<double>& saliencies, uint n)
{
	//Build a map from the saliencies
	mapValueIndex<double> result = for_each(saliencies.begin(), saliencies.end(), mapValueIndex<double>());
	map<double, uint> sals = result.sals;
	//Find the indices to remove
	uint cntr = 1;
	vector<uint> indices;
	for(map<double, uint>::iterator it = sals.begin(); it != sals.end(); ++it){
		//cout<<it->second<<" "<<it->first<<endl;
		if(cntr++ <= n) indices.push_back(it->second);
	}
	//Update the input columns of the config.
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
	oss<<config.suffix()<<"."<<config.architecture().front();
	config.suffix(oss.str());

	return config;
}

void parseConf(string fname, Config& config)
{
	ifstream confStream;

	cout<<"Parsing and storing Configuration."<<endl<<endl;
	confStream.open(fname.c_str(), ios::in);
	if(!confStream){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
}


void parseData(Config& config, DataSet& trnData, DataSet& valData, DataSet& tstData)
{
	ifstream trnStream;
	ifstream tstStream;
	CoreDataSet* trnCoreData = new CoreDataSet();
	CoreDataSet* tstCoreData = new CoreDataSet();

	trnStream.open(config.fileName().c_str(), ios::in);
	if(!trnStream){
		cerr<<"Could not open data file: "<<config.fileName()<<endl;
		abort();
	}
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(), 
			config.outputColumns(), config.rowRange(), *trnCoreData);
	trnStream.close();
	trnData.coreDataSet(*trnCoreData);

	tstStream.open(config.fileNameT().c_str(), ios::in);
	if(!tstStream){
		cerr<<"Could not open data file: "<<config.fileNameT()<<endl;
		abort();
	}
	Parser::readDataFile(tstStream, config.idColumnT(), config.inputColumnsT(), 
			config.outputColumnsT(), config.rowRangeT(), *tstCoreData);
	tstStream.close();
	tstData.coreDataSet(*tstCoreData);

	//Split trnData into trn and val
	DataManager dm;
	pair<DataSet, DataSet>* data = dm.split(trnData, 0.75);
	trnData = data->first;
	valData = data->second;
	delete data;
}

void featureSelect(Config& config)
{
	Config conf = config, origConf = config;
	vector<double> saliencies;
	pair<Ensemble, double> result;
	uint n = 10;

	string fname = "backward_fs." + config.suffix() + ".txt";
	ofstream os(fname.c_str());
	while(conf.inputColumns().size() > n){
		//Parse data based on input columns.
		DataSet trnData, valData, tstData;
		Normaliser norm;
		parseData(conf, trnData, valData, tstData);
		if(config.normalization() == "Z"){
			norm.calcAndNormalise(trnData, true); 
			norm.normalise(tstData);
		}
		//Check result
		result = validateFeatures(trnData, conf);
		os<<"Inputs: ";
		copy(conf.inputColumns().begin(), conf.inputColumns().end(), ostream_iterator<uint>(os, ","));
		os<<endl<<"AUC: "<<result.second<<endl;
		saliencies = Saliency::saliencyMagnitude(result.first, valData, false);
		conf.suffix(origConf.suffix());
		conf = excludeFeatures(conf, saliencies, n);
		//Delete the core data
		delete &(trnData.coreDataSet());
		delete &(tstData.coreDataSet());
	}
	os.close();
}

void parseCmdLine(Config& config, int argc, char* argv[])
{
	string filename(argv[1]);
	parseConf(filename, config);
	//config.print(cout);
}

int main(int argc, char* argv[])
{
	Config config;
	if(argc == 2){
		parseConf(argv[1], config);
	}else if(argc == 1){
		cout<<"Usage: "<<endl<<"neuralnethack configfile"<<endl;
		exit(0);
	}else{
		parseCmdLine(config, argc, argv);
	}

	srand48(config.seed() == 0 ? time(0) : config.seed()); //This is the ONLY place one may set the seed!

	featureSelect(config);

	return 0;
}
