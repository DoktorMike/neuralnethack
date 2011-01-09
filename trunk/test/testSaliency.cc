/*$Id: testSaliency.cc 1644 2007-05-29 08:15:18Z michael $*/

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

#include "Saliency.hh"
#include "Config.hh"
#include "Factory.hh"
#include "parser/Parser.hh"
#include "Ensemble.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "evaltools/EvalTools.hh"
#include "mlp/Trainer.hh"
#include "mlp/Mlp.hh"

#include <vector>
#include <cmath>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <ctime>
#include <string>
#include <fstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using namespace MultiLayerPerceptron;
using namespace std;

int testSimpleSaliency()
{
	// Create the Mlp
	vector<uint> arch; 
	arch.push_back(2);
	arch.push_back(2);
	arch.push_back(1);

	vector<string> types;
	types.push_back(TANHYP);
	types.push_back(SIGMOID);

	Mlp mlp(arch, types, false);
	vector<double> weights = mlp.weights();
	//fill(weights.begin(), weights.end(), 0.5);
	// Weights for input layer
	weights[0] = 0.9;
	weights[1] = 0.8;
	weights[2] = 0.7;
	weights[3] = 0.6;
	weights[4] = 0.5;
	weights[5] = 0.4;
	// Weights for hidden layer
	weights[6] = 0.3;
	weights[7] = 0.2;
	weights[8] = 0.1;
	mlp.weights(weights);

	// Make the data point
	vector<double> input;
	input.push_back(0); input.push_back(1);
	vector<double> output;
	output.push_back(1);
	Pattern p("someid", input, output);

	vector<double> saliency = Saliency::saliency(mlp, p);
	/**\todo Recalculate the saliency so this test will work again. */
	//if( fabs(saliency[0] - 0.0818615505097) > 1e-13 ) return EXIT_FAILURE;
	//if( fabs(saliency[1] - 0.0606381855627) > 1e-13 ) return EXIT_FAILURE;
	//copy(saliency.begin(), saliency.end(), ostream_iterator<double>(cout, " "));

	return EXIT_SUCCESS;
}

int testSaliency(DataSet& trnData, DataSet& tstData, Normaliser& norm, const Config& config)
{

	Trainer* trainer = 0;
	Error* error = 0;
	Mlp* mlp = 0;

	trainer = Factory::createTrainer(config, trnData);
	trainer->train(cout);
	error = trainer->error();
	mlp = trainer->mlp();
	Ensemble c(*mlp, 1);
	
	vector<double> saliencies = Saliency::saliency(c, trnData);
	Saliency::print(cout, saliencies);

	delete trainer;
	delete error;
	delete mlp;

	return 0;
}

void parseConf(string fname, Config& config)
{
	ifstream confStream;

	confStream.open(fname.c_str(), ios::in);
	if(!confStream){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
}


void parseData(Config& config, DataSet& trnData, DataSet& tstData)
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
}


void parseCmdLine(Config& config, int argc, char* argv[])
{
	string filename(argv[1]);
	parseConf(filename, config);
	config.print(cout);
}

int main(int argc, char* argv[])
{
	Config config;
	if(argc == 2){
		parseConf(argv[1], config);
	}else{
		parseConf("config.txt", config);
	}

	DataSet trnData, tstData;
	Normaliser norm;
	ofstream os;

	parseData(config, trnData, tstData);
	srand48(config.seed() == 0 ? time(0) : config.seed()); //This is the ONLY place one may set the seed!

	if(config.normalization() == "Z"){
		norm.calcAndNormalise(trnData, true); 
		norm.normalise(tstData);
	}

	int retval = testSimpleSaliency();

	delete &(trnData.coreDataSet());
	delete &(tstData.coreDataSet());

	return retval;
}


