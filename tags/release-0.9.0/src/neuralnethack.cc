/*$Id: neuralnethack.cc 1628 2007-05-09 10:37:15Z michael $*/

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


#include <neuralnethack/parser/Parser.hh>
#include <neuralnethack/datatools/Normaliser.hh>
#include <neuralnethack/datatools/DataManager.hh>
#include <neuralnethack/datatools/CoreDataSet.hh>
#include <neuralnethack/evaltools/Roc.hh>
#include <neuralnethack/Factory.hh>
#include <neuralnethack/PrintUtils.hh>
#include <neuralnethack/NeuralNetHack.hh>
#include <neuralnethack/ModelEstimator.hh>
#include <neuralnethack/ModelSelector.hh>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstdlib>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

/* Return the cross-validation for a specific model */
ModelEstimator* trainAndValidateModel(DataSet& trnData, const Config& config)
{
	ModelEstimator* me = 0;
	if(config.msParamN() > 0){
		me = Factory::createModelEstimator(config, trnData);
		pair<double, double>* tmp = me->runAndEstimateModel(&ErrorMeasures::auc);
		delete tmp;
	}
	return me;
}

/* Return the cross-validation for a specific model */
EnsembleBuilder* trainAndTestModel(DataSet& trnData, DataSet& tstData, const Config& config)
{
	EnsembleBuilder* eb = 0;
	if(config.ensParamN() > 0){
		eb = Factory::createEnsembleBuilder(config, trnData);
		Ensemble* ensemble = eb->buildEnsemble();
		delete ensemble;
	}
	return eb;
}

void trainAndTestSingle(DataSet& trnData, DataSet& tstData, Normaliser& norm, const Config& config)
{
	Trainer* trainer = Factory::createTrainer(config, trnData);
	trainer->train(cout);
	Error* error = trainer->error();
	Mlp* mlp = trainer->mlp();
	Ensemble c(*mlp, 1);
	//double aucTrn = EvalTools::ErrorMeasures::auc(c, trnData);
	//double aucTst = EvalTools::ErrorMeasures::auc(c, tstData);

	delete trainer;
	delete error;
	delete mlp;
}

void saveOutputList(vector<Session>& sessions, DataSet& trnData, DataSet& tstData, Config& config)
{
	if(config.saveOutputList() == false) return;
	ofstream os;
	string fname = "outputlist."+config.suffix()+".txt";
	os.open(fname.c_str(), ios::out);
	if(!os){
		cerr<<"Could not open output file: "<<fname<<endl;
		abort();
	}
	PrintUtils::printEnslist(os, sessions, trnData, tstData, config);
	os.close();
}

void saveSession(vector<Session>& sessions, Normaliser& norm, Config& config)
{
	if(config.saveSession() == false) return;
	ofstream os;
	string fname = "networks."+config.suffix()+".xml";
	os.open(fname.c_str(), ios::out);
	if(!os){
		cerr<<"Could not open output file: "<<fname<<endl;
		abort();
	}
	PrintUtils::printXML(os, sessions, norm, config);
	os.close();
}


void doStuff(DataSet& trn, DataSet &tst, Normaliser& norm, Config& config)
{
	Config best = config;
	// Open the file to print results to
	ofstream os;
	string fname = "result."+config.suffix()+".txt";
	os.open(fname.c_str(), ios::out);
	if(!os){
		cerr<<"Could not open output file: "<<fname<<endl;
		abort();
	}

	vector<Session> sessions;
	if(!config.vary().empty()){ //Model selection, training and testing
		ModelSelector ms;
		best = ms.findBestModel(trn, config);
		EnsembleBuilder* eb = Factory::createEnsembleBuilder(best, trn);
		Ensemble* ensemble = eb->buildEnsemble();
		double trnAuc = ErrorMeasures::auc(*ensemble, trn);
		double tstAuc = ErrorMeasures::auc(*ensemble, tst);
		os<<setw(3)<<" "<<setw(14)<<"Trn"<<setw(14)<<"Tst"<<endl;
		os<<setw(3)<<"AUC"<<setw(14)<<trnAuc<<setw(14)<<tstAuc<<endl;
		sessions = eb->sessions();
		delete eb;
		delete ensemble;
	}else if(config.msParamN() > 0){ //Validation, training and testing
		ModelEstimator* me = Factory::createModelEstimator(config, trn);
		pair<double, double>* auc = me->runAndEstimateModel(&ErrorMeasures::auc);
		double trnAuc = auc->first;
		double valAuc = auc->second;
		os<<setw(3)<<" "<<setw(14)<<"Trn"<<setw(14)<<"Val"<<endl;
		os<<setw(3)<<"AUC"<<setw(14)<<trnAuc<<setw(14)<<valAuc<<endl;
		sessions = me->sessions();
		delete auc;
		delete me;
	}else{ //Training and testing
		EnsembleBuilder* eb = Factory::createEnsembleBuilder(best, trn);
		Ensemble* ensemble = eb->buildEnsemble();
		double trnAuc = ErrorMeasures::auc(*ensemble, trn);
		double tstAuc = ErrorMeasures::auc(*ensemble, tst);
		os<<setw(3)<<" "<<setw(14)<<"Trn"<<setw(14)<<"Tst"<<endl;
		os<<setw(3)<<"AUC"<<setw(14)<<trnAuc<<setw(14)<<tstAuc<<endl;
		sessions = eb->sessions();
		delete eb;
		delete ensemble;
	}
	os.close();
	if(config.saveSession() == true) saveSession(sessions, norm, best);
	if(config.saveOutputList() == true) saveOutputList(sessions, trn, tst, best);
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


void parseData(Config& config, DataSet& trnData, DataSet& tstData)
{
	ifstream trnStream;
	ifstream tstStream;
	CoreDataSet* trnCoreData = new CoreDataSet();
	CoreDataSet* tstCoreData = new CoreDataSet();

	cout<<"Parsing and adding data to the training DataSet."<<endl<<endl;
	trnStream.open(config.fileName().c_str(), ios::in);
	if(!trnStream){
		cerr<<"Could not open data file: "<<config.fileName()<<endl;
		abort();
	}
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(), 
			config.outputColumns(), config.rowRange(), *trnCoreData);
	trnStream.close();
	trnData.coreDataSet(*trnCoreData);

	cout<<"Parsing and adding data to the testing DataSet."<<endl<<endl;
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

	DataSet trnData, tstData;
	Normaliser norm;
	ofstream os;

	parseData(config, trnData, tstData);
	cout<<"Printing configuration file."<<endl<<endl; 
	config.print(cout);
	srand48(config.seed() == 0 ? time(0) : config.seed()); //This is the ONLY place one may set the seed!

	//Normalise training data last since those are the coeff we want printed
	if(config.normalization() == "Z"){
		norm.normalise(trnData, true); 
		norm.normalise(tstData, true);
	}

	doStuff(trnData, tstData, norm, config);
	//trainAndTest(trnData, tstData, norm, config);
	//trainAndTestSingle(trnData, tstData, norm, config);

	delete &(trnData.coreDataSet());
	delete &(tstData.coreDataSet());

	return 0;
}
