/*$Id: modelselector.cc 1642 2007-05-27 07:41:21Z michael $*/

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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstdlib>
#include <list>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

double bs632Rule(double meanTrn, double meanTst)
{
	return 0.368*meanTrn + 0.632*meanTst;
}

/** Calculate the .632+ rule.
 * \param gamma The no-information error rate. For AUC it's 0.5.
 * \param meanTrn The mean training performance over the bootstrap samples.
 * \param meanTst The mean testing performance over the bootstrap samples.
 */
double Auc632PlusRule(double meanTrn, double meanTst)
{
	double r = 0;
	if(meanTrn > meanTst && meanTst > 0.5) 
		r = (meanTst - meanTrn) / (0.5 - meanTrn);
	double w = 0.632 / (1.0 - 0.368*r);
	double meanTstPrime = (meanTst > 0.5) ? meanTst : 0.5;
	return (1.0-w)*meanTrn + w*meanTstPrime;
}

void printStats(string err, double trnStats, double valStats, double tstStats)
{
	cout<<"Total training "<<err<<": "<<trnStats<<endl;
	cout<<"Total validation "<<err<<": "<<valStats<<endl;
	cout<<"Total testing "<<err<<": "<<tstStats<<endl;
}

/* Return the cross-validated error for a specific model */
pair<double, double>* trainAndValidateModel(DataSet& trnData, const Config& config)
{

	ModelEstimator* me = 0;
	pair<double, double>* auc = 0;

	if(config.msParamN() > 0){
		me = Factory::createModelEstimator(config, trnData);
		auc = me->runAndEstimateModel(&ErrorMeasures::auc);
		//Use 632 rule if bootstrap was used.
		if(config.msParamDataSelection() == "boot"){ 
			auc->second = Auc632PlusRule(auc->first, auc->second);
		}
	}else{
		cerr<<"Can't do model selection without MSParam set"<<endl;
		abort();
	}
	delete me;
	return auc;
}

Config findBestModel(DataSet& trnData, const Config& config)
{
	Config bestConfig = config;
	Config tmpConfig = config;
	double bestAuc = 0;
	list<double> alphas;
	copy(istream_iterator<double>(cin), istream_iterator<double>(), back_inserter(alphas));

	// Create output file and print header
	string fname = "msresult" + config.suffix() + ".txt";
	ofstream of(fname.c_str());
	of<<"#"<<setw(13)<<"Alpha"<<setw(14)<<"TrnAUC"<<setw(14)<<"ValAUC"<<endl;

	// Print results for all values of alpha and save the best
	for(list<double>::iterator it = alphas.begin(); it != alphas.end(); ++it){
		tmpConfig.weightElimAlpha(*it);
		pair<double, double>* auc = trainAndValidateModel(trnData, tmpConfig);
		of<<setw(14)<<tmpConfig.weightElimAlpha()<<setw(14)<<auc->first<<setw(14)<<auc->second<<endl;
		if( auc->second > bestAuc ){
			bestConfig = tmpConfig;
			bestAuc = auc->second;
		}
		delete auc;
	}
	of.close();

	return bestConfig;
}

void trainAndTest(DataSet& trnData, DataSet& tstData, Normaliser& norm, const Config& config)
{
	EnsembleBuilder* eb = Factory::createEnsembleBuilder(config, trnData);
	Ensemble* committee = eb->buildEnsemble();
	double trn = ErrorMeasures::auc(*committee, trnData);
	double tst = ErrorMeasures::auc(*committee, tstData);
	printStats("AUC", trn, 0, tst);
	trn = ErrorMeasures::crossEntropy(*committee, trnData);
	tst = ErrorMeasures::crossEntropy(*committee, tstData);
	printStats("CEE", trn, 0, tst);
	trn = ErrorMeasures::gof(*committee, trnData);
	tst = ErrorMeasures::gof(*committee, tstData);
	printStats("GOF", trn, 0, tst);
	vector<Session>* sessions = &(eb->sessions());
	delete committee;

	if(config.saveOutputList() == true){
		ofstream os;
		string fname = "outputlist."+config.suffix()+".txt";
		os.open(fname.c_str(), ios::out);
		if(!os){
			cerr<<"Could not open output file: "<<fname<<endl;
			abort();
		}
		PrintUtils::printTstEnslist(os, *sessions, trnData, tstData, config);
		os.close();
	}
	if(config.saveSession() == true){ //This should be replaced later.
		ofstream os;
		string fname = "networks."+config.suffix()+".xml";
		os.open(fname.c_str(), ios::out);
		if(!os){
			cerr<<"Could not open output file: "<<fname<<endl;
			abort();
		}
		PrintUtils::printXML(os, *sessions, norm, config);
		os.close();
	}

	delete eb;
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
	config.print(cout);
}

int main(int argc, char* argv[])
{
	Config config;
	if(argc == 2){
		parseConf(argv[1], config);
	}else if(argc == 1){
		cout<<"Usage: "<<endl<<"modelselector configfile"<<endl;
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
		norm.calcAndNormalise(trnData, true);
		norm.normalise(tstData); 
	}

	//Find the best weight elimanation constant and test it.
	Config bestConfig = findBestModel(trnData, config);
	trainAndTest(trnData, tstData, norm, bestConfig);

	delete &(trnData.coreDataSet());
	delete &(tstData.coreDataSet());

	return 0;
}
