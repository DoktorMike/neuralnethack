#include "Parser.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
#include "evaltools/Roc.hh"
#include "Factory.hh"
#include "PrintUtils.hh"
#include "ErrorMeasures.hh"

#include <iostream>
#include <fstream>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace Factory;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

void printAuc(double trnAuc, double valAuc, double tstAuc)
{
	cout<<"Total training AUC: "<<trnAuc<<endl;
	cout<<"Total validation AUC: "<<valAuc<<endl;
	cout<<"Total testing AUC: "<<tstAuc<<endl;
}

void printOutputTargetList(ModelEstimator* ms, Committee* c, DataSet& trnData, DataSet& tstData, Config& config)
{
	if(config.saveOutputList == true){
		string fname = "outputlist."+config.suffix;
		ofstream os(fname.c_str(), ios::out);
		if(!os){
			cerr<<"Could not open output file: "<<fname<<endl;
			abort();
		}
		os<<"#! ptype\t"<<!config.problemType<<endl;
		os<<"#! targets\t"<<"1"<<endl;
		os<<"#! nout \t"<<trnData.nOutput()<<endl;

		if(c != 0){ //Only train and test.
			PrintUtils::printTargetList(os, ">>target trn", trnData);
			PrintUtils::printTargetList(os, ">>target tst", tstData);
			PrintUtils::printOutputList(os, ">>trn", *c, trnData);
			PrintUtils::printOutputList(os, ">>tst", *c, tstData);
		}else{ //Training and validation and no test.
			ms->printOutputTargetList(os);
		}
	}
}

void trainAndTest(DataSet& trnData, DataSet& tstData, Config& config)
{

	ModelEstimator* me = createModelEstimator(config, trnData);
	EnsembleBuilder* eb = me->ensembleBuilder();
	Trainer* trainer = eb->trainer();
	Error* error = trainer->error();
	Mlp* mlp = trainer->mlp();

	if(config.msParamN > 0){
		pair<double, double>* auc = me->estimateModel();
		printAuc(auc->first, auc->second, 0);
		delete auc;
		printOutputTargetList(me, 0, trnData, tstData, config);
	}else{
		Committee* committee = eb->buildEnsemble();
		double trnAuc = ErrorMeasures::auc(*committee, trnData);
		double tstAuc = ErrorMeasures::auc(*committee, tstData);
		printAuc(trnAuc, 0, tstAuc);
		printOutputTargetList(0, committee, trnData, tstData, config);
		delete committee;
	}

	delete me;
	delete eb;
	delete trainer;
	delete error;
	delete mlp;
}

void parseConfAndData(string fname, Config& config, DataSet& trnData, DataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;
	CoreDataSet* trnCoreData = new CoreDataSet();
	CoreDataSet* tstCoreData = new CoreDataSet();

	cout<<"Parsing and storing Configuration."<<endl<<endl;
	confStream.open(fname.c_str(), ios::in);
	if(!confStream){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();

	cout<<"Parsing and adding data to the training DataSet."<<endl<<endl;
	trnStream.open(config.fileName.c_str(), ios::in);
	if(!trnStream){
		cerr<<"Could not open data file: "<<config.fileName<<endl;
		abort();
	}
	Parser::readDataFile(trnStream, config.inCols, 
			config.outCols, *trnCoreData);
	trnStream.close();
	trnData.coreDataSet(*trnCoreData);

	cout<<"Parsing and adding data to the testing DataSet."<<endl<<endl;
	tstStream.open(config.fileNameT.c_str(), ios::in);
	if(!tstStream){
		cerr<<"Could not open data file: "<<config.fileNameT<<endl;
		abort();
	}
	Parser::readDataFile(tstStream, config.inColsT, 
			config.outColsT, *tstCoreData);
	tstStream.close();
	tstData.coreDataSet(*tstCoreData);
}

string parseCmdLine(int argc, char* argv[])
{
	if(argc>1)
		return string(argv[1]);
	else{
		cerr<<"Usage: "<<argv[0]<<" configfile"<<endl;
		exit(1);
	}
}

int main(int argc, char* argv[])
{
	Config config;
	string fname = parseCmdLine(argc, argv);
	DataSet trnData, tstData;
	Normaliser norm;
	ofstream os;


	parseConfAndData(fname, config, trnData, tstData);
	cout<<"Printing configuration file."<<endl<<endl; 
	config.print();
	srand(config.seed == 0 ? time(0) : config.seed); //This is the ONLY place one may set the seed!

	norm.normalise(trnData, true); 
	norm.normalise(tstData, true);

	trainAndTest(trnData, tstData, config);

	delete &(trnData.coreDataSet());
	delete &(tstData.coreDataSet());

	return 0;
}
