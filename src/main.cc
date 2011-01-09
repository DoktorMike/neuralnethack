#include "Parser.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
#include "evaltools/Roc.hh"
#include "GradientDescent.hh"
#include "QuasiNewton.hh"
#include "SummedSquare.hh"
#include "CrossEntropy.hh"
#include "CrossValidator.hh"

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;

Trainer* createTrainer(Config& config)
{
	Trainer* trainer = 0;
	if(config.minMethod == GD){
		trainer = new GradientDescent(
				MAX_ERROR, 
				config.batchSize,
				config.learningRate,
				config.decLearningRate,
				config.momentum,
				config.weightElimOn,
				config.weightElimAlpha,
				config.weightElimW0);
	}else{
		trainer = new QuasiNewton(
				MAX_ERROR, 
				config.batchSize,
				config.weightElimOn,
				config.weightElimAlpha,
				config.weightElimW0);
	}
	Error* error = 0;
	if(config.errFcn == SSE)
		error = new SummedSquare();
	else if(config.errFcn == CEE)
		error = new CrossEntropy();
	Mlp* mlp = new Mlp(config.arch, config.actFcn, false);
	error->mlp(mlp);
	trainer->error(error);
	trainer->numEpochs(config.maxEpochs);
	return trainer;
}

/**Perform stats on the crossValidation and the test set.
 * The validation AUC is based on a committee size of N. The Test AUC is also
 * based on N mlps by reducing the final N*K committee size to N.
 */
void doStats(Trainer* trainer, DataSet& trnData, DataSet& tstData, Config& config)
{
	CrossValidator cv;
	Committee committee;
	
	if(config.msParamK > 1){
		cout<<"Performing "<<config.msParamN<<" runs of "<<
			config.msParamK<<"-fold crossvalidation.\n";
		cv.crossValidate(*trainer, trnData, config.msParamN, config.msParamK);
		committee = cv.committee();
	}else{
		cout<<"Performing "<<config.msParamN<<
			" runs of training and testing using ratio "<<
			config.msParamNumTrainingData<<".\n";
		cv.crossValidate(*trainer, trnData, config.msParamN, config.msParamNumTrainingData);
		committee = cv.committee();
	}

	cout<<"Reducing committee size to "<<config.msParamN<<".\n";
	uint cntr = 0;
	while(committee.size() > config.msParamN){
		if(cntr < committee.size() - 1)
			cntr++;
		else
			cntr = 0;
		committee.delMlp(cntr);
	}
	
	tstData.reset();
	vector<double> output(0);
	vector<uint> target(0);
	while(tstData.remaining()){
		Pattern& pat = tstData.nextPattern();
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
	}
	Roc roc;

	cout<<"Total training ";
	cout<<"AUC: "<<cv.auc(false)<<endl;
	cout<<"Total validation ";
	cout<<"AUC: "<<cv.auc(true)<<endl;
	cout<<"Total testing ";
	cout<<"AUC: "<<roc.calcAucWmw(output, target)<<endl;
}

void crossValidate(DataSet& trnData, DataSet& tstData, Config& config)
{
	Trainer* trainer = createTrainer(config);
	Error* error = trainer->error();
	Mlp* mlp = error->mlp();

	doStats(trainer, trnData, tstData, config);

	delete mlp;
	delete error;
	delete trainer;
}

/**Perform stats on the crossValidation and the test set.
 * We set the newN to N*K and keep K thus yielding a validation AUC based on 
 * newN = N*K mlps. When testing we remove (newN - N)*K mlps from the final
 * N*K^2.
 */
void doSpecialStats(Trainer* trainer, DataSet& trnData, DataSet& tstData, Config& config)
{
	CrossValidator cv;
	Committee committee;

	uint newN = config.msParamN * config.msParamK;
	
	if(config.msParamK > 1){
		cout<<"Performing "<<newN<<" runs of "<<
			config.msParamK<<"-fold crossvalidation.\n";
		cv.crossValidate(*trainer, trnData, newN, config.msParamK);
		committee = cv.committee();
	}else{
		cout<<"Performing "<<newN<<
			" runs of training and testing using ratio "<<
			config.msParamNumTrainingData<<".\n";
		cv.crossValidate(*trainer, trnData, newN, config.msParamNumTrainingData);
		committee = cv.committee();
	}

	cout<<"Reducing committee size to "<<newN<<".\n";
	while(committee.size() > newN)
		committee.delMlp(committee.size() - 1);
	cout<<"Final committee size is "<<committee.size()<<endl;
	
	tstData.reset();
	vector<double> output(0);
	vector<uint> target(0);
	while(tstData.remaining()){
		Pattern& pat = tstData.nextPattern();
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
	}
	Roc roc;
	
	cout<<"Total training ";
	cout<<"AUC: "<<cv.auc(false)<<endl;
	cout<<"Total validation ";
	cout<<"AUC: "<<cv.auc(true)<<endl;
	cout<<"Total testing ";
	cout<<"AUC: "<<roc.calcAucWmw(output, target)<<endl;
}

void specialCrossValidate(DataSet& trnData, DataSet& tstData, Config& config)
{
	Trainer* trainer = createTrainer(config);
	Error* error = trainer->error();
	Mlp* mlp = error->mlp();

	doSpecialStats(trainer, trnData, tstData, config);

	delete mlp;
	delete error;
	delete trainer;
}

void parseConfAndData(string fname, Config& config, CoreDataSet& trnData, CoreDataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;

	cout<<endl<<"Parsing and storing Configuration."<<endl;
	confStream.open(fname.c_str(), ios::in);
	if(!confStream){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
	//config.print();

	cout<<endl<<"Parsing and adding data to the training DataSet."<<endl;
	trnStream.open(config.fileName.c_str(), ios::in);
	if(!trnStream){
		cerr<<"Could not open data file: "<<config.fileName<<endl;
		abort();
	}
	Parser::readDataFile(trnStream, config.inCols, 
			config.outCols, trnData);
	trnStream.close();

	cout<<endl<<"Parsing and adding data to the testing DataSet."<<endl;
	tstStream.open(config.fileNameT.c_str(), ios::in);
	if(!tstStream){
		cerr<<"Could not open data file: "<<config.fileNameT<<endl;
		abort();
	}
	Parser::readDataFile(tstStream, config.inColsT, 
			config.outColsT, tstData);
	tstStream.close();
}

int main(int argc, char* argv[])
{
	string fname;
	if(argc>1)
		fname=string(argv[1]);
	else{
		cerr<<"Usage: "<<argv[0]<<" configfile\n";
		return 1;
	}

	Config config;
	CoreDataSet trnCoreData;
	CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	config.print();
	DataSet trnData;
	DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	//trnCoreData.print(cout);
	//tstCoreData.print(cout);

	Normaliser norm;
	norm.normalise(trnData, true);
	norm.normalise(tstData, true);
	//crossValidate(trnData, tstData, config);
	specialCrossValidate(trnData, tstData, config);
	return 0;
}
