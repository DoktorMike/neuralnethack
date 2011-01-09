#include "matrixtools/MatrixTools.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
#include "evaltools/Roc.hh"
#include "Parser.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "CrossValidator.hh"
#include "CrossSplitter.hh"
#include "Bagger.hh"
#include "Bootstrapper.hh"
#include "ErrorMeasures.hh"
#include "Factory.hh"

#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace EvalTools;
using namespace MatrixTools;
using namespace std;

Trainer* createTrainer(Config& config, DataSet& data)
{
	Error* error = 0;
	if(config.errFcn == SSE) error = new SummedSquare();
	else if(config.errFcn == CEE) error = new CrossEntropy();
	Mlp* mlp = new Mlp(config.arch, config.actFcn, false);
	error->mlp(mlp);
	error->weightElimOn(config.weightElimOn);
	error->weightElimAlpha(config.weightElimAlpha);
	error->weightElimW0(config.weightElimW0);
	Trainer* trainer = 0;
	if(config.minMethod == GD){
		trainer = new GradientDescent(
				*mlp, data, *error,
				MAX_ERROR, config.batchSize,
				config.learningRate, config.decLearningRate, config.momentum);
	}else{
		trainer = new QuasiNewton(*mlp, data, *error, MAX_ERROR, config.batchSize);
	}
	trainer->numEpochs(config.maxEpochs);

	return trainer;
}

void testRoc()
{
	Roc roc;

/*
	double out[] = {0.630477,0.52469,0.630477,0.630477,0.461932,0.630477,
		0.630477,0.630477,0.630477,0.63048,0.630477,0.405618,0.412675,0.524524,
		0.524525,0.461721,0.630477,0.461731};
	uint tar[]   = {1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0};
	vector<double> output(out, &out[17]+1);
	vector<uint> target(tar, &tar[17]+1);
*/
	double out[] = {0, -10, 2, -3, 13, 1, 25, 15};
	uint tar[] = {1,0,1,0,1,0,1,0};
	vector<double> output(out, &out[7]+1);
	vector<uint> target(tar, &tar[7]+1);

	//print(output); print(target);

	cout<<"AUC: ";
	cout<<"Trpz: "<<roc.calcAucTrapezoidal(output, target)<<" ";
	cout<<"Wmw: "<<roc.calcAucWmw(output, target)<<" ";
	cout<<"WmwFast: "<<roc.calcAucWmwFast(output, target)<<" ";
	cout<<endl;
}

void testMlpCopy()
{
	vector<uint> arch;
	arch.push_back(10);
	arch.push_back(5);
	arch.push_back(4);
	vector<string> types;
	types.push_back(TANHYP);
	types.push_back(SIGMOID);
	vector<Mlp*> mlps;

	cout<<"Creating 30000 Mlps.\n";
	for(uint i=0; i<30000; ++i){
		Mlp* mlp = new Mlp(arch, types, false);
		mlps.push_back(mlp);
	}
	cout<<"Deleting 30000 Mlps.\n";
	for(vector<Mlp*>::iterator it = mlps.begin(); it != mlps.end(); ++it)
		delete *it;
	mlps.clear();
	cout<<"Creating 30000 copies of one Mlp.\n";
	Mlp* mlpOrig = new Mlp(arch, types, false);
	for(uint i=0; i<30000; ++i){
		Mlp* mlp = new Mlp(*mlpOrig);
		mlps.push_back(mlp);
	}
	cout<<"Deleting the 30000 copies of one Mlp.\n";
	for(vector<Mlp*>::iterator it = mlps.begin(); it != mlps.end(); ++it)
		delete *it;
	mlps.clear();
}

void testNormaliser(DataSet& data)
{
	ofstream dataOrig("DataOrig.txt");
	ofstream dataNorm("DataNorm.txt");
	ofstream dataUnnorm("DataUnnorm.txt");

	data.print(dataOrig);
	cout<<"Normalising data...\n";

	Normaliser norm;
	norm.normalise(data, true);
	data.print(dataNorm);

	cout<<"Unnormalising data...\n";
	norm.unnormalise(data);
	data.print(dataUnnorm);

	dataOrig.close();
	dataNorm.close();
	dataUnnorm.close();
}

void testDataManager(DataSet& data, Config& config)
{
	DataManager manager;
	manager.random(true);

	//ofstream os("crapbefore",ios::out); data.print(os); os.close();
	//vector<DataSet> dataSets = manager.split(data, (uint)1);
	//os.open("crapafter",ios::out); dataSets.front().print(os); os.close();

	cout<<"Testing splitting of data using number of training data.\n";
	pair<DataSet, DataSet> trnVal = manager.split(data, config.msParamNumTrainingData);
	trnVal.first.print(cout); cout<<endl;
	trnVal.second.print(cout); cout<<endl;

	cout<<"Testing splitting the first of the two previous.\n";
	trnVal = manager.split(trnVal.first, config.msParamNumTrainingData);
	trnVal.first.print(cout); cout<<endl;
	trnVal.second.print(cout); cout<<endl;

	cout<<"Testing splitting of data using bootstrapping.\n";
	trnVal = manager.split(data);
	trnVal.first.print(cout); cout<<endl;
	trnVal.second.print(cout); cout<<endl;

	cout<<"Testing splitting the first of the previous.\n";
	trnVal = manager.split(trnVal.first);
	trnVal.first.print(cout); cout<<endl;
	trnVal.second.print(cout); cout<<endl;

	cout<<"Testing splitting of data using number K.\n";
	vector<DataSet> datasets = manager.split(data, config.msParamK);
	vector<DataSet>::iterator it = datasets.begin();
	do{
		it->print(cout);
		cout<<"\n\n";
	}while(++it != datasets.end());

	cout<<"Testing splitting the first of the previous.\n";
	datasets = manager.split(datasets.front(), config.msParamK);
	it = datasets.begin();
	do{
		it->print(cout);
		cout<<endl<<endl;
	}while(++it != datasets.end());

	cout<<"Testing joining of data.\n";
	DataSet d = manager.join(datasets);
	d.print(cout);
}

void buildOutputTarget(Committee& committee, DataSet& dset, vector<double>& output, vector<uint>& target)
{
	output.clear();
	target.clear();

	for(uint i=0; i<dset.size(); ++i){
		Pattern& pat = dset.pattern(i);
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
		//cout<<(uint)pat.output().front()<<"\t";
		//cout<<tmp.front()<<"\n";
	}
}

void evaluateModel(Committee& committee, DataSet& data)
{
	vector<double> output;
	vector<uint> target;
	Roc roc;
	buildOutputTarget(committee, data, output, target);
	cout<<"Trpz: "<<roc.calcAucTrapezoidal(output, target)<<" ";
	cout<<"Wmw: "<<roc.calcAucWmw(output, target)<<" ";
	cout<<"WmwFast: "<<roc.calcAucWmwFast(output, target)<<"\n";
}

void testCrossSplitter(DataSet& trnData, DataSet& tstData, Config& config)
{
	Trainer* trainer = createTrainer(config, trnData);
	CrossSplitter crossSplitter;

	crossSplitter.trainer(trainer);
	crossSplitter.data(&trnData);
	crossSplitter.numRuns(1);
	crossSplitter.numParts(2);
	Committee* committee = crossSplitter.buildEnsemble();
	cout<<"Ensemble size: "<<committee->size()<<endl;

	cout<<"Ensemble tstAUC: ";
	evaluateModel(*committee, tstData);
	cout<<"Ensemble trnAUC: ";
	evaluateModel(*committee, trnData);
	for(uint i=0; i<committee->size(); ++i){
		Committee com;
		com.addMlp(committee->mlp(i));
		cout<<"Model "<<i<<" tstAUC: ";
		evaluateModel(com, tstData);
		cout<<"Model "<<i<<" trnAUC: ";
		evaluateModel(com, trnData);
	}

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
	delete committee;
}

void testBagger(DataSet& trnData, DataSet& tstData, Config& config)
{
	Trainer* trainer = createTrainer(config, trnData);
	Bagger bagger;

	bagger.trainer(trainer);
	bagger.data(&trnData);
	bagger.numRuns(1);
	Committee* committee = bagger.buildEnsemble();
	cout<<"Ensemble size: "<<committee->size()<<endl;

	cout<<"Ensemble tstAUC: ";
	evaluateModel(*committee, tstData);
	cout<<"Ensemble trnAUC: ";
	evaluateModel(*committee, trnData);
	for(uint i=0; i<committee->size(); ++i){
		Committee com;
		com.addMlp(committee->mlp(i));
		cout<<"Model "<<i<<" tstAUC: ";
		evaluateModel(com, tstData);
		cout<<"Model "<<i<<" trnAUC: ";
		evaluateModel(com, trnData);
	}

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
	delete committee;
}

void testBootstrapper(DataSet& trnData, DataSet& tstData, Config& config)
{
	Bootstrapper* bootstrapper = new Bootstrapper();
	Bagger* bagger = new Bagger();
	Trainer* trainer = createTrainer(config, trnData);
	bagger->trainer(trainer);
	bagger->data(&trnData);
	bagger->numRuns(1);
	bootstrapper->ensembleBuilder(bagger);
	bootstrapper->data(&trnData);
	bootstrapper->numRuns(1);

	pair<double, double>* auc = bootstrapper->estimateModel();
	cout<<"AUC trn: "<<auc->first<<"AUC val: "<<auc->second<<"\n";

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
	delete bagger;
	delete bootstrapper;
	delete auc;
}

void testCrossValidator(DataSet& trnData, DataSet& tstData, Config& config)
{
	CrossValidator* crossValidator = new CrossValidator();
	Bagger* bagger = new Bagger();
	Trainer* trainer = createTrainer(config, trnData);
	bagger->trainer(trainer);
	bagger->data(&trnData);
	bagger->numRuns(4);
	crossValidator->ensembleBuilder(bagger);
	crossValidator->data(&trnData);
	crossValidator->numRuns(2);
	crossValidator->numParts(2);

	pair<double, double>* auc = crossValidator->estimateModel();
	cout<<"AUC trn: "<<auc->first<<"AUC val: "<<auc->second<<"\n";
	crossValidator->printOutputTargetList(cout);

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
	delete bagger;
	delete crossValidator;
	delete auc;
}

void testCoreDataSetAndDataSet(CoreDataSet& data)
{
	cout<<"Creating a DataSet from CoreDataSet."<<endl;
	vector<uint> indices;
	//indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	//indices.push_back(3);
	//indices.push_back(4);
	DataSet dset;
	dset.coreDataSet(data);
	dset.indices(indices);
	dset.print(cout);

	cout<<"Creating a copy of the DataSet."<<endl;
	DataSet dset2(dset);
	dset2.print(cout);
}

void testMlp(Mlp* mlp, DataSet& dset)
{
	for(uint i=0; i<dset.size(); ++i){
		Pattern& p = dset.pattern(i);
		vector<double>& in = p.input();
		vector<double>& out = mlp->propagate(in);
		vector<double>& dout = p.output();
		cout<<"In: ";
		for(uint i=0; i<in.size(); ++i)
			cout<<in[i]<<" ";
		cout<<"\tOut: ";
		for(uint i=0; i<out.size(); ++i)
			cout<<out[i]<<" ";
		cout<<"\tTarget: ";
		for(uint i=0; i<dout.size(); ++i)
			cout<<dout[i]<<" ";
		cout<<"\n";
	}
}

void testTrainer(Config& config, DataSet& trnData, DataSet& tstData)
{
	Trainer* trainer = Factory::createTrainer(config, trnData);
	trainer->train();

	//testMlp(trainer->mlp(), trnData);
	Committee c(*(trainer->mlp()), 1);
	double trnErr = ErrorMeasures::auc(c, trnData);
	double tstErr = ErrorMeasures::auc(c, tstData);
	cout<<"TrnAUC: "<<trnErr<<endl;
	cout<<"TstAUC: "<<tstErr<<endl;

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
}

void parseConfAndData(string fname, Config& config, CoreDataSet& trnData, CoreDataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;

	cout<<endl<<"Parsing and storing Configuration."<<endl;
	confStream.open(fname.c_str(), ios::in);
	assert(confStream);
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
	//config.print();

	cout<<endl<<"Parsing and adding data to the training DataSet."<<endl;
	trnStream.open(config.fileName.c_str(), ios::in);
	assert(trnStream);
	Parser::readDataFile(trnStream, config.inCols, 
			config.outCols, trnData);
	trnStream.close();

	cout<<endl<<"Parsing and adding data to the testing DataSet."<<endl;
	tstStream.open(config.fileNameT.c_str(), ios::in);
	assert(tstStream);
	Parser::readDataFile(tstStream, config.inColsT, 
			config.outColsT, tstData);
	tstStream.close();
}

void errFcnVsPType(Config& config)
{
	if(config.problemType == true && config.errFcn == CEE){
		cerr<<"Regression should not be performed with kullback leibler error function.\n";
		abort();
	}
}

int main(int argc, char* argv[])
{
	//srand(time(0)); //This is the ONLY place one may set the seed!
	srand(1);
	cout<<rand()<<endl;

	string fname;
	if(argc>1)
		fname=string(argv[1]);
	else
		fname="./config.txt";

	//testMlpCopy(); abort();
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
	//testCoreDataSetAndDataSet(trnCoreData);
	//testDataManager(trnData, config);
	Normaliser norm;
	norm.normalise(trnData, true);
	norm.normalise(tstData, true);
	testTrainer(config, trnData, tstData);
	//testCrossValidator(trnData, config);
	//testCrossSplitter(trnData, tstData, config);
	//testBagger(trnData, tstData, config);
	//testBootstrapper(trnData, tstData, config);
	//testCrossValidator(trnData, tstData, config);
	//testRoc();

	return 1;
}
