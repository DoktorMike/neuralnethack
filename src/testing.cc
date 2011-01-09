#include "Parser.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
#include "GradientDescent.hh"
#include "QuasiNewton.hh"
#include "SummedSquare.hh"
#include "CrossEntropy.hh"
#include "CrossValidator.hh"

using namespace NeuralNetHack;
using namespace DataTools;

void testMlpCopy()
{
	vector<uint> arch;
	arch.push_back(10);
	arch.push_back(5);
	arch.push_back(1);
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
	for(uint i=0; i<30000; ++i){
		delete mlps[i];
	}
	mlps.clear();
	cout<<"Creating 30 copies of one Mlp.\n";
	Mlp* mlpOrig = new Mlp(arch, types, false);
	for(uint i=0; i<30; ++i){
		Mlp* mlp = new Mlp(*mlpOrig);
		mlps.push_back(mlp);
	}
	cout<<"Deleting the 30 copies of one Mlp.\n";
	for(uint i=0; i<30; ++i){
		delete mlps[i];
	}
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
	cout<<"Testing splitting of data.\n";
	DataManager manager;
	manager.random(true);
	vector<DataSet> datasets = manager.split(data, config.msParamNumTrainingData);
	vector<DataSet>::iterator it = datasets.begin();
	while(it != datasets.end()){
		(it++)->print(cout);
		cout<<"\n";
	}
	cout<<"Testing joining of data.\n";
	DataSet d = manager.join(datasets);
	d.print(cout);
}

void testCoreDataSetAndDataSet(CoreDataSet& data)
{
	vector<uint> indices;
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(3);
	indices.push_back(4);
	DataSet dset;
	dset.coreDataSet(data);
	dset.indices(indices);
	dset.print(cout);
}

void testMlp(Mlp* mlp, DataSet& dset)
{
	dset.reset();
	while(dset.remaining()){
		Pattern& p = dset.nextPattern();
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

void testCrossValidator(DataSet& data, Config& config)
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

	CrossValidator cv;
	if(config.msParamK > 1){
		cout<<"Performing "<<config.msParamN<<" runs of "<<config.msParamK<<"-fold crossvalidation.\n";
		cv.crossValidate(*trainer, data, config.msParamN, config.msParamK);
		cout<<"Training ROC for all boxes:\n";
		//cv.printTrainingStats(cout, 0, 0);
		cout<<"Validation ROC for all boxes:\n";
		//cv.printValidationStats(cout, 0, 0);
	}else{
		cout<<"Performing "<<config.msParamN<<" runs of training and testing using ratio "<<config.msParamNumTrainingData<<".\n";
		//cv.crossValidate(*trainer, data, config.msParamN, config.msParamNumTrainingData);
		cout<<"Training ROC for all boxes:\n";
		//cv.printTrainingStats(cout, 0, 0);
		cout<<"Validation ROC for all boxes:\n";
		//cv.printValidationStats(cout, 0, 0);
	}

	delete trainer;
	delete error;
	delete mlp;
}

void testTrainer(Config& config, DataSet& dset)
{

	Trainer* trainer = 0;
	Error* error = 0;
	Mlp* mlp = new Mlp(config.arch,config.actFcn,false);

	if(config.minMethod == GD)
		trainer = new GradientDescent(
				MAX_ERROR, 
				config.batchSize, 
				config.learningRate, 
				config.decLearningRate, 
				config.momentum,
				config.weightElimOn,
				config.weightElimAlpha, 
				config.weightElimW0);
	else if(config.minMethod == QN)
		trainer = new QuasiNewton(
				MAX_ERROR, 
				config.batchSize,
				config.weightElimOn,
				config.weightElimAlpha, 
				config.weightElimW0);
	if(config.errFcn == SSE)
		error = new SummedSquare();
	else if(config.errFcn == CEE)
		error = new CrossEntropy();
	trainer->batchSize(config.batchSize);
	trainer->numEpochs(config.maxEpochs);
	trainer->error(error);
	trainer->train(*mlp, dset);
	testMlp(mlp, dset);
}

void parseConfAndData(string fname, Config& config, CoreDataSet& trnData, CoreDataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;

	cout<<endl<<"Parsing and storing Configuration."<<endl;
	confStream.open(fname.c_str(), ios::in);
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
	config.print();

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
	if(config.problemType == "regr" && config.errFcn == CEE){
		cerr<<"Regression should not be performed with kullback leibler error function.\n";
		abort();
	}
}

int main(int argc, char* argv[])
{
	string fname;
	if(argc>1)
		fname=string(argv[1]);
	else
		fname="./config.txt";

	testMlpCopy();
	/*
	Config config;
	CoreDataSet trnCoreData;
	CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	config.print();
	DataSet trnData;
	DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	trnCoreData.print(cout);
	tstCoreData.print(cout);
	*/
	//testCoreDataSetAndDataSet(trnCoreData);
	//testDataManager(trnData, config);
	//Normaliser norm;
	//norm.normalise(trnData, true);
	//testTrainer(config, trnData);
	//testCrossValidator(trnData, config);
	return 1;
}
