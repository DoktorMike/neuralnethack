#include "Supervisor.hh"
#include "Parser.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
#include "GradientDescent.hh"
#include "QuasiNewton.hh"
#include "SummedSquare.hh"
#include "CrossValidator.hh"

using namespace NeuralNetHack;
using namespace DataTools;

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

void testDataManager(DataSet& data)
{
	cout<<"Testing splitting of data.\n";
	DataManager manager;
	manager.random(true);
	vector<DataSet> datasets = manager.split(data,2);
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
	Mlp* mlp = new Mlp(config.arch, config.actFcn, false);
	error->mlp(mlp);
	trainer->error(error);
	trainer->numEpochs(config.maxEpochs);

	CrossValidator cv;
	cout<<"Performing "<<config.msParamN<<" runs of "<<config.msParamK<<"-fold crossvalidation.\n";
	cv.crossValidate(*trainer, data, config.msParamN, config.msParamK);
	cout<<"Training ROC for all boxes:\n";
	cv.printTrainingStats(cout, 0, 0);
	cout<<"Validation ROC for all boxes:\n";
	cv.printValidationStats(cout, 0, 0);

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
	trainer->batchSize(config.batchSize);
	trainer->numEpochs(config.maxEpochs);
	trainer->error(error);
	trainer->train(*mlp, dset);
}

void testSupervisor(Config& config, DataSet& dset)
{
	   Supervisor supervisor(config, dset);
	   supervisor.train();
}

void parseConfAndData(string fname, Config& config, CoreDataSet& data)
{
	ifstream in;
	ifstream apa;

	cout<<endl<<"Parsing and storing Configuration."<<endl;
	in.open(fname.c_str(), ios::in);
	Parser::readConfigurationFile(in, config);
	in.close();
	config.print();

	cout<<endl<<"Parsing and adding data to the DataSet."<<endl;
	apa.open(config.fileName.c_str(), ios::in);
	assert(apa);
	Parser::readDataFile(apa, config.inCols, 
			config.outCols, data);
	apa.close();
}

int main(int argc, char* argv[])
{
	string fname;
	if(argc>1)
		fname=string(argv[1]);
	else
		fname="./config.txt";

	Config config;
	CoreDataSet coreData;
	parseConfAndData(fname, config, coreData);
	DataSet data;
	data.coreDataSet(coreData);
	//testCoreDataSetAndDataSet(coreData);
	//testDataManager(data);
	Normaliser norm;
	norm.normalise(data, true);
	//testTrainer(config, data);
	testCrossValidator(data, config);
	//testSupervisor(config, data);
	return 1;
}
