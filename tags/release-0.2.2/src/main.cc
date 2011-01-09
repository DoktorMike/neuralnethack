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
	Mlp* mlp = new Mlp(config.arch, config.actFcn, false);
	error->mlp(mlp);
	trainer->error(error);
	trainer->numEpochs(config.maxEpochs);
	return trainer;
}

void doStats(Trainer* trainer, DataSet& data, Config& config)
{
	CrossValidator cv;
	cout<<"Performing "<<config.msParamN<<" runs of "<<config.msParamK<<"-fold crossvalidation.\n";
	cv.crossValidate(*trainer, data, config.msParamN, config.msParamK);
	for(uint i=1; i<=config.msParamN; ++i){
		cout<<"Training\t(N="<<i<<") ";
		cv.printTrainingStats(cout, i, 0, false);
		cout<<"Validation\t(N="<<i<<") ";
		cv.printValidationStats(cout, i, 0, false);
	}

	cout<<"Total training ";
	cv.printTrainingStats(cout, 0, 0, false);
	cout<<"Total validation ";
	cv.printValidationStats(cout, 0, 0, false);

	ofstream myOstream;
	myOstream.open("tr_tot.dat", ios::out);
	cv.printTrainingStats(myOstream, 0, 0, true);
	myOstream.close();
	myOstream.open("val_tot.dat", ios::out);
	cv.printValidationStats(myOstream, 0, 0, true);
	myOstream.close();
}

void crossValidate(DataSet& data, Config& config)
{
	Trainer* trainer = createTrainer(config);
	Error* error = trainer->error();
	Mlp* mlp = error->mlp();

	doStats(trainer, data, config);


	delete mlp;
	delete error;
	delete trainer;
}

void parseConfAndData(string fname, Config& config, CoreDataSet& data)
{
	ifstream in;
	ifstream apa;

	cout<<endl<<"Parsing and storing Configuration."<<endl;
	in.open(fname.c_str(), ios::in);
	if(!in){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(in, config);
	in.close();
	//config.print();

	cout<<endl<<"Parsing and adding data to the DataSet."<<endl;
	apa.open(config.fileName.c_str(), ios::in);
	if(!apa){
		cerr<<"Could not open data file: "<<config.fileName<<endl;
		abort();
	}
	Parser::readDataFile(apa, config.inCols, 
			config.outCols, data);
	apa.close();
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
	CoreDataSet coreData;
	parseConfAndData(fname, config, coreData);
	DataSet data;
	data.coreDataSet(coreData);
	Normaliser norm;
	norm.normalise(data, true);
	crossValidate(data, config);
	return 0;
}
