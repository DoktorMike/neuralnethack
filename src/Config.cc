#include "Config.hh"

using namespace NeuralNetHack;

Config::Config():fileName(""),inCols(0),outCols(0),nLayers(0),
    arch(0),actFcn(0),errFcn(""),minMethod(""),maxEpochs(0),
    batchSize(0),learningRate(0),decLearningRate(0),momentum(0),
	msParamN(0), msParamK(0), msParamSplitMode(true), 
	msParamNumTrainingData(0){}

Config::~Config(){}

void Config::print()
{
	cout<<"Filename\t"<<fileName<<endl;
	cout<<"InCol\t\t";
	for(vector<uint>::iterator it=inCols.begin(); it!=inCols.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"OutCol\t\t";
	for(vector<uint>::iterator it=outCols.begin(); it!=outCols.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"NLay\t\t"<<nLayers<<endl;
	cout<<"Size\t\t";
	for(vector<uint>::iterator it=arch.begin(); it!=arch.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"ActFcn\t\t";
	for(vector<string>::iterator it=actFcn.begin(); it!=actFcn.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"ErrFcn\t\t"<<errFcn<<endl;
	cout<<"MinMethod\t"<<minMethod<<endl;
	cout<<"MaxEpochs\t"<<maxEpochs<<endl;
	cout<<"GDParam\t\t"<<batchSize<<" "<<learningRate<<" "
		<<decLearningRate<<" "<<momentum<<endl;
	cout<<"MSParam\t\t"<<msParamN<<" "<<msParamK<<" "
		<<msParamSplitMode<<" "<<msParamNumTrainingData<<endl;
	cout<<"WeightElim\t"<<weightElimOn<<" "<<weightElimAlpha<<" "
		<<weightElimW0<<endl;
}

//PRIVATE--------------------------------------------------------------------//

Config::Config(const Config& config){*this=config;}

Config& Config::operator=(const Config& config)
{
	if(this != &config){
		fileName = config.fileName;
		inCols = config.inCols;
		outCols = config.outCols;
		nLayers = config.nLayers;
		arch = config.arch;
		actFcn = config.actFcn;
		errFcn = config.errFcn;
		minMethod = config.minMethod;
		maxEpochs = config.maxEpochs;
		batchSize = config.batchSize;
		learningRate = config.learningRate;
		decLearningRate = config.decLearningRate;
		momentum = config.momentum;
		msParamN = config.msParamN;
		msParamK = config.msParamK;
		msParamSplitMode = config.msParamSplitMode;
		msParamNumTrainingData = config.msParamNumTrainingData;
		weightElimOn = config.weightElimOn;
		weightElimAlpha = config.weightElimAlpha;
		weightElimW0 = config.weightElimW0;
	}
	return *this;
}
