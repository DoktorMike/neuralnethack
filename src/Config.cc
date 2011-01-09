#include "Config.hh"

#include <iostream>

using namespace NeuralNetHack;
using namespace std;

Config::Config():suffix(""),
	fileName(""),inCols(0),outCols(0),
	fileNameT(""),inColsT(0),outColsT(0),
	problemType(false),nLayers(0),
    arch(0),actFcn(0),errFcn(""),minMethod(""),maxEpochs(0),
    batchSize(0),learningRate(0),decLearningRate(0),momentum(0),
	weightElimOn(false), weightElimAlpha(0), weightElimW0(1),
	ensParamDataSelection(""), ensParamN(0), ensParamK(0), ensParamSplitMode(true), ensParamNewWeights(false),
	msParamDataSelection(""), msParamN(0), msParamK(0), msParamSplitMode(true), msParamNumTrainingData(0),
	msgParamN(0), msgParamK(0), msgParamSplitMode(true), msgParamNumTrainingData(0),
	saveSession(false), info(0), saveOutputList(false), seed(0){}

Config::Config(const Config& config){*this=config;}

Config::~Config(){}

Config& Config::operator=(const Config& config)
{
	if(this != &config){
		suffix = config.suffix;
		fileName = config.fileName;
		inCols = config.inCols;
		outCols = config.outCols;
		fileNameT = config.fileNameT;
		inColsT = config.inColsT;
		outColsT = config.outColsT;
		problemType = config.problemType;
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
		weightElimOn = config.weightElimOn;
		weightElimAlpha = config.weightElimAlpha;
		weightElimW0 = config.weightElimW0;
		ensParamDataSelection = config.ensParamDataSelection;
		ensParamN = config.ensParamN;
		ensParamK = config.ensParamK;
		ensParamSplitMode = config.ensParamSplitMode;
		ensParamNewWeights = config.ensParamNewWeights;
		msParamDataSelection = config.msParamDataSelection;
		msParamN = config.msParamN;
		msParamK = config.msParamK;
		msParamSplitMode = config.msParamSplitMode;
		msParamNumTrainingData = config.msParamNumTrainingData;
		msgParamN = config.msgParamN;
		msgParamK = config.msgParamK;
		msgParamSplitMode = config.msgParamSplitMode;
		msgParamNumTrainingData = config.msgParamNumTrainingData;
		saveSession = config.saveSession;
		info = config.info;
		saveOutputList = config.saveOutputList;
		seed = config.seed;
	}
	return *this;
}

void Config::print()
{
	cout<<"Suffix\t\t"<<suffix<<endl;
	cout<<"Filename\t"<<fileName<<endl;
	cout<<"InCol\t\t";
	for(vector<uint>::iterator it=inCols.begin(); it!=inCols.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"OutCol\t\t";
	for(vector<uint>::iterator it=outCols.begin(); it!=outCols.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"FilenameT\t"<<fileNameT<<endl;
	cout<<"InColT\t\t";
	for(vector<uint>::iterator it=inColsT.begin(); it!=inColsT.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"OutColT\t\t";
	for(vector<uint>::iterator it=outColsT.begin(); it!=outColsT.end(); ++it)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"PType\t\t"<<problemType<<endl;
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
	cout<<"WeightElim\t"<<weightElimOn<<" "<<weightElimAlpha<<" "
		<<weightElimW0<<endl;
	cout<<"EnsParam\t"<<ensParamDataSelection<<" "<<ensParamN<<" "
		<<ensParamK<<" "<<ensParamSplitMode<<" "<<ensParamNewWeights<<endl;
	cout<<"MSParam\t\t"<<msParamDataSelection<<" "<<msParamN<<" "<<msParamK<<" "
		<<msParamSplitMode<<" "<<msParamNumTrainingData
		<<endl;
	cout<<"MSGParam\t"<<msgParamN<<" "<<msgParamK<<" "
		<<msgParamSplitMode<<" "<<msgParamNumTrainingData<<endl;
	cout<<"SaveSession\t"<<saveSession<<endl;
	cout<<"Info\t\t"<<info<<endl;
	cout<<"SaveOutputList\t"<<saveOutputList<<endl;
	cout<<"Seed\t"<<seed<<endl;
}

//PRIVATE--------------------------------------------------------------------//


