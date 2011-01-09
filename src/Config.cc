#include "Config.hh"

using namespace NetHack;

Config::Config():theFileName(""),theInCols(0),theOutCols(0),theNLayers(0),
    theArch(0),theActFcn(0),theErrFcn(""),theMinMethod(""),theMaxEpochs(0),
    theBatchSize(0),theLearningRate(0),theDecLearningRate(0),theMomentum(0){}

Config::~Config(){}

string Config::filename(){return theFileName;}
vector<uint>& Config::inputColumns(){return theInCols;}
vector<uint>& Config::outputColumns(){return theOutCols;}
uint Config::nLayers(){return theNLayers;}
vector<uint>& Config::arch(){return theArch;}
vector<string>& Config::activationFunctions(){return theActFcn;}
string Config::errorFunction(){return theErrFcn;}
string Config::learningAlgorithm(){return theMinMethod;}
uint Config::nEpochs(){return theMaxEpochs;}
uint Config::batchSize(){return theBatchSize;}
double Config::learningRate(){return theLearningRate;}
double Config::decLearningRate(){return theDecLearningRate;}
double Config::momentum(){return theMomentum;}

void Config::filename(string fname){theFileName=fname;}
void Config::inputColumns(vector<uint> in){theInCols=in;}
void Config::outputColumns(vector<uint> out){theOutCols=out;}
void Config::nLayers(uint n){theNLayers=n;}
void Config::arch(vector<uint> a){theArch=a;}
void Config::activationFunctions(vector<string> af){theActFcn=af;}
void Config::errorFunction(string ef){theErrFcn=ef;}
void Config::learningAlgorithm(string la){theMinMethod=la;}
void Config::nEpochs(uint me){theMaxEpochs=me;}
void Config::batchSize(uint bs){theBatchSize=bs;}
void Config::learningRate(double lr){theLearningRate=lr;}
void Config::decLearningRate(double dlr){theDecLearningRate=dlr;}
void Config::momentum(double m){theMomentum=m;}

void Config::print()
{
    cout<<"Filename\t"<<theFileName<<endl;
    cout<<"InCol\t\t";
    for(vector<uint>::iterator it=theInCols.begin(); it!=theInCols.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
    cout<<"OutCol\t\t";
    for(vector<uint>::iterator it=theOutCols.begin(); it!=theOutCols.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
    cout<<"NLay\t\t"<<theNLayers<<endl;
    cout<<"Size\t\t";
    for(vector<uint>::iterator it=theArch.begin(); it!=theArch.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
    cout<<"ActFcn\t\t";
    for(vector<string>::iterator it=theActFcn.begin(); it!=theActFcn.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
    cout<<"ErrFcn\t\t"<<theErrFcn<<endl;
    cout<<"MinMethod\t"<<theMinMethod<<endl;
    cout<<"MaxEpochs\t"<<theMaxEpochs<<endl;
    cout<<"GDParam\t\t"<<theBatchSize<<" "<<theLearningRate<<" "
	<<theDecLearningRate<<" "<<theMomentum<<endl;
}

//PRIVATE--------------------------------------------------------------------//

Config::Config(const Config& config){*this=config;}

Config& Config::operator=(const Config& config)
{
	if(this != &config){
	    theFileName = config.theFileName;
	    theInCols = config.theInCols;
	    theOutCols = config.theOutCols;
	    theNLayers = config.theNLayers;
	    theArch = config.theArch;
	    theActFcn = config.theActFcn;
	    theErrFcn = config.theErrFcn;
	    theMinMethod = config.theMinMethod;
	    theMaxEpochs = config.theMaxEpochs;
	    theBatchSize = config.theBatchSize;
	    theLearningRate = config.theLearningRate;
	    theDecLearningRate = config.theDecLearningRate;
	    theMomentum = config.theMomentum;
	}
	return *this;
}
