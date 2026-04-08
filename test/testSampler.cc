#include <datatools/CrossSplitSampler.hh>
#include <datatools/BootstrapSampler.hh>
#include <datatools/HoldOutSampler.hh>
#include <parser/Parser.hh>
#include <Config.hh>

#include <utility>
#include <fstream>
#include <cassert>

using namespace NeuralNetHack;
using namespace std;

void parseConfAndData(const string fname, Config& config, DataTools::CoreDataSet& trnData, DataTools::CoreDataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;

	confStream.open(fname.c_str(), ios::in);
	assert(confStream);
	Parser::readConfigurationFile(confStream, config);
	confStream.close();

	trnStream.open(config.fileName().c_str(), ios::in);
	assert(trnStream);
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(), 
			config.outputColumns(), config.rowRange(), trnData);
	trnStream.close();

	tstStream.open(config.fileNameT().c_str(), ios::in);
	assert(tstStream);
	Parser::readDataFile(tstStream, config.idColumnT(), config.inputColumnsT(), 
			config.outputColumnsT(), config.rowRangeT(), tstData);
	tstStream.close();
}

int main(const int argc, const char *argv[])
{
	Config config;
	DataTools::CoreDataSet trnCoreData;
	DataTools::CoreDataSet tstCoreData;
	parseConfAndData("./config.txt", config, trnCoreData, tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	uint cntr = 0;
	bool ok = true;

	//Testing the CrossSplitSampler sampling.
	DataTools::Sampler* sampler = new DataTools::CrossSplitSampler(trnData, 3, 5);
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		/*
		cout<<"Printing Training set "<<cntr<<endl;
		trnVal->first.print(cout);
		cout<<"Printing Validation set "<<cntr++<<endl;
		trnVal->second.print(cout);
		*/
		delete trnVal;
	}
	if(cntr != 15) ok = false;
	cntr = 0;
	sampler->reset();
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		delete trnVal;
	}
	if(cntr != 15) ok = false;
	cntr = 0;

	delete sampler;
	
	//Testing the HoldOutSampler sampling.
	sampler = new DataTools::HoldOutSampler(trnData, 0.2, 15);
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		/*
		cout<<"Printing Training set "<<cntr<<endl;
		trnVal->first.print(cout);
		cout<<"Printing Validation set "<<cntr<<endl;
		trnVal->second.print(cout);
		*/
		delete trnVal;
	}
	if(cntr != 15) ok = false;
	cntr = 0;
	sampler->reset();
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		delete trnVal;
	}
	if(cntr != 15) ok = false;
	cntr = 0;

	delete sampler;

	//Testing the BootstrapSampler sampling.
	sampler = new DataTools::BootstrapSampler(trnData, 15);
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		delete trnVal;
	}
	if(cntr != 15) ok = false;
	cntr = 0;
	sampler->reset();
	while(sampler->hasNext()){
		cntr++;
		std::pair<DataTools::DataSet, DataTools::DataSet>* trnVal = sampler->next();
		delete trnVal;
	}
	if(cntr != 15) ok = false;

	return (ok == true) ? 0 : -1;
}
