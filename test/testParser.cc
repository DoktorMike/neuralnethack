#include "datatools/DataSet.hh"
#include "datatools/CoreDataSet.hh"
#include "parser/Parser.hh"

#include <iostream>
#include <fstream>
#include <cassert>

using namespace NeuralNetHack;
using namespace std;

int testParser(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData)
{
	ofstream ofs("tmp.txt");
	config.print(ofs);
	ofs.close();
	int ret = system("diff -q parserconfigure.txt tmp.txt");
	system("rm tmp.txt");
	return (ret > 0) ? -1 : 0;
}

void parseConfAndData(string fname, Config& config, DataTools::CoreDataSet& trnData, DataTools::CoreDataSet& tstData)
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

int main(int argc, char* argv[])
{
	srand(1);

	string fname;
	//if(argc>1) fname=string(argv[1]); else fname="./apa.txt";
	if(argc>1) fname=string(argv[1]); else fname="./parserconfigure.txt";

	Config config;
	DataTools::CoreDataSet trnCoreData;
	DataTools::CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	return testParser(config, trnData, tstData);
}
