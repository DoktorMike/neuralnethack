#include "datatools/Normaliser.hh"
#include "datatools/CoreDataSet.hh"
#include "parser/Parser.hh"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace NeuralNetHack;
using namespace std;

int testNormaliser(DataTools::DataSet& data)
{
	// Save original values
	vector<vector<double>> origInputs, origOutputs;
	for(uint i = 0; i < data.size(); ++i){
		origInputs.push_back(data.pattern(i).input());
		origOutputs.push_back(data.pattern(i).output());
	}

	// Normalise then unnormalise
	DataTools::Normaliser norm;
	norm.calcAndNormalise(data, true);
	norm.unnormalise(data);

	// Compare with tolerance (ffast-math may cause small roundtrip differences)
	const double tol = 1e-6;
	for(uint i = 0; i < data.size(); ++i){
		vector<double>& in = data.pattern(i).input();
		vector<double>& out = data.pattern(i).output();
		for(uint j = 0; j < in.size(); ++j)
			if(fabs(in[j] - origInputs[i][j]) > tol) return -1;
		for(uint j = 0; j < out.size(); ++j)
			if(fabs(out[j] - origOutputs[i][j]) > tol) return -1;
	}
	return 0;
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
	if(argc>1) fname=string(argv[1]); else fname="./config.txt";

	Config config;
	DataTools::CoreDataSet trnCoreData;
	DataTools::CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	return testNormaliser(trnData);
}
