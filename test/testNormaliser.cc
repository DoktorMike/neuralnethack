#include "datatools/Normaliser.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <memory>
#include <string>
#include "parser/Parser.hh"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace NeuralNetHack;
using namespace std;

int testNormaliser(DataTools::DataSet& data) {
	// Save original values
	vector<vector<double>> origInputs, origOutputs;
	for (uint i = 0; i < data.size(); ++i) {
		origInputs.push_back(data.pattern(i).input());
		origOutputs.push_back(data.pattern(i).output());
	}

	// Normalise then unnormalise
	DataTools::Normaliser norm;
	norm.calcAndNormalise(data, true);
	norm.unnormalise(data);

	// Compare with tolerance (ffast-math may cause small roundtrip differences)
	const double tol = 1e-6;
	for (uint i = 0; i < data.size(); ++i) {
		vector<double>& in = data.pattern(i).input();
		vector<double>& out = data.pattern(i).output();
		for (uint j = 0; j < in.size(); ++j)
			if (fabs(in[j] - origInputs[i][j]) > tol) return -1;
		for (uint j = 0; j < out.size(); ++j)
			if (fabs(out[j] - origOutputs[i][j]) > tol) return -1;
	}
	return 0;
}

// Max-abs mode: zero preservation, sign preservation, grouped scaling,
// scale into [-1,1], and round-trip.
int testMaxAbs() {
	using namespace DataTools;
	auto core = std::make_shared<CoreDataSet>();
	// 3 inputs: two lag columns of one "channel" (group 0) at wildly
	// different maxima, one covariate that goes negative (group 1);
	// 1 output (group 2).
	std::vector<std::vector<double>> rows = {
	    {20000, 0, -0.5, 5.0}, {0, 10000, 0.25, 3.0}, {15000, 20000, 0.0, 4.0}};
	uint id = 0;
	for (auto& r : rows) {
		std::vector<double> in(r.begin(), r.begin() + 3), out(r.begin() + 3, r.end());
		core->addPattern(Pattern(std::to_string(id++), in, out));
	}
	DataSet d;
	d.coreDataSet(core);

	Normaliser norm;
	std::vector<uint> groups = {0, 0, 1}; // inputs only
	norm.calcAndNormaliseMaxAbs(d, groups);

	const double tol = 1e-12;
	// Grouped scale: both lag columns divided by 20000
	if (fabs(d.pattern(0).input()[0] - 1.0) > tol) return -1;
	if (fabs(d.pattern(1).input()[1] - 0.5) > tol) return -1;
	// Zeros stay zeros
	if (d.pattern(0).input()[1] != 0.0 || d.pattern(1).input()[0] != 0.0) return -1;
	// Sign preserved, scaled by max|x| = 0.5
	if (fabs(d.pattern(0).input()[2] - (-1.0)) > tol) return -1;
	if (fabs(d.pattern(1).input()[2] - 0.5) > tol) return -1;
	// Outputs untouched (target keeps natural units)
	if (fabs(d.pattern(1).output()[0] - 3.0) > tol) return -1;

	// Round-trip
	norm.unnormalise(d);
	for (uint i = 0; i < d.size(); ++i)
		for (uint j = 0; j < 3; ++j)
			if (fabs(d.pattern(i).input()[j] - rows[i][j]) > 1e-6) return -1;

	// Ungrouped: each column its own max
	DataSet d2;
	d2.coreDataSet(core);
	Normaliser norm2;
	norm2.calcAndNormaliseMaxAbs(d2);
	if (fabs(d2.pattern(1).input()[1] - 0.5) > tol) return -1; // 10000/20000
	return 0;
}

void parseConfAndData(string fname, Config& config, DataTools::CoreDataSet& trnData,
                      DataTools::CoreDataSet& tstData) {
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

int main(int argc, char* argv[]) {
	srand(1);

	string fname;
	if (argc > 1)
		fname = string(argv[1]);
	else
		fname = "./config.toml";

	Config config;
	auto trnCoreData = std::make_shared<DataTools::CoreDataSet>();
	auto tstCoreData = std::make_shared<DataTools::CoreDataSet>();
	parseConfAndData(fname, config, *trnCoreData, *tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	if (testMaxAbs() != 0) return -1;
	return testNormaliser(trnData);
}
