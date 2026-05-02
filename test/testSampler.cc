#include <datatools/CrossSplitSampler.hh>
#include <datatools/BootstrapSampler.hh>
#include <datatools/HoldOutSampler.hh>
#include <parser/Parser.hh>
#include <Config.hh>

#include <cassert>
#include <fstream>
#include <memory>
#include <utility>

using namespace NeuralNetHack;
using namespace std;

void parseConfAndData(const string fname, Config& config, DataTools::CoreDataSet& trnData,
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

template <class S> static bool drainSampler(S& sampler, uint expected) {
	uint cntr = 0;
	while (sampler.hasNext()) {
		++cntr;
		auto trnVal = sampler.next();
		(void)trnVal;
	}
	if (cntr != expected) return false;
	cntr = 0;
	sampler.reset();
	while (sampler.hasNext()) {
		++cntr;
		auto trnVal = sampler.next();
		(void)trnVal;
	}
	return cntr == expected;
}

int main() {
	Config config;
	auto trnCoreData = std::make_shared<DataTools::CoreDataSet>();
	auto tstCoreData = std::make_shared<DataTools::CoreDataSet>();
	parseConfAndData("./config.toml", config, *trnCoreData, *tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);

	bool ok = true;

	DataTools::CrossSplitSampler cs(trnData, 3, 5);
	if (!drainSampler(cs, 15)) ok = false;

	DataTools::HoldOutSampler ho(trnData, 0.2, 15);
	if (!drainSampler(ho, 15)) ok = false;

	DataTools::BootstrapSampler bs(trnData, 15);
	if (!drainSampler(bs, 15)) ok = false;

	return ok ? 0 : -1;
}
