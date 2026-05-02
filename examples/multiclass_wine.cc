// Softmax MLP on the UCI Wine dataset (3 classes, 13 features).
//
// Loads test/wine/wine.{trn,tst}.tab (tab-separated; col 1 = id, cols
// 2-14 = features, cols 15-17 = one-hot target), Z-normalises features,
// trains a softmax + cross-entropy MLP, reports train/test accuracy.
//
// Build:   cmake --build build --target multiclass_wine
// Run:     ./build/multiclass_wine [path-to-test/wine-dir]

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "datatools/Pattern.hh"
#include "parser/Parser.hh"
#include "Random.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/Mlp.hh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using NeuralNetHack::Parser;

namespace {
constexpr uint K = 3;

DataSet loadTab(const std::string& path) {
	std::ifstream in(path);
	if (!in) {
		std::cerr << "cannot open " << path << "\n";
		std::exit(1);
	}
	auto core = std::make_shared<CoreDataSet>();
	std::vector<uint> inCols;
	for (uint c = 2; c <= 14; ++c) inCols.push_back(c);
	std::vector<uint> outCols = {15, 16, 17};
	std::vector<uint> rowRange = {0};
	Parser::readDataFile(in, /*idCol=*/1, inCols, outCols, rowRange, *core);
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

double accuracy(Mlp& mlp, DataSet& ds) {
	uint correct = 0;
	for (uint i = 0; i < ds.size(); ++i) {
		const auto& p = mlp.propagate(ds.pattern(i).input());
		const auto& t = ds.pattern(i).output();
		uint argP = 0, argT = 0;
		for (uint j = 1; j < K; ++j) {
			if (p[j] > p[argP]) argP = j;
			if (t[j] > t[argT]) argT = j;
		}
		if (argP == argT) ++correct;
	}
	return static_cast<double>(correct) / ds.size();
}
} // namespace

int main(int argc, char** argv) {
	const std::string dir = (argc > 1) ? argv[1] : "test/wine";

	srand(7);
	nnh::rand::seed(7);

	DataSet trn = loadTab(dir + "/wine.trn.tab");
	DataSet tst = loadTab(dir + "/wine.tst.tab");

	Normaliser norm;
	norm.calcAndNormalise(trn, true);
	norm.normalise(tst);

	std::vector<uint> arch = {13, 16, K};
	std::vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, /*softmax=*/true);

	CrossEntropy loss(mlp, trn);
	Adam opt(mlp, trn, loss, /*te=*/0.0, /*bs=*/32, /*lr=*/0.01);
	opt.numEpochs(1500);
	std::ostringstream sink;
	opt.train(sink);

	std::cout << "wine (n=" << trn.size() << " trn, " << tst.size() << " tst):\n"
	          << "  train acc = " << accuracy(mlp, trn) << "\n"
	          << "  test acc  = " << accuracy(mlp, tst) << "\n";
	return 0;
}
