// Deep MLP regression: residual vs plain.
//
// Trains two networks with identical architecture and identical weight
// init on a synthetic regression task. The only difference is that one
// has skip connections wired in. With saturating tanh activations
// across 12 hidden layers the plain network's gradient vanishes long
// before reaching the early layers and training stalls; the residual
// version reaches a noticeably lower MSE because the skips give the
// gradient a non-vanishing identity path.
//
// Synthetic target:  y = x + 0.3 * sin(5x) + 0.1 * noise,  x in [-3, 3]
//
// The "x +" baseline plus a wiggly correction is the kind of signal
// where identity-skip connections help the most: the residual stack
// only has to learn the correction term on top of what the skips
// already provide.
//
// Build:   cmake --build build --target residual_vs_plain
// Run:     ./build/residual_vs_plain
//
// Side effect: writes a per-checkpoint loss curve to
// `residual_vs_plain.csv` in the current working directory.

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {

DataSet makeSyntheticData(uint n, uint seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> uni(-3.0, 3.0);
	std::normal_distribution<double> noise(0.0, 0.1);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		double x = uni(rng);
		double y = x + 0.3 * std::sin(5.0 * x) + noise(rng);
		std::vector<double> in = {x};
		std::vector<double> out = {y};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);
	return data;
}

std::unique_ptr<Mlp> makeNet(bool residual, uint seed) {
	srand(seed);
	srand48(seed);
	// 12 width-16 hidden layers + 1 output. Tanh saturates, so a plain
	// 12-layer net's gradient vanishes long before reaching layer 0.
	std::vector<uint> arch = {1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1};
	std::vector<std::string> types = {"tansig", "tansig", "tansig",  "tansig", "tansig",
	                                  "tansig", "tansig", "tansig",  "tansig", "tansig",
	                                  "tansig", "tansig", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, false);
	if (residual) {
		// Six residual blocks of two layers each across hidden layers 0..11.
		// Layer 12 is the output (width 1) and can't be a skip target.
		mlp->skipFrom(2, 0);
		mlp->skipFrom(4, 2);
		mlp->skipFrom(6, 4);
		mlp->skipFrom(8, 6);
		mlp->skipFrom(10, 8);
	}
	return mlp;
}

} // namespace

int main() {
	const uint nSamples = 500;
	const uint epochsPerChunk = 50;
	const uint nChunks = 40; // 2000 epochs total
	const uint dataSeed = 1;
	const uint initSeed = 42;

	DataSet data = makeSyntheticData(nSamples, dataSeed);

	auto plain = makeNet(false, initSeed);
	SummedSquare lossPlain(*plain, data);
	Adam optPlain(*plain, data, lossPlain, /*te=*/0.0, /*bs=*/32, /*lr=*/0.005);

	auto resi = makeNet(true, initSeed);
	SummedSquare lossResi(*resi, data);
	Adam optResi(*resi, data, lossResi, /*te=*/0.0, /*bs=*/32, /*lr=*/0.005);

	std::ofstream csv("residual_vs_plain.csv");
	csv << "epoch,plain_mse,residual_mse\n";
	csv << std::fixed << std::setprecision(6);

	for (uint c = 0; c <= nChunks; ++c) {
		double mp = lossPlain.outputError();
		double mr = lossResi.outputError();
		csv << (c * epochsPerChunk) << "," << mp << "," << mr << "\n";

		if (c == nChunks) break;
		std::ostringstream sink;
		optPlain.numEpochs(epochsPerChunk);
		optPlain.train(sink);
		optResi.numEpochs(epochsPerChunk);
		optResi.train(sink);
	}
	csv.close();

	double finalP = lossPlain.outputError();
	double finalR = lossResi.outputError();

	std::cout << std::fixed << std::setprecision(6);
	std::cout << "12-layer tanh MLP regression on y = x + 0.3 sin(5x) + noise ("
	          << nSamples << " samples)\n\n";
	std::cout << "Final training MSE after " << (nChunks * epochsPerChunk) << " epochs:\n";
	std::cout << "  plain     : " << finalP << "\n";
	std::cout << "  residual  : " << finalR << "\n";
	std::cout << "  plain / residual ratio: " << (finalP / std::max(finalR, 1e-12))
	          << "x\n";
	std::cout << "\nLoss curve: residual_vs_plain.csv\n";

	return finalR < finalP ? 0 : 1;
}
