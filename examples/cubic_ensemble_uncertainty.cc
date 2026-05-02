// Residual MLP ensemble: Amini et al. cubic regression benchmark.
//
// Reproduces the toy regression task from Amini et al., "Deep Evidential
// Regression" (NeurIPS 2020): y = x^3 + N(0, 3) with training samples
// drawn uniformly from x in [-4, 4]. We evaluate on x in [-6, 6], so
// the regions [-6, -4) and (4, 6] are out of distribution.
//
// Inside the training range the ensemble members agree (low epistemic
// uncertainty). Outside it they extrapolate to wildly different cubic-
// like surrogates, which shows up as a large spread between members.
// Same setup as residual_ensemble_uncertainty.cc but with the canonical
// Amini cubic instead of the smooth-sine target.
//
// Build:   cmake --build build --target cubic_ensemble_uncertainty
// Run:     ./build/cubic_ensemble_uncertainty
//
// Side effect: writes `cubic_ensemble_uncertainty.csv` in the cwd with
// columns: x, truth, is_ood, m0..m{N-1}, mean, std.

#include "Ensemble.hh"
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
using NeuralNetHack::Ensemble;

namespace {

constexpr double kTrainMin = -4.0;
constexpr double kTrainMax = 4.0;
constexpr double kEvalMin = -6.0;
constexpr double kEvalMax = 6.0;
constexpr double kNoiseStd = 3.0; // matches Amini et al.

double truth(double x) {
	return x * x * x;
}

DataSet makeTrainingData(uint n, uint seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> uni(kTrainMin, kTrainMax);
	std::normal_distribution<double> noise(0.0, kNoiseStd);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		double x = uni(rng);
		double y = truth(x) + noise(rng);
		std::vector<double> in = {x};
		std::vector<double> out = {y};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);
	return data;
}

std::unique_ptr<Mlp> trainMember(DataSet& data, uint seed, uint epochs) {
	srand(seed);
	srand48(seed);
	// Same architecture as residual_ensemble_uncertainty: 5 hidden ReLU
	// layers of width 16 with two residual blocks. Linear output handles
	// the wide ±64 target range without saturation.
	std::vector<uint> arch = {1, 16, 16, 16, 16, 16, 1};
	std::vector<std::string> types = {"relu", "relu", "relu", "relu", "relu", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, false);
	mlp->skipFrom(2, 0);
	mlp->skipFrom(4, 2);

	SummedSquare loss(*mlp, data);
	Adam opt(*mlp, data, loss, /*te=*/0.0, /*bs=*/64, /*lr=*/0.005);
	opt.numEpochs(epochs);
	std::ostringstream sink;
	opt.train(sink);
	return mlp;
}

} // namespace

int main(int argc, char* argv[]) {
	uint nMembers = 7;
	if (argc > 1) {
		int v = std::atoi(argv[1]);
		if (v < 1) {
			std::cerr << "Usage: " << argv[0] << " [n_members]\n";
			return 1;
		}
		nMembers = static_cast<uint>(v);
	}
	const uint nSamples = 1000;
	const uint epochs = 2000;
	const uint nEvalPoints = 241; // dx = 0.05 across [-6, 6]
	const uint dataSeed = 1;
	const uint baseSeed = 100;

	DataSet data = makeTrainingData(nSamples, dataSeed);

	std::cout << "Training " << nMembers
	          << " residual MLPs on Amini cubic (y = x^3 + N(0, 3))\n";
	std::cout << "Training data: " << nSamples << " samples on x in [" << kTrainMin << ", "
	          << kTrainMax << "]\n";
	std::cout << "Evaluating on x in [" << kEvalMin << ", " << kEvalMax << "]\n\n";

	Ensemble ensemble;
	for (uint i = 0; i < nMembers; ++i) {
		std::cout << "  member " << i << "..." << std::flush;
		auto mlp = trainMember(data, baseSeed + i, epochs);
		std::cout << " done\n";
		ensemble.addMlp(std::move(mlp));
	}

	std::ofstream csv("cubic_ensemble_uncertainty.csv");
	csv << std::fixed << std::setprecision(6);
	csv << "x,truth,is_ood";
	for (uint i = 0; i < nMembers; ++i) csv << ",m" << i;
	csv << ",mean,std\n";

	double sumSqErrIn = 0.0, sumSqErrOut = 0.0;
	uint nIn = 0, nOut = 0;

	for (uint i = 0; i < nEvalPoints; ++i) {
		double x = kEvalMin + (kEvalMax - kEvalMin) * static_cast<double>(i) /
		                          static_cast<double>(nEvalPoints - 1);
		bool isOod = (x < kTrainMin || x > kTrainMax);

		std::vector<double> input = {x};
		std::vector<double> preds(nMembers);
		double mean = 0.0;
		for (uint m = 0; m < nMembers; ++m) {
			Mlp& mlp = const_cast<Mlp&>(ensemble.mlp(m));
			preds[m] = mlp.propagate(input)[0];
			mean += preds[m];
		}
		mean /= static_cast<double>(nMembers);

		double var = 0.0;
		for (double p : preds) var += (p - mean) * (p - mean);
		var /= static_cast<double>(nMembers);
		double sd = std::sqrt(var);

		double t = truth(x);
		double err = mean - t;
		if (isOod) {
			sumSqErrOut += err * err;
			++nOut;
		} else {
			sumSqErrIn += err * err;
			++nIn;
		}

		csv << x << "," << t << "," << (isOod ? 1 : 0);
		for (double p : preds) csv << "," << p;
		csv << "," << mean << "," << sd << "\n";
	}
	csv.close();

	double mseIn = sumSqErrIn / nIn;
	double mseOut = sumSqErrOut / nOut;

	std::cout << "\nEnsemble mean MSE vs truth (no noise):\n";
	std::cout << "  in-distribution  (" << nIn << " points): " << mseIn << "\n";
	std::cout << "  out-of-dist      (" << nOut << " points): " << mseOut << "\n";
	std::cout << "  OOD / ID ratio: " << (mseOut / std::max(mseIn, 1e-12)) << "x\n";
	std::cout << "\nCurves: cubic_ensemble_uncertainty.csv\n";
	return 0;
}
