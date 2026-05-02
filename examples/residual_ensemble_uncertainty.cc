// Residual MLP ensemble: visualizing in/out-of-distribution uncertainty.
//
// Trains N residual networks on a synthetic 1D regression task where the
// training data is restricted to x in [-3, 3]. At evaluation time we
// sweep x over [-6, 6], so the regions x in [-6, -3) and (3, 6] are
// out of distribution.
//
// Inside the training range the ensemble members agree closely (low
// epistemic uncertainty). Outside it they extrapolate to wildly
// different functions, which shows up as a large spread between members
// — the signature of high epistemic uncertainty in OOD territory.
//
// Build:   cmake --build build --target residual_ensemble_uncertainty
// Run:     ./build/residual_ensemble_uncertainty
//
// Side effect: writes `residual_ensemble_uncertainty.csv` in the cwd
// with columns: x, truth, is_ood, m0..m{N-1}, mean, std.
// Plot truth, each m_i (faint), mean (bold), and shade mean±std to see
// the uncertainty grow on the OOD bands.

#include "Ensemble.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "Random.hh"
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

constexpr double kTrainMin = -3.0;
constexpr double kTrainMax = 3.0;
constexpr double kEvalMin = -6.0;
constexpr double kEvalMax = 6.0;

double truth(double x) {
	return x + 0.3 * std::sin(5.0 * x);
}

DataSet makeTrainingData(uint n, uint seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> uni(kTrainMin, kTrainMax);
	std::normal_distribution<double> noise(0.0, 0.1);
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
	nnh::rand::seed(seed);

	// Modest residual MLP: 5 hidden layers of width 16 with two residual
	// blocks. ReLU hidden so OOD extrapolation is piecewise-linear (and
	// different per random init) rather than saturating at the same
	// boundary value for everybody.
	std::vector<uint> arch = {1, 16, 16, 16, 16, 16, 1};
	std::vector<std::string> types = {"relu", "relu", "relu", "relu", "relu", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, false);
	mlp->skipFrom(2, 0);
	mlp->skipFrom(4, 2);

	SummedSquare loss(*mlp, data);
	Adam opt(*mlp, data, loss, /*te=*/0.0, /*bs=*/32, /*lr=*/0.005);
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
	const uint nSamples = 500;
	const uint epochs = 1500;
	const uint nEvalPoints = 241; // dx = 0.05 across [-6, 6]
	const uint dataSeed = 1;
	const uint baseSeed = 100;

	DataSet data = makeTrainingData(nSamples, dataSeed);

	std::cout << "Training " << nMembers << " residual MLPs (1-16x5-1 with skip 2->0, 4->2)\n";
	std::cout << "Training data: " << nSamples << " samples on x in [" << kTrainMin << ", "
	          << kTrainMax << "]\n";
	std::cout << "Evaluating on x in [" << kEvalMin << ", " << kEvalMax << "]\n\n";

	std::vector<std::unique_ptr<Mlp>> trained(nMembers);
#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < static_cast<int>(nMembers); ++i) {
		trained[i] = trainMember(data, baseSeed + i, epochs);
#pragma omp critical
		std::cout << "  member " << i << " done\n";
	}
	Ensemble ensemble;
	for (auto& mlp : trained) ensemble.addMlp(std::move(mlp));

	std::ofstream csv("residual_ensemble_uncertainty.csv");
	csv << std::fixed << std::setprecision(6);
	csv << "x,truth,is_ood";
	for (uint i = 0; i < nMembers; ++i) csv << ",m" << i;
	csv << ",mean,std\n";

	// Track in-distribution vs OOD MSE for the summary.
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
	std::cout << "\nCurves: residual_ensemble_uncertainty.csv\n";
	std::cout << "Plot suggestion: lines for truth + each m_i (light), heavy line for mean,\n";
	std::cout << "  shaded band for mean ± std, vertical guides at x=" << kTrainMin
	          << " and x=" << kTrainMax << " to mark the training boundary.\n";
	return 0;
}
