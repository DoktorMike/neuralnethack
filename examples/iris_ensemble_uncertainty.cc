// Iris classification uncertainty: train an ensemble of softmax MLPs on
// the two most discriminative features (petal length, petal width) and
// produce a decision surface coloured by mean class probabilities, with
// intensity scaled by 1 - normalized predictive entropy.
//
// Counterpart to cubic_ensemble_uncertainty.cc but for 3-class classifi-
// cation. The grid extends past the training feature range so the OOD
// regions show up with high entropy (low intensity in the plot).
//
// Build:   cmake --build build --target iris_ensemble_uncertainty
// Run:     ./build/iris_ensemble_uncertainty [N_members]
//
// Side effects (in cwd):
//   iris_uncertainty_grid.csv  -- columns: x1,x2,p0,p1,p2,entropy,is_ood
//   iris_uncertainty_obs.csv   -- columns: x1,x2,true_class,pred_class,set
//
// Plot suggestion (Python):
//   import pandas as pd, numpy as np, matplotlib.pyplot as plt
//   g = pd.read_csv("iris_uncertainty_grid.csv")
//   o = pd.read_csv("iris_uncertainty_obs.csv")
//   N = int(np.sqrt(len(g)))
//   X = g.x1.values.reshape(N, N); Y = g.x2.values.reshape(N, N)
//   rgb = np.stack([g.p0, g.p1, g.p2], axis=1).reshape(N, N, 3)
//   intensity = (1 - g.entropy / np.log(3)).values.reshape(N, N)
//   plt.imshow(rgb * intensity[..., None], extent=[X.min(),X.max(),Y.min(),Y.max()],
//              origin="lower", aspect="auto")
//   for cls, marker in enumerate("oxs"):
//       sel = o.true_class == cls
//       plt.scatter(o.x1[sel], o.x2[sel], marker=marker, c="white",
//                   edgecolor="black", s=40, label=f"class {cls}")
//   plt.xlabel("petal length (z-norm)"); plt.ylabel("petal width (z-norm)")
//   plt.legend(); plt.show()

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "datatools/Pattern.hh"
#include "parser/Parser.hh"
#include "Random.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "mlp/Mlp.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

// Load iris .tab and project to (petal_length, petal_width). Tab column
// layout: 1=id, 2=sepal_len, 3=sepal_wid, 4=petal_len, 5=petal_wid, 6-8=onehot.
DataSet loadPetal(const std::string& path) {
	std::ifstream in(path);
	if (!in) {
		std::cerr << "cannot open " << path << "\n";
		std::exit(1);
	}
	auto core = std::make_shared<CoreDataSet>();
	std::vector<uint> inCols = {4, 5};
	std::vector<uint> outCols = {6, 7, 8};
	std::vector<uint> rowRange = {0};
	Parser::readDataFile(in, /*idCol=*/1, inCols, outCols, rowRange, *core);
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

std::unique_ptr<Mlp> trainMember(DataSet& data, uint seed, uint epochs) {
	srand(seed);
	nnh::rand::seed(seed);
	std::vector<uint> arch = {2, 8, K};
	std::vector<std::string> types = {"tansig", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, /*softmax=*/true);
	CrossEntropy loss(*mlp, data);
	Adam opt(*mlp, data, loss, /*te=*/0.0, /*bs=*/16, /*lr=*/0.02);
	opt.numEpochs(epochs);
	std::ostringstream sink;
	opt.train(sink);
	return mlp;
}

uint argmax(const std::vector<double>& v) {
	uint a = 0;
	for (uint i = 1; i < v.size(); ++i)
		if (v[i] > v[a]) a = i;
	return a;
}

double predictiveEntropy(const std::vector<double>& p) {
	double h = 0;
	for (double q : p)
		if (q > 1e-12) h -= q * std::log(q);
	return h;
}
} // namespace

int main(int argc, char** argv) {
	const uint nMembers = (argc > 1) ? static_cast<uint>(std::atoi(argv[1])) : 10;
	const uint epochs = 1500;
	const uint baseSeed = 17;

	DataSet trn = loadPetal("test/iris/iris.trn.tab");
	DataSet tst = loadPetal("test/iris/iris.tst.tab");

	Normaliser norm;
	norm.calcAndNormalise(trn, true);
	norm.normalise(tst);

	// Training-feature extent in z-space (post-normalisation) for OOD flag.
	double x1Min = 1e9, x1Max = -1e9, x2Min = 1e9, x2Max = -1e9;
	for (uint i = 0; i < trn.size(); ++i) {
		const auto& in = trn.pattern(i).input();
		x1Min = std::min(x1Min, in[0]);
		x1Max = std::max(x1Max, in[0]);
		x2Min = std::min(x2Min, in[1]);
		x2Max = std::max(x2Max, in[1]);
	}

	std::vector<std::unique_ptr<Mlp>> members;
	members.reserve(nMembers);
	for (uint i = 0; i < nMembers; ++i)
		members.push_back(trainMember(trn, baseSeed + i, epochs));

	auto ensemblePredict = [&](const std::vector<double>& x) {
		std::vector<double> mean(K, 0.0);
		for (auto& m : members) {
			const auto& p = m->propagate(x);
			for (uint k = 0; k < K; ++k) mean[k] += p[k];
		}
		const double inv = 1.0 / nMembers;
		for (double& v : mean) v *= inv;
		return mean;
	};

	// Decision-surface grid. Pad past training extent so OOD regions show.
	const double padX = 0.5 * (x1Max - x1Min);
	const double padY = 0.5 * (x2Max - x2Min);
	const double gx0 = x1Min - padX, gx1 = x1Max + padX;
	const double gy0 = x2Min - padY, gy1 = x2Max + padY;
	const uint G = 120;

	{
		std::ofstream csv("iris_uncertainty_grid.csv");
		csv << std::fixed << std::setprecision(6);
		csv << "x1,x2,p0,p1,p2,entropy,is_ood\n";
		for (uint iy = 0; iy < G; ++iy) {
			const double x2 = gy0 + (gy1 - gy0) * iy / (G - 1);
			for (uint ix = 0; ix < G; ++ix) {
				const double x1 = gx0 + (gx1 - gx0) * ix / (G - 1);
				const auto p = ensemblePredict({x1, x2});
				const double h = predictiveEntropy(p);
				const bool ood = (x1 < x1Min || x1 > x1Max || x2 < x2Min || x2 > x2Max);
				csv << x1 << "," << x2 << "," << p[0] << "," << p[1] << "," << p[2] << "," << h
				    << "," << (ood ? 1 : 0) << "\n";
			}
		}
	}

	// Observations: emit train + test with predicted vs true class.
	auto dumpObs = [&](DataSet& ds, const std::string& tag, std::ofstream& csv) {
		for (uint i = 0; i < ds.size(); ++i) {
			const auto& x = ds.pattern(i).input();
			const auto& t = ds.pattern(i).output();
			const auto p = ensemblePredict(x);
			csv << x[0] << "," << x[1] << "," << argmax(t) << "," << argmax(p) << "," << tag
			    << "\n";
		}
	};
	{
		std::ofstream csv("iris_uncertainty_obs.csv");
		csv << std::fixed << std::setprecision(6);
		csv << "x1,x2,true_class,pred_class,set\n";
		dumpObs(trn, "trn", csv);
		dumpObs(tst, "tst", csv);
	}

	// Quick console summary.
	uint trnCorrect = 0, tstCorrect = 0;
	for (uint i = 0; i < trn.size(); ++i) {
		if (argmax(ensemblePredict(trn.pattern(i).input())) == argmax(trn.pattern(i).output()))
			++trnCorrect;
	}
	for (uint i = 0; i < tst.size(); ++i) {
		if (argmax(ensemblePredict(tst.pattern(i).input())) == argmax(tst.pattern(i).output()))
			++tstCorrect;
	}
	std::cout << "iris uncertainty (n_members=" << nMembers << ", 2 features):\n"
	          << "  train acc = " << static_cast<double>(trnCorrect) / trn.size() << "\n"
	          << "  test acc  = " << static_cast<double>(tstCorrect) / tst.size() << "\n"
	          << "Grid: iris_uncertainty_grid.csv (" << G << "x" << G << ")\n"
	          << "Obs:  iris_uncertainty_obs.csv\n";
	return 0;
}
