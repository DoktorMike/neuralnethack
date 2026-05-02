// 3-arm Archimedean spiral classification with ensemble uncertainty.
//
// Synthesises a 3-class spiral dataset (each arm is one class), trains an
// ensemble of softmax MLPs, and emits the same CSV layout as
// iris_ensemble_uncertainty.cc so the R plot function can be reused
// directly.
//
// Build:   cmake --build build --target spiral_ensemble_uncertainty
// Run:     ./build/spiral_ensemble_uncertainty [N_members]
//
// Side effects (in cwd):
//   spiral_uncertainty_grid.csv -- x1,x2,p0,p1,p2,
//                                  entropy_total,entropy_aleatoric,
//                                  entropy_epistemic,is_ood
//   spiral_uncertainty_obs.csv  -- x1,x2,true_class,pred_class,set
//
// Plot from R: see scripts/plotexamplesresultdata.r.

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "datatools/Pattern.hh"
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
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;

namespace {
constexpr uint K = 3;
constexpr double kPi = 3.14159265358979323846;

// Generate a 3-arm Archimedean spiral dataset. Each arm k spans theta in
// [theta_min, theta_max] with rotation offset 2*pi*k/K, radius r = b*theta,
// gaussian noise added in (x, y).
DataSet makeSpiral(uint nPerArm, uint seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> uTheta(1.0, 4.0 * kPi);
	std::normal_distribution<double> noise(0.0, 0.25);

	auto core = std::make_shared<CoreDataSet>();
	uint id = 0;
	for (uint k = 0; k < K; ++k) {
		const double offset = 2.0 * kPi * k / K;
		for (uint i = 0; i < nPerArm; ++i) {
			const double theta = uTheta(rng);
			const double r = theta;
			const double x1 = r * std::cos(theta + offset) + noise(rng);
			const double x2 = r * std::sin(theta + offset) + noise(rng);
			std::vector<double> in = {x1, x2};
			std::vector<double> tgt(K, 0.0);
			tgt[k] = 1.0;
			core->addPattern(Pattern(std::to_string(id++), in, tgt));
		}
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

std::unique_ptr<Mlp> trainMember(DataSet& data, uint seed, uint epochs) {
	srand(seed);
	nnh::rand::seed(seed);
	// Spirals need real capacity. Two tansig hidden layers with a residual
	// skip cuts the training time roughly in half vs. plain depth.
	std::vector<uint> arch = {2, 32, 32, 32, K};
	std::vector<std::string> types = {"tansig", "tansig", "tansig", "purelin"};
	auto mlp = std::make_unique<Mlp>(arch, types, /*softmax=*/true);
	mlp->skipFrom(2, 0);
	CrossEntropy loss(*mlp, data);
	Adam opt(*mlp, data, loss, /*te=*/0.0, /*bs=*/64, /*lr=*/0.01);
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
	const uint epochs = 3000;
	const uint baseSeed = 23;

	DataSet trn = makeSpiral(/*nPerArm=*/200, /*seed=*/baseSeed);
	DataSet tst = makeSpiral(/*nPerArm=*/60, /*seed=*/baseSeed + 9999);

	Normaliser norm;
	norm.calcAndNormalise(trn, true);
	norm.normalise(tst);

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
		double aleatoric = 0.0;
		for (auto& m : members) {
			const auto& p = m->propagate(x);
			for (uint k = 0; k < K; ++k) mean[k] += p[k];
			aleatoric += predictiveEntropy(p);
		}
		const double inv = 1.0 / nMembers;
		for (double& v : mean) v *= inv;
		aleatoric *= inv;
		const double total = predictiveEntropy(mean);
		return std::tuple<std::vector<double>, double, double>{mean, total, aleatoric};
	};

	auto ensembleMean = [&](const std::vector<double>& x) {
		return std::get<0>(ensemblePredict(x));
	};

	const double padX = 0.4 * (x1Max - x1Min);
	const double padY = 0.4 * (x2Max - x2Min);
	const double gx0 = x1Min - padX, gx1 = x1Max + padX;
	const double gy0 = x2Min - padY, gy1 = x2Max + padY;
	const uint G = 160;

	{
		std::ofstream csv("spiral_uncertainty_grid.csv");
		csv << std::fixed << std::setprecision(6);
		csv << "x1,x2,p0,p1,p2,entropy_total,entropy_aleatoric,entropy_epistemic,is_ood\n";
		for (uint iy = 0; iy < G; ++iy) {
			const double x2 = gy0 + (gy1 - gy0) * iy / (G - 1);
			for (uint ix = 0; ix < G; ++ix) {
				const double x1 = gx0 + (gx1 - gx0) * ix / (G - 1);
				const auto [p, hTot, hAle] = ensemblePredict({x1, x2});
				const double hEpi = std::max(0.0, hTot - hAle);
				const bool ood = (x1 < x1Min || x1 > x1Max || x2 < x2Min || x2 > x2Max);
				csv << x1 << "," << x2 << "," << p[0] << "," << p[1] << "," << p[2] << "," << hTot
				    << "," << hAle << "," << hEpi << "," << (ood ? 1 : 0) << "\n";
			}
		}
	}

	auto dumpObs = [&](DataSet& ds, const std::string& tag, std::ofstream& csv) {
		for (uint i = 0; i < ds.size(); ++i) {
			const auto& x = ds.pattern(i).input();
			const auto& t = ds.pattern(i).output();
			const auto p = ensembleMean(x);
			csv << x[0] << "," << x[1] << "," << argmax(t) << "," << argmax(p) << "," << tag
			    << "\n";
		}
	};
	{
		std::ofstream csv("spiral_uncertainty_obs.csv");
		csv << std::fixed << std::setprecision(6);
		csv << "x1,x2,true_class,pred_class,set\n";
		dumpObs(trn, "trn", csv);
		dumpObs(tst, "tst", csv);
	}

	uint trnCorrect = 0, tstCorrect = 0;
	for (uint i = 0; i < trn.size(); ++i)
		if (argmax(ensembleMean(trn.pattern(i).input())) == argmax(trn.pattern(i).output()))
			++trnCorrect;
	for (uint i = 0; i < tst.size(); ++i)
		if (argmax(ensembleMean(tst.pattern(i).input())) == argmax(tst.pattern(i).output()))
			++tstCorrect;
	std::cout << "spiral uncertainty (n_members=" << nMembers << ", " << trn.size() << " trn, "
	          << tst.size() << " tst):\n"
	          << "  train acc = " << static_cast<double>(trnCorrect) / trn.size() << "\n"
	          << "  test acc  = " << static_cast<double>(tstCorrect) / tst.size() << "\n"
	          << "Grid: spiral_uncertainty_grid.csv (" << G << "x" << G << ")\n"
	          << "Obs:  spiral_uncertainty_obs.csv\n";
	return 0;
}
