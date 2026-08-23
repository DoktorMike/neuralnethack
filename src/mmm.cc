// mmm: turnkey marketing-mix modeling from a config file.
//
//   mmm config.toml
//
// Reads a traditional MMM table (CSV or whitespace; a header line is
// skipped automatically) with one row per period: media spends,
// covariates, and the KPI. With `adstock.window_raw = true` the raw
// table is expanded into the channel-major lag-window layout
// automatically (DataTools::windowLagged); pre-windowed files work too.
// A missing test file plus `data.holdout_weeks = N` gives a
// chronological split. Inputs are scaled by grouped max-abs and the
// target to unit sd (normalization = "maxabs", strongly recommended --
// see doc/adstock.md).
//
// Training is two-phase per model: warmup with the entropy penalty OFF
// (training.max_epochs), then, when `adstock.harden_epochs > 0` and
// `adstock.entropy_penalty > 0`, a routing-hardening phase at a
// quarter of the learning rate. With `ensemble.runs = M > 1`, M
// bootstrap members train and the report carries percentile bands and
// per-channel assignment stability.
//
// Output: a report to stdout and result.mmm.<suffix>.txt.

#include "Config.hh"
#include "Factory.hh"
#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "datatools/Pattern.hh"
#include "datatools/Windowing.hh"
#include "Ensemble.hh"
#include "evaltools/Uncertainty.hh"
#include "mlp/Adam.hh"
#include "mlp/Adstock.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/Trainer.hh"
#include "parser/Parser.hh"
#include "parser/TomlParser.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace MultiLayerPerceptron;
using std::vector;

namespace {

std::shared_ptr<CoreDataSet> readFile(const std::string& path, int idCol,
                                      const vector<uint>& inCols, const vector<uint>& outCols,
                                      const vector<uint>& rowRange) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("mmm: cannot open " + path);
	auto core = std::make_shared<CoreDataSet>();
	Parser::readDataFile(in, idCol, inCols, outCols, rowRange, *core);
	return core;
}

double r2(Mlp& mlp, DataSet& ds) {
	double sr = 0, st = 0, mn = 0;
	for (uint i = 0; i < ds.size(); ++i)
		mn += ds.pattern(i).output()[0];
	mn /= ds.size();
	for (uint i = 0; i < ds.size(); ++i) {
		Pattern& p = ds.pattern(i);
		const double d = p.output()[0] - mlp.propagate(p.input())[0];
		sr += d * d;
		st += (p.output()[0] - mn) * (p.output()[0] - mn);
	}
	return 1.0 - sr / st;
}

double rmse(Mlp& mlp, DataSet& ds, double targetScale) {
	double sr = 0;
	for (uint i = 0; i < ds.size(); ++i) {
		Pattern& p = ds.pattern(i);
		const double d = p.output()[0] - mlp.propagate(p.input())[0];
		sr += d * d;
	}
	return targetScale * std::sqrt(sr / ds.size());
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: mmm config.toml" << std::endl;
		return 1;
	}

	Config config;
	TomlParser::parseFile(argv[1], config);
	const auto& ap = config.adstock();
	if (!ap.enabled) {
		std::cerr << "mmm: config has no [adstock] section; this tool is adstock-specific."
		          << std::endl;
		return 1;
	}
	nnh::rand::seed(config.seed() == 0 ? 1 : config.seed());

	// ---- Load (and window) the data -------------------------------------
	auto trnRaw = readFile(config.fileName(), config.idColumn(), config.inputColumns(),
	                       config.outputColumns(), config.rowRange());
	std::shared_ptr<CoreDataSet> trnCore = trnRaw, tstCore;

	if (ap.windowRaw) {
		DataSet rawView;
		rawView.coreDataSet(trnRaw);
		trnCore = windowLagged(rawView, ap.channels, ap.lags, ap.passthrough);
	}
	if (!config.fileNameT().empty()) {
		tstCore = readFile(config.fileNameT(), config.idColumnT(), config.inputColumnsT(),
		                   config.outputColumnsT(), config.rowRangeT());
		if (ap.windowRaw) {
			DataSet rawView;
			rawView.coreDataSet(tstCore);
			tstCore = windowLagged(rawView, ap.channels, ap.lags, ap.passthrough);
		}
	}

	DataSet trn, tst;
	if (tstCore) {
		trn.coreDataSet(trnCore);
		tst.coreDataSet(tstCore);
	} else {
		const uint H = config.holdoutWeeks();
		if (H == 0 || H >= trnCore->size()) {
			std::cerr << "mmm: no test file and no usable data.holdout_weeks; set one of them."
			          << std::endl;
			return 1;
		}
		const uint n = trnCore->size();
		vector<uint> trnIdx, tstIdx;
		for (uint i = 0; i < n - H; ++i)
			trnIdx.push_back(i);
		for (uint i = n - H; i < n; ++i)
			tstIdx.push_back(i);
		trn.coreDataSet(trnCore);
		trn.indices(trnIdx);
		tst.coreDataSet(trnCore);
		tst.indices(tstIdx);
	}
	std::printf("mmm: %u train / %u holdout rows, %u channels x %u lags + %u covariates\n",
	            trn.size(), tst.size(), ap.channels, ap.lags, ap.passthrough);

	// ---- Normalise -------------------------------------------------------
	Normaliser norm;
	double targetScale = 1.0;
	if (config.normalization() == "maxabs") {
		norm.calcAndNormaliseMaxAbs(trn, Factory::adstockColumnGroups(config));
		norm.normalise(tst);
		targetScale = norm.stdDev()[trn.nInput()];
	} else if (config.normalization() == "Z") {
		std::cerr << "mmm: normalization = \"Z\" centers the spends and breaks the Hill "
		             "domain; use \"maxabs\"."
		          << std::endl;
		return 1;
	} else {
		std::cout << "mmm: WARNING: no normalization; real-scale spends usually need "
		             "normalization = \"maxabs\" (see doc/adstock.md)"
		          << std::endl;
	}

	// ---- Train (two-phase, optionally an ensemble) ----------------------
	const uint M = std::max(1u, config.ensParamN());
	const bool harden = ap.hardenEpochs > 0 && ap.entropyPenalty > 0.0;
	auto warmupTrainer = Factory::createTrainer(config, trn);
	std::ostringstream sink;

	vector<uint> base = trn.indices();
	if (base.empty()) {
		base.resize(trn.size());
		for (uint i = 0; i < trn.size(); ++i)
			base[i] = i;
	}

	NeuralNetHack::Ensemble ens;
	for (uint m = 0; m < M; ++m) {
		DataSet boot;
		boot.coreDataSet(trnCore);
		vector<uint> idx = base;
		if (M > 1) // bootstrap resample; M == 1 trains on the full data
			for (auto& v : idx)
				v = base[static_cast<uint>(nnh::rand::uniform() * base.size()) % base.size()];
		boot.indices(idx);

		auto member = warmupTrainer->trainNew(boot, sink);
		if (harden && member->adstock()) {
			member->adstock()->entropyPenalty(ap.entropyPenalty);
			SummedSquare loss(*member, boot);
			Adam opt(*member, boot, loss, 0.0, config.batchSize(), config.adamLearningRate() / 4.0,
			         config.adamBeta1(), config.adamBeta2(), config.adamEpsilon(),
			         config.adamWeightDecay());
			opt.numEpochs(ap.hardenEpochs);
			member->training(true);
			opt.train(sink);
			member->training(false);
		}
		std::printf("member %u/%u trained\n", m + 1, M);
		ens.addMlp(std::move(member), 1.0);
	}
	Mlp& model = ens.mlp(0);

	// ---- Report ----------------------------------------------------------
	std::ostringstream rep;
	rep.setf(std::ios::fixed);
	rep.precision(4);
	rep << "=== MMM report (" << config.suffix() << ") ===\n\n";
	rep << "fit: train R^2 " << r2(model, trn) << ", holdout R^2 " << r2(model, tst)
	    << ", holdout RMSE " << rmse(model, tst, targetScale) << " (natural units)\n\n";

	const Adstock* a = model.adstock();
	const uint L = a->nLags();
	auto printKernel = [&](const vector<double>& w) {
		for (uint l = 0; l < std::min(L, 8u); ++l)
			rep << w[l] << " ";
		if (L > 8) rep << "...";
		rep << "\n";
	};

	if (a->boxed() && M > 1) {
		const auto s = EvalTools::Uncertainty::summarizeBoxedAdstock(ens, 0.1);
		rep << "boxes (canonical fast -> slow; kernel mean over " << M << " members):\n";
		for (uint k = 0; k < s.boxes; ++k) {
			rep << "box " << k << ": ";
			printKernel(s.kernelMean[k]);
			if (s.hill)
				rep << "   half-sat " << s.satMean[k] << " [" << s.satLower[k] << ", "
				    << s.satUpper[k] << "], exponent " << s.expMean[k] << " [" << s.expLower[k]
				    << ", " << s.expUpper[k] << "] (normalized-spend scale)\n";
		}
		rep << "\nchannel routing (modal box, stability = fraction of members agreeing):\n";
		for (uint c = 0; c < s.channels; ++c) {
			rep << "channel " << c << ": box " << s.modalBox[c] << "  stability " << s.stability[c];
			if (s.stability[c] < 0.75) rep << "  UNSTABLE -- do not present as known";
			rep << "\n";
		}
	} else if (a->boxed()) {
		rep << "boxes (single fit; run ensemble.runs > 1 for stability bands):\n";
		for (uint k = 0; k < a->nBoxes(); ++k) {
			rep << "box " << k << ": ";
			printKernel(a->boxKernel(k));
			if (a->saturation() == Adstock::Saturation::Hill)
				rep << "   half-sat " << a->boxSaturation(k) << ", exponent "
				    << a->boxHillExponent(k) << " (normalized-spend scale)\n";
		}
		rep << "\nchannel routing:\n";
		const auto assign = a->boxAssignments();
		for (uint c = 0; c < a->nChannels(); ++c) {
			const auto pi = a->routingProbs(c);
			rep << "channel " << c << ": box " << assign[c] << "  (max pi "
			    << *std::max_element(pi.begin(), pi.end()) << ")\n";
		}
	} else {
		rep << "per-channel kernels (natural params):\n";
		const auto nat = a->naturalParams();
		const uint ppk = a->nParamsPerChannel();
		for (uint c = 0; c < a->nChannels(); ++c) {
			rep << "channel " << c << ": ";
			for (uint p = 0; p < ppk; ++p)
				rep << nat[c * ppk + p] << " ";
			printKernel(a->kernelWeights(c));
		}
	}
	if (config.normalization() == "maxabs") {
		rep << "\nreal-unit half-saturation per channel = (normalized half-sat) x (channel max "
		       "spend); channel scales from the normalizer:\n";
		for (uint c = 0; c < a->nChannels(); ++c)
			rep << "channel " << c << ": max spend " << norm.stdDev()[c * L] << "\n";
	}

	std::cout << "\n" << rep.str();
	const std::string outPath = "result.mmm." + config.suffix() + ".txt";
	std::ofstream out(outPath);
	out << rep.str();
	std::cout << "\nreport written to " << outPath << std::endl;
	return 0;
}
