// End-to-end smoke test for the Factory → EnsembleBuilder → ModelEstimator
// pipeline. The point is coverage, not statistical claims: drive the full
// stack with a tiny in-memory Config + dataset and assert that nothing
// crashes and the returned (train, val) error pair is a finite number.

#include "Config.hh"
#include "Ensemble.hh"
#include "EnsembleBuilder.hh"
#include "Factory.hh"
#include "ModelEstimator.hh"
#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "evaltools/EvalTools.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace NeuralNetHack;
using namespace DataTools;

namespace {

// 32-pattern XOR-with-noise: enough rows for K=2 cross-validation in both
// the model-selection and ensemble-building stages without any split
// being empty.
DataSet buildToyData() {
	auto core = std::make_shared<CoreDataSet>();
	uint id = 0;
	for (uint rep = 0; rep < 8; ++rep) {
		double in_[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
		double out_[][1] = {{0}, {1}, {1}, {0}};
		for (int i = 0; i < 4; ++i) {
			std::vector<double> in(in_[i], in_[i] + 2);
			std::vector<double> out(out_[i], out_[i] + 1);
			core->addPattern(Pattern(std::to_string(id++), in, out));
		}
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

Config makeConfig() {
	Config c;
	c.architecture({2, 4, 1});
	c.actFcn({"tansig", "logsig"});
	c.errFcn("kullback");
	c.softmax(false);
	c.minMethod("adam");
	c.maxEpochs(30);
	c.batchSize(8);
	c.adamLearningRate(0.05);
	c.adamBeta1(0.9);
	c.adamBeta2(0.999);
	c.adamEpsilon(1e-8);
	c.adamWeightDecay(0.0);
	c.weightElimOn(false);
	c.weightElimAlpha(0.0);
	c.weightElimW0(1.0);
	c.ensParamDataSelection("bagg");
	c.ensParamN(2);
	c.ensParamK(2);
	c.ensParamSplitMode(true);
	c.ensParamNewWeights(false);
	c.msParamDataSelection("cv");
	c.msParamN(1);
	c.msParamK(2);
	c.msParamSplitMode(true);
	c.seed(13);
	return c;
}

double xentErr(NeuralNetHack::Ensemble& e, DataTools::DataSet& d) {
	return EvalTools::ErrorMeasures::crossEntropy(e, d);
}

bool testFactoryWiring() {
	std::cout << "Test: Factory builds Mlp/Error/Trainer/Ensemble pipeline ... ";
	Config cfg = makeConfig();
	DataSet data = buildToyData();
	auto mlp = Factory::createMlp(cfg);
	if (!mlp || mlp->nLayers() != 2) {
		std::cerr << "FAIL (Mlp not built)\n";
		return false;
	}
	auto err = Factory::createError(cfg, data);
	if (!err) {
		std::cerr << "FAIL (Error not built)\n";
		return false;
	}
	auto trn = Factory::createTrainer(cfg, data);
	if (!trn || trn->numEpochs() != cfg.maxEpochs()) {
		std::cerr << "FAIL (Trainer not built)\n";
		return false;
	}
	std::cout << "PASS\n";
	return true;
}

bool testEnsembleBuilderEndToEnd() {
	std::cout << "Test: EnsembleBuilder builds an ensemble of 2 members ... ";
	srand(13);
	nnh::rand::seed(13);
	Config cfg = makeConfig();
	DataSet data = buildToyData();
	auto eb = Factory::createEnsembleBuilder(cfg, data);
	std::unique_ptr<NeuralNetHack::Ensemble> ens(eb->buildEnsemble());
	if (!ens || ens->size() != cfg.ensParamN()) {
		std::cerr << "FAIL (ensemble size " << (ens ? ens->size() : 0) << " != " << cfg.ensParamN()
		          << ")\n";
		return false;
	}
	const auto y = ens->propagate(std::vector<double>{0.0, 1.0});
	if (y.size() != 1 || !std::isfinite(y[0])) {
		std::cerr << "FAIL (ensemble propagate produced non-finite output)\n";
		return false;
	}
	std::cout << "PASS (size=" << ens->size() << ", y=" << y[0] << ")\n";
	return true;
}

bool testModelEstimatorEndToEnd() {
	std::cout << "Test: ModelEstimator::runAndEstimateModel returns finite (trn,val) ... ";
	srand(17);
	nnh::rand::seed(17);
	Config cfg = makeConfig();
	DataSet data = buildToyData();
	auto me = Factory::createModelEstimator(cfg, data);
	std::unique_ptr<std::pair<double, double>> r(me->runAndEstimateModel(&xentErr));
	if (!r || !std::isfinite(r->first) || !std::isfinite(r->second)) {
		std::cerr << "FAIL (non-finite result)\n";
		return false;
	}
	if (me->sessions().empty()) {
		std::cerr << "FAIL (no sessions recorded)\n";
		return false;
	}
	std::cout << "PASS (trn=" << r->first << ", val=" << r->second
	          << ", sessions=" << me->sessions().size() << ")\n";
	return true;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== EnsembleBuilder Smoke Test Suite ===\n\n";
	allPass &= testFactoryWiring();
	allPass &= testEnsembleBuilderEndToEnd();
	allPass &= testModelEstimatorEndToEnd();
	std::cout << "\n";
	if (allPass) {
		std::cout << "All tests PASSED.\n";
		return EXIT_SUCCESS;
	}
	std::cout << "Some tests FAILED.\n";
	return EXIT_FAILURE;
}
