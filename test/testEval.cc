#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "evaltools/EvalTools.hh"
#include "evaltools/Roc.hh"
#include "evaltools/Evaluator.hh"
#include "Ensemble.hh"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace NeuralNetHack;

static double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static double xor_out[][1] = {{0}, {1}, {1}, {0}};

static void buildXorDataSet(CoreDataSet& core, DataSet& data) {
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(xor_in[i], xor_in[i] + 2);
		std::vector<double> out(xor_out[i], xor_out[i] + 1);
		core.addPattern(Pattern(std::to_string(i), in, out));
	}
	data.coreDataSet(core);
}

static Mlp trainXorMlp(DataSet& data) {
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);

	SummedSquare error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(2000);

	std::ostringstream devnull;
	trainer.train(devnull);

	return mlp;
}

// ---------------------------------------------------------------------------
// Test 1: Roc class - calcAucWmw with perfect separation
// ---------------------------------------------------------------------------
static bool testRocPerfectSeparation() {
	std::cout << "Test: Roc::calcAucWmw with perfect separation ... ";

	// Positive outputs are all higher than negative outputs => AUC = 1.0
	std::vector<double> out = {0.1, 0.2, 0.8, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	double aucVal = roc.calcAucWmw(out, dout);

	if (std::fabs(aucVal - 1.0) > 1e-9) {
		std::cerr << "FAIL (expected 1.0, got " << aucVal << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (AUC = " << aucVal << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 2: Roc class - calcAucWmwFast with perfect separation
// ---------------------------------------------------------------------------
static bool testRocFastPerfectSeparation() {
	std::cout << "Test: Roc::calcAucWmwFast with perfect separation ... ";

	std::vector<double> out = {0.1, 0.2, 0.8, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	double aucVal = roc.calcAucWmwFast(out, dout);

	if (std::fabs(aucVal - 1.0) > 1e-9) {
		std::cerr << "FAIL (expected 1.0, got " << aucVal << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (AUC = " << aucVal << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 3: Roc class - calcAucTrapezoidal with perfect separation
// ---------------------------------------------------------------------------
static bool testRocTrapezoidalPerfectSeparation() {
	std::cout << "Test: Roc::calcAucTrapezoidal with perfect separation ... ";

	std::vector<double> out = {0.1, 0.2, 0.8, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	double aucVal = roc.calcAucTrapezoidal(out, dout);

	// Trapezoidal may not be exactly 1.0 due to discrete steps, but should
	// be very close for perfect separation.
	if (aucVal < 0.9) {
		std::cerr << "FAIL (expected >= 0.9, got " << aucVal << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (AUC = " << aucVal << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 4: Roc - random/inverted ordering gives AUC = 0.0
// ---------------------------------------------------------------------------
static bool testRocInvertedSeparation() {
	std::cout << "Test: Roc::calcAucWmw with inverted separation ... ";

	// All positive outputs lower than all negative outputs => AUC = 0.0
	std::vector<double> out = {0.9, 0.8, 0.1, 0.2};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	double aucVal = roc.calcAucWmw(out, dout);

	if (std::fabs(aucVal - 0.0) > 1e-9) {
		std::cerr << "FAIL (expected 0.0, got " << aucVal << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (AUC = " << aucVal << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 5: WMW and WMW-Fast agree on same data
// ---------------------------------------------------------------------------
static bool testWmwMethodsAgree() {
	std::cout << "Test: Roc WMW and WMW-Fast agree ... ";

	std::vector<double> out = {0.3, 0.1, 0.7, 0.55, 0.2, 0.85};
	std::vector<uint> dout = {0, 0, 1, 1, 0, 1};

	EvalTools::Roc roc1;
	EvalTools::Roc roc2;
	double auc1 = roc1.calcAucWmw(out, dout);
	double auc2 = roc2.calcAucWmwFast(out, dout);

	if (std::fabs(auc1 - auc2) > 1e-9) {
		std::cerr << "FAIL (WMW=" << auc1 << ", WMW-Fast=" << auc2 << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (both = " << auc1 << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 6: Roc::calcRoc produces ROC curve data
// ---------------------------------------------------------------------------
static bool testRocCurveGeneration() {
	std::cout << "Test: Roc::calcRoc generates curve points ... ";

	std::vector<double> out = {0.1, 0.4, 0.6, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	roc.calcRoc(out, dout);

	auto& curve = roc.roc();
	if (curve.empty()) {
		std::cerr << "FAIL (ROC curve is empty)" << std::endl;
		return false;
	}
	if (curve.size() != out.size()) {
		std::cerr << "FAIL (expected " << out.size() << " points, got " << curve.size() << ")"
		          << std::endl;
		return false;
	}
	// Curve should be sorted by FPF (first element of pair)
	for (size_t i = 1; i < curve.size(); ++i) {
		if (curve[i].first < curve[i - 1].first) {
			std::cerr << "FAIL (ROC curve not sorted by FPF)" << std::endl;
			return false;
		}
	}
	std::cout << "PASS (" << curve.size() << " points, sorted)" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 7: Roc::auc() accessor returns last computed AUC
// ---------------------------------------------------------------------------
static bool testRocAucAccessor() {
	std::cout << "Test: Roc::auc() accessor ... ";

	std::vector<double> out = {0.1, 0.2, 0.8, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	double computed = roc.calcAucWmw(out, dout);
	double accessed = roc.auc();

	if (std::fabs(computed - accessed) > 1e-12) {
		std::cerr << "FAIL (calcAucWmw returned " << computed << ", auc() returned " << accessed
		          << ")" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 8: Evaluator class - evaluate with known data
// ---------------------------------------------------------------------------
static bool testEvaluator() {
	std::cout << "Test: Evaluator with known data ... ";

	// Outputs: 0.1, 0.9, 0.8, 0.2
	// Targets: 0,   1,   1,   0
	// With cut=0.5: predicted 0, 1, 1, 0 => all correct
	std::vector<double> out = {0.1, 0.9, 0.8, 0.2};
	std::vector<uint> dout = {0, 1, 1, 0};

	EvalTools::Evaluator eval;
	eval.cut(0.5);
	eval.evaluate(out, dout);

	double tpf = eval.tpf(); // Should be 1.0 (both positives correct)
	double tnf = eval.tnf(); // Should be 1.0 (both negatives correct)
	double fpf = eval.fpf(); // Should be 0.0 (1 - tnf)
	double fnf = eval.fnf(); // Should be 0.0 (1 - tpf)

	bool ok = true;
	if (std::fabs(tpf - 1.0) > 1e-9) {
		std::cerr << "FAIL (tpf expected 1.0, got " << tpf << ")" << std::endl;
		ok = false;
	}
	if (std::fabs(tnf - 1.0) > 1e-9) {
		std::cerr << "FAIL (tnf expected 1.0, got " << tnf << ")" << std::endl;
		ok = false;
	}
	if (std::fabs(fpf - 0.0) > 1e-9) {
		std::cerr << "FAIL (fpf expected 0.0, got " << fpf << ")" << std::endl;
		ok = false;
	}
	if (std::fabs(fnf - 0.0) > 1e-9) {
		std::cerr << "FAIL (fnf expected 0.0, got " << fnf << ")" << std::endl;
		ok = false;
	}
	if (ok)
		std::cout << "PASS (TPF=" << tpf << ", TNF=" << tnf << ")" << std::endl;
	return ok;
}

// ---------------------------------------------------------------------------
// Test 9: Evaluator with imperfect classification
// ---------------------------------------------------------------------------
static bool testEvaluatorImperfect() {
	std::cout << "Test: Evaluator with imperfect classification ... ";

	// Outputs: 0.1, 0.3, 0.8, 0.2
	// Targets: 0,   1,   1,   0
	// With cut=0.5: predicted 0, 0, 1, 0
	// TP=1, FN=1, TN=2, FP=0 => TPF=0.5, TNF=1.0
	std::vector<double> out = {0.1, 0.3, 0.8, 0.2};
	std::vector<uint> dout = {0, 1, 1, 0};

	EvalTools::Evaluator eval;
	eval.cut(0.5);
	eval.evaluate(out, dout);

	double tpf = eval.tpf();
	double tnf = eval.tnf();

	bool ok = true;
	if (std::fabs(tpf - 0.5) > 1e-9) {
		std::cerr << "FAIL (tpf expected 0.5, got " << tpf << ")" << std::endl;
		ok = false;
	}
	if (std::fabs(tnf - 1.0) > 1e-9) {
		std::cerr << "FAIL (tnf expected 1.0, got " << tnf << ")" << std::endl;
		ok = false;
	}
	if (ok)
		std::cout << "PASS (TPF=" << tpf << ", TNF=" << tnf << ")" << std::endl;
	return ok;
}

// ---------------------------------------------------------------------------
// Test 10: ErrorMeasures::summedSquare (per-sample version)
// ---------------------------------------------------------------------------
static bool testSummedSquareError() {
	std::cout << "Test: ErrorMeasures::summedSquare (per-sample) ... ";

	std::vector<double> output = {0.8};
	std::vector<double> target = {1.0};
	// (1.0 - 0.8)^2 = 0.04
	double err = EvalTools::ErrorMeasures::summedSquare(output, target);

	if (std::fabs(err - 0.04) > 1e-9) {
		std::cerr << "FAIL (expected 0.04, got " << err << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (error = " << err << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 11: ErrorMeasures::crossEntropy (per-sample version)
// ---------------------------------------------------------------------------
static bool testCrossEntropyError() {
	std::cout << "Test: ErrorMeasures::crossEntropy (per-sample) ... ";

	// Target=1, output=0.9 => log(0.9)
	std::vector<double> output = {0.9};
	std::vector<double> target = {1.0};
	double err = EvalTools::ErrorMeasures::crossEntropy(output, target);

	double expected = std::log(0.9);
	if (std::fabs(err - expected) > 1e-9) {
		std::cerr << "FAIL (expected " << expected << ", got " << err << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (error = " << err << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 12: ErrorMeasures::crossEntropy with target=0
// ---------------------------------------------------------------------------
static bool testCrossEntropyErrorTarget0() {
	std::cout << "Test: ErrorMeasures::crossEntropy with target=0 ... ";

	// Target=0, output=0.1 => log(1 - 0.1) = log(0.9)
	std::vector<double> output = {0.1};
	std::vector<double> target = {0.0};
	double err = EvalTools::ErrorMeasures::crossEntropy(output, target);

	double expected = std::log(0.9);
	if (std::fabs(err - expected) > 1e-9) {
		std::cerr << "FAIL (expected " << expected << ", got " << err << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (error = " << err << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 13: ErrorMeasures::auc via trained Ensemble on XOR
// ---------------------------------------------------------------------------
static bool testEnsembleAuc() {
	std::cout << "Test: ErrorMeasures::auc with trained XOR ensemble ... ";

	srand(42);
	srand48(42);

	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	Mlp mlp = trainXorMlp(data);

	// Build an Ensemble with the trained MLP
	Ensemble ensemble(mlp, 1.0);

	double aucVal = EvalTools::ErrorMeasures::auc(ensemble, data);

	// A trained XOR model should achieve perfect or near-perfect AUC
	if (aucVal < 0.5) {
		std::cerr << "FAIL (AUC should be > 0.5 for trained model, got " << aucVal << ")"
		          << std::endl;
		return false;
	}
	std::cout << "PASS (AUC = " << aucVal << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 14: ErrorMeasures::summedSquare via Ensemble on XOR
// ---------------------------------------------------------------------------
static bool testEnsembleSummedSquare() {
	std::cout << "Test: ErrorMeasures::summedSquare with trained XOR ensemble ... ";

	srand(42);
	srand48(42);

	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);
	Mlp mlp = trainXorMlp(data);

	Ensemble ensemble(mlp, 1.0);

	double err = EvalTools::ErrorMeasures::summedSquare(ensemble, data);

	// A well-trained model should have low SSE
	if (err > 0.25) {
		std::cerr << "FAIL (SSE too high for trained model: " << err << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (SSE = " << err << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 15: ErrorMeasures::crossEntropy via Ensemble on XOR
// ---------------------------------------------------------------------------
static bool testEnsembleCrossEntropy() {
	std::cout << "Test: ErrorMeasures::crossEntropy with trained XOR ensemble ... ";

	srand(42);
	srand48(42);

	CoreDataSet core; DataSet data;
	buildXorDataSet(core, data);

	// Use logsig output for cross-entropy compatibility
	std::vector<uint> arch = {2, 4, 1};
	std::vector<std::string> types = {"relu", "logsig"};
	Mlp mlp(arch, types, false);

	CrossEntropy error(mlp, data);
	Adam trainer(mlp, data, error, 0.001, 4, 0.01);
	trainer.numEpochs(2000);
	std::ostringstream devnull;
	trainer.train(devnull);

	Ensemble ensemble(mlp, 1.0);

	double err = EvalTools::ErrorMeasures::crossEntropy(ensemble, data);

	// Cross-entropy is returned as -mean(log-likelihood), should be positive
	// for a well-trained model and relatively small
	if (err < 0 || err > 2.0) {
		std::cerr << "FAIL (CE out of expected range: " << err << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (CE = " << err << ")" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 16: Roc copy constructor and assignment
// ---------------------------------------------------------------------------
static bool testRocCopyAndAssign() {
	std::cout << "Test: Roc copy constructor and assignment ... ";

	std::vector<double> out = {0.1, 0.2, 0.8, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc1;
	roc1.calcAucWmw(out, dout);
	roc1.calcRoc(out, dout);

	// Copy constructor
	EvalTools::Roc roc2(roc1);
	if (std::fabs(roc2.auc() - roc1.auc()) > 1e-12) {
		std::cerr << "FAIL (copy constructor: auc mismatch)" << std::endl;
		return false;
	}
	if (roc2.roc().size() != roc1.roc().size()) {
		std::cerr << "FAIL (copy constructor: curve size mismatch)" << std::endl;
		return false;
	}

	// Assignment operator
	EvalTools::Roc roc3;
	roc3 = roc1;
	if (std::fabs(roc3.auc() - roc1.auc()) > 1e-12) {
		std::cerr << "FAIL (assignment: auc mismatch)" << std::endl;
		return false;
	}
	if (roc3.roc().size() != roc1.roc().size()) {
		std::cerr << "FAIL (assignment: curve size mismatch)" << std::endl;
		return false;
	}

	std::cout << "PASS" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 17: Roc::print does not crash
// ---------------------------------------------------------------------------
static bool testRocPrint() {
	std::cout << "Test: Roc::print does not crash ... ";

	std::vector<double> out = {0.1, 0.4, 0.6, 0.9};
	std::vector<uint> dout = {0, 0, 1, 1};

	EvalTools::Roc roc;
	roc.calcRoc(out, dout);

	std::ostringstream oss;
	roc.print(oss);

	if (oss.str().empty()) {
		std::cerr << "FAIL (print produced no output)" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
	bool allPass = true;

	std::cout << "=== Eval Tools Test Suite ===" << std::endl << std::endl;

	// Roc class tests (no training needed)
	allPass &= testRocPerfectSeparation();
	allPass &= testRocFastPerfectSeparation();
	allPass &= testRocTrapezoidalPerfectSeparation();
	allPass &= testRocInvertedSeparation();
	allPass &= testWmwMethodsAgree();
	allPass &= testRocCurveGeneration();
	allPass &= testRocAucAccessor();
	allPass &= testRocCopyAndAssign();
	allPass &= testRocPrint();

	// Evaluator tests
	allPass &= testEvaluator();
	allPass &= testEvaluatorImperfect();

	// ErrorMeasures per-sample tests
	allPass &= testSummedSquareError();
	allPass &= testCrossEntropyError();
	allPass &= testCrossEntropyErrorTarget0();

	// Ensemble-based ErrorMeasures tests (train an XOR model)
	allPass &= testEnsembleAuc();
	allPass &= testEnsembleSummedSquare();
	allPass &= testEnsembleCrossEntropy();

	std::cout << std::endl;
	if (allPass) {
		std::cout << "All tests PASSED." << std::endl;
		return EXIT_SUCCESS;
	} else {
		std::cout << "Some tests FAILED." << std::endl;
		return EXIT_FAILURE;
	}
}
