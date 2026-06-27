#include "Random.hh"
#include "mlp/Mlp.hh"
#include "Ensemble.hh"
#include "evaltools/Uncertainty.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace NeuralNetHack;
using EvalTools::Uncertainty::decomposeEntropy;
using EvalTools::Uncertainty::EntropyDecomposition;
using EvalTools::Uncertainty::predictiveEntropy;

// ---------------------------------------------------------------------------
// Test 1: predictiveEntropy known values
// ---------------------------------------------------------------------------
static bool testEntropyValues() {
	std::cout << "Test: predictiveEntropy known values ... ";
	bool ok = true;

	// Uniform over 2 classes -> ln 2
	double h2 = predictiveEntropy({0.5, 0.5});
	if (std::fabs(h2 - std::log(2.0)) > 1e-12) {
		std::cerr << "FAIL (uniform-2 expected ln2, got " << h2 << ")" << std::endl;
		ok = false;
	}
	// Uniform over 3 classes -> ln 3
	double h3 = predictiveEntropy({1.0 / 3, 1.0 / 3, 1.0 / 3});
	if (std::fabs(h3 - std::log(3.0)) > 1e-12) {
		std::cerr << "FAIL (uniform-3 expected ln3, got " << h3 << ")" << std::endl;
		ok = false;
	}
	// One-hot -> 0
	double h0 = predictiveEntropy({1.0, 0.0, 0.0});
	if (std::fabs(h0) > 1e-12) {
		std::cerr << "FAIL (one-hot expected 0, got " << h0 << ")" << std::endl;
		ok = false;
	}
	if (ok) std::cout << "PASS" << std::endl;
	return ok;
}

// ---------------------------------------------------------------------------
// Test 2: members in full agreement -> epistemic == 0, total == aleatoric
// ---------------------------------------------------------------------------
static bool testAgreementZeroEpistemic() {
	std::cout << "Test: agreeing members have zero epistemic ... ";
	std::vector<std::vector<double>> probs = {{0.7, 0.3}, {0.7, 0.3}, {0.7, 0.3}};
	EntropyDecomposition d = decomposeEntropy(probs);

	bool ok = true;
	if (std::fabs(d.epistemic) > 1e-12) {
		std::cerr << "FAIL (epistemic expected 0, got " << d.epistemic << ")" << std::endl;
		ok = false;
	}
	if (std::fabs(d.total - d.aleatoric) > 1e-12) {
		std::cerr << "FAIL (total != aleatoric)" << std::endl;
		ok = false;
	}
	if (ok) std::cout << "PASS (total=" << d.total << ", epi=" << d.epistemic << ")" << std::endl;
	return ok;
}

// ---------------------------------------------------------------------------
// Test 3: confident-but-disagreeing members -> high epistemic, low aleatoric
// ---------------------------------------------------------------------------
static bool testDisagreementEpistemic() {
	std::cout << "Test: disagreeing confident members have high epistemic ... ";
	// Each member is near one-hot (low aleatoric) but they point at different
	// classes, so the mean is near-uniform (high total) -> epistemic dominates.
	std::vector<std::vector<double>> probs = {{0.99, 0.01}, {0.01, 0.99}};
	EntropyDecomposition d = decomposeEntropy(probs);

	bool ok = true;
	if (!(d.epistemic > d.aleatoric)) {
		std::cerr << "FAIL (epistemic " << d.epistemic << " should exceed aleatoric " << d.aleatoric
		          << ")" << std::endl;
		ok = false;
	}
	if (!(d.total >= d.aleatoric - 1e-12) || d.epistemic < -1e-12) {
		std::cerr << "FAIL (decomposition invariants violated)" << std::endl;
		ok = false;
	}
	if (ok)
		std::cout << "PASS (total=" << d.total << ", ale=" << d.aleatoric << ", epi=" << d.epistemic
		          << ")" << std::endl;
	return ok;
}

// ---------------------------------------------------------------------------
// Test 4: invariant total >= aleatoric and epistemic >= 0 on random probs
// ---------------------------------------------------------------------------
static bool testInvariants() {
	std::cout << "Test: total >= aleatoric, epistemic >= 0 ... ";
	std::vector<std::vector<double>> probs = {
	    {0.6, 0.4}, {0.3, 0.7}, {0.5, 0.5}, {0.8, 0.2}};
	EntropyDecomposition d = decomposeEntropy(probs);
	if (d.total < d.aleatoric - 1e-12 || d.epistemic < -1e-12) {
		std::cerr << "FAIL (total=" << d.total << ", ale=" << d.aleatoric << ", epi=" << d.epistemic
		          << ")" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

// ---------------------------------------------------------------------------
// Test 5: Ensemble overload, binary sigmoid members -> {1-p, p}
// ---------------------------------------------------------------------------
static bool testEnsembleOverload() {
	std::cout << "Test: Ensemble overload with sigmoid members ... ";
	nnh::rand::seed(3);

	// Two untrained single-output nets give different outputs -> some epistemic.
	Mlp a({2, 3, 1}, {"tansig", "logsig"}, false);
	Mlp b({2, 3, 1}, {"tansig", "logsig"}, false);
	Ensemble ens(a, 1.0);
	ens.addMlp(b, 1.0);

	std::vector<double> x = {0.4, -0.2};
	EntropyDecomposition d = decomposeEntropy(ens, x);

	bool ok = (d.total >= d.aleatoric - 1e-12) && d.epistemic >= -1e-12 && d.total >= 0.0;
	if (!ok) {
		std::cerr << "FAIL (total=" << d.total << ", ale=" << d.aleatoric << ", epi=" << d.epistemic
		          << ")" << std::endl;
		return false;
	}
	std::cout << "PASS (total=" << d.total << ", epi=" << d.epistemic << ")" << std::endl;
	return true;
}

int main() {
	bool allPass = true;
	std::cout << "=== Uncertainty Test Suite ===" << std::endl << std::endl;

	allPass &= testEntropyValues();
	allPass &= testAgreementZeroEpistemic();
	allPass &= testDisagreementEpistemic();
	allPass &= testInvariants();
	allPass &= testEnsembleOverload();

	std::cout << std::endl;
	if (allPass) {
		std::cout << "All tests PASSED." << std::endl;
		return EXIT_SUCCESS;
	}
	std::cout << "Some tests FAILED." << std::endl;
	return EXIT_FAILURE;
}
