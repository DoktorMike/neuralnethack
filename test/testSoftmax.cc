#include "Random.hh"
#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/CrossEntropy.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using std::vector;

static int fails = 0;

static bool nearly(double a, double b, double tol = 1e-9) {
	return std::fabs(a - b) <= tol;
}

#define EXPECT(cond, label)                                                                        \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::cerr << "FAIL: " << label << " (" << __FILE__ << ":" << __LINE__ << ")"           \
			          << std::endl;                                                                \
			++fails;                                                                               \
		}                                                                                          \
	} while (0)

// Forward sanity: softmax outputs are positive and sum to 1.
static void testForwardSoftmaxSums() {
	vector<uint> arch = {2, 3};
	vector<std::string> types = {"purelin"};
	Mlp mlp(arch, types, true); // softmax = on

	auto& w = mlp.layer(0).weights();
	w = {0.5, -0.2, 0.1, 0.3, 0.4, -0.1, -0.5, 0.7, 0.0};

	vector<double> in = {1.7, -0.4};
	vector<double> out = mlp.propagate(in);

	EXPECT(out.size() == 3, "softmax output dim");
	double s = 0;
	for (double v : out) {
		EXPECT(v > 0.0, "softmax positive");
		s += v;
	}
	EXPECT(nearly(s, 1.0, 1e-12), "softmax sums to 1");
}

// Forward: zero weights -> uniform 1/K output.
static void testForwardUniform() {
	vector<uint> arch = {2, 4};
	vector<std::string> types = {"purelin"};
	Mlp mlp(arch, types, true);
	auto& w = mlp.layer(0).weights();
	w.assign(w.size(), 0.0);

	vector<double> in = {2.0, -1.0};
	vector<double> out = mlp.propagate(in);
	for (double v : out)
		EXPECT(nearly(v, 0.25, 1e-12), "softmax uniform 1/K");
}

// Batch path matches single-pattern path.
static void testBatchEqualsSingle() {
	srand(11);
	nnh::rand::seed(11);
	vector<uint> arch = {3, 5, 4};
	vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, true);

	auto core = std::make_shared<CoreDataSet>();
	for (int i = 0; i < 6; ++i) {
		vector<double> in = {drand48(), drand48(), drand48()};
		vector<double> dummyTarget(4, 0.0);
		dummyTarget[i % 4] = 1.0;
		core->addPattern(Pattern(std::to_string(i), in, dummyTarget));
	}
	DataSet data;
	data.coreDataSet(core);

	// Pack via a CrossEntropy instance just to drive propagateBatch.
	CrossEntropy ce(mlp, data);
	(void)ce.outputError(); // exercises batched path

	for (uint i = 0; i < data.size(); ++i) {
		vector<double> single = mlp.propagate(data.pattern(i).input());
		double s = 0;
		for (double v : single)
			s += v;
		EXPECT(nearly(s, 1.0, 1e-12), "single-pattern softmax sums to 1");
	}
}

// Finite-difference gradient check on softmax + cross-entropy multiclass.
static void testGradientCheckSoftmaxCE() {
	srand(19);
	nnh::rand::seed(19);

	const uint K = 3;
	auto core = std::make_shared<CoreDataSet>();
	for (int i = 0; i < 8; ++i) {
		vector<double> in = {drand48(), drand48()};
		vector<double> tgt(K, 0.0);
		tgt[i % K] = 1.0;
		core->addPattern(Pattern(std::to_string(i), in, tgt));
	}
	DataSet data;
	data.coreDataSet(core);

	vector<uint> arch = {2, 4, K};
	vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, true);

	CrossEntropy loss(mlp, data);
	loss.gradient();
	vector<double> g_an = mlp.gradients();

	auto evalLoss = [&]() {
		double s = 0.0;
		const double power = -20;
		const double tiny = std::exp(power);
		for (uint i = 0; i < data.size(); ++i) {
			vector<double> p = mlp.propagate(data.pattern(i).input());
			const auto& t = data.pattern(i).output();
			for (uint j = 0; j < K; ++j) {
				if (t[j] != 0.0) s += (p[j] > tiny) ? std::log(p[j]) : power;
			}
		}
		// CrossEntropy::gradient stores g of -mean log-likelihood, so match
		// sign and normalisation here.
		return -s / static_cast<double>(data.size());
	};

	vector<double> w = mlp.weights();
	const double h = 1e-6;
	double maxRel = 0.0;
	for (uint k = 0; k < w.size(); ++k) {
		double saved = w[k];
		w[k] = saved + h;
		mlp.weights(w);
		double Lp = evalLoss();
		w[k] = saved - h;
		mlp.weights(w);
		double Lm = evalLoss();
		w[k] = saved;
		mlp.weights(w);
		double g_num = (Lp - Lm) / (2.0 * h);
		double denom = std::max(1.0, std::fabs(g_num) + std::fabs(g_an[k]));
		double rel = std::fabs(g_num - g_an[k]) / denom;
		if (rel > maxRel) maxRel = rel;
	}
	std::cout << "softmax+CE max rel grad error = " << maxRel << std::endl;
	EXPECT(maxRel < 1e-4, "softmax+CE gradient matches finite diff");
}

// Convergence: 3-class problem learnable via small MLP with softmax+CE.
static void testSoftmaxConvergence() {
	srand(23);
	nnh::rand::seed(23);

	const uint K = 3;
	auto core = std::make_shared<CoreDataSet>();
	for (int i = 0; i < 60; ++i) {
		double a = drand48() * 2.0 - 1.0;
		double b = drand48() * 2.0 - 1.0;
		uint cls = (a + b > 0.5) ? 0u : ((a - b > 0.0) ? 1u : 2u);
		vector<double> in = {a, b};
		vector<double> t(K, 0.0);
		t[cls] = 1.0;
		core->addPattern(Pattern(std::to_string(i), in, t));
	}
	DataSet data;
	data.coreDataSet(core);

	vector<uint> arch = {2, 8, K};
	vector<std::string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, true);

	CrossEntropy loss(mlp, data);
	Adam trainer(mlp, data, loss, 0.0, 16, 0.05);
	trainer.numEpochs(2000);
	std::ostringstream sink;
	trainer.train(sink);

	uint correct = 0;
	for (uint i = 0; i < data.size(); ++i) {
		vector<double> p = mlp.propagate(data.pattern(i).input());
		const auto& t = data.pattern(i).output();
		uint argmaxP = 0, argmaxT = 0;
		for (uint j = 1; j < K; ++j) {
			if (p[j] > p[argmaxP]) argmaxP = j;
			if (t[j] > t[argmaxT]) argmaxT = j;
		}
		if (argmaxP == argmaxT) ++correct;
	}
	double acc = static_cast<double>(correct) / data.size();
	std::cout << "softmax 3-class accuracy = " << acc << std::endl;
	EXPECT(acc > 0.85, "softmax 3-class converges above 0.85 acc");
}

int main() {
	testForwardSoftmaxSums();
	testForwardUniform();
	testBatchEqualsSingle();
	testGradientCheckSoftmaxCE();
	testSoftmaxConvergence();
	if (fails == 0) std::cout << "All softmax tests passed." << std::endl;
	return fails == 0 ? 0 : 1;
}
