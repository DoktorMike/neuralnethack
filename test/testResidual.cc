#include "mlp/Mlp.hh"
#include "mlp/Adam.hh"
#include "mlp/SummedSquare.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"

#include <cmath>
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

#define EXPECT(cond, label)                                                                       \
	do {                                                                                          \
		if (!(cond)) {                                                                            \
			std::cerr << "FAIL: " << label << " (" << __FILE__ << ":" << __LINE__ << ")"          \
			          << std::endl;                                                               \
			++fails;                                                                              \
		}                                                                                         \
	} while (0)

// Test 1: forward pass identity through skip.
// arch = [2, 2, 2], two LinearLayers. Layer 0 = identity, layer 1 = zero.
// Skip 0 -> 1. Output should equal input.
static void testForwardIdentity() {
	vector<uint> arch = {2, 2, 2};
	vector<std::string> types = {"purelin", "purelin"};
	Mlp mlp(arch, types, false);

	// Layer 0: [ncurr x (nprev+1)] = [2 x 3]; rows = neurons, last col = bias.
	// Identity: w[0,0]=1 w[0,1]=0 b=0; w[1,0]=0 w[1,1]=1 b=0.
	auto& w0 = mlp.layer(0).weights();
	w0 = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
	auto& w1 = mlp.layer(1).weights();
	w1.assign(w1.size(), 0.0);

	mlp.skipFrom(1, 0);

	vector<double> in = {3.0, 5.0};
	vector<double> out = mlp.propagate(in);
	EXPECT(out.size() == 2, "output size = 2");
	EXPECT(nearly(out[0], 3.0), "skip identity dim 0");
	EXPECT(nearly(out[1], 5.0), "skip identity dim 1");
}

// Test 2: residual MLP trains XOR.
// arch = [2, 4, 4, 1], skip layer 0 -> layer 1 (both have 4 neurons).
static void testResidualXor() {
	srand(7);
	srand48(7);

	auto core = std::make_shared<CoreDataSet>();
	double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double xor_out[][1] = {{0}, {1}, {1}, {0}};
	for (int i = 0; i < 4; ++i) {
		vector<double> in(xor_in[i], xor_in[i] + 2);
		vector<double> out(xor_out[i], xor_out[i] + 1);
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);

	vector<uint> arch = {2, 4, 4, 1};
	vector<std::string> types = {"tansig", "tansig", "logsig"};
	Mlp mlp(arch, types, false);
	mlp.skipFrom(1, 0); // residual: layer 1 += layer 0 (both width 4)

	SummedSquare error(mlp, data);
	Adam trainer(mlp, data, error, 0.0, 4, 0.05);
	trainer.numEpochs(3000);
	std::ostringstream sink;
	trainer.train(sink);

	double err = 0;
	for (int i = 0; i < 4; ++i) {
		auto pred = mlp.propagate(data.pattern(i).input());
		double diff = pred[0] - data.pattern(i).output()[0];
		err += diff * diff;
	}
	err /= 4.0;
	std::cout << "residual XOR final MSE = " << err << std::endl;
	EXPECT(err < 0.05, "residual XOR converges below MSE 0.05");
}

// Test 2.5: sanity baseline. Same gradient check but WITHOUT a skip — confirms
// the test scaffolding is correct before we point fingers at skip plumbing.
static void testGradientCheckBaseline() {
	srand(13);
	srand48(13);

	auto core = std::make_shared<CoreDataSet>();
	for (int i = 0; i < 8; ++i) {
		vector<double> in = {drand48(), drand48()};
		vector<double> out = {drand48()};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);

	vector<uint> arch = {2, 3, 3, 1};
	vector<std::string> types = {"tansig", "tansig", "purelin"};
	Mlp mlp(arch, types, false);
	// no skip!

	SummedSquare loss(mlp, data);
	loss.gradient();
	vector<double> g_analytic = mlp.gradients();

	// SummedSquare's stored gradient corresponds to loss (1/2N) Σ (t - y)²
	// (the implicit 1/2 cancels the 2 from differentiating the square).
	// outputError() returns the un-halved version, so use the halved form
	// here for a consistent gradient check.
	auto evalLoss = [&]() {
		double s = 0.0;
		for (uint i = 0; i < data.size(); ++i) {
			vector<double> y = mlp.propagate(data.pattern(i).input());
			double d = y[0] - data.pattern(i).output()[0];
			s += d * d;
		}
		return 0.5 * s / static_cast<double>(data.size());
	};

	vector<double> w = mlp.weights();
	const double h = 1e-6;
	double maxRel = 0.0;
	for (uint k = 0; k < w.size(); ++k) {
		double saved = w[k];
		w[k] = saved + h; mlp.weights(w);
		double Lp = evalLoss();
		w[k] = saved - h; mlp.weights(w);
		double Lm = evalLoss();
		w[k] = saved; mlp.weights(w);
		double g_num = (Lp - Lm) / (2.0 * h);
		double g_an = g_analytic[k];
		double denom = std::max(1.0, std::fabs(g_num) + std::fabs(g_an));
		double rel = std::fabs(g_num - g_an) / denom;
		if (rel > maxRel) maxRel = rel;
	}
	std::cout << "BASELINE max relative grad error = " << maxRel << std::endl;
	EXPECT(maxRel < 1e-4, "baseline (no skip) gradient check");
}

// Test 3: gradient sanity. Verify analytic gradient (from SummedSquare) matches
// numerical finite-difference gradient on a small skip-network.
static void testGradientCheck() {
	srand(13);
	srand48(13);

	auto core = std::make_shared<CoreDataSet>();
	for (int i = 0; i < 8; ++i) {
		vector<double> in = {drand48(), drand48()};
		vector<double> out = {drand48()};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);

	vector<uint> arch = {2, 3, 3, 1};
	vector<std::string> types = {"tansig", "tansig", "purelin"};
	Mlp mlp(arch, types, false);
	mlp.skipFrom(1, 0); // skip the two width-3 hidden layers

	SummedSquare loss(mlp, data);
	loss.gradient();
	vector<double> g_analytic = mlp.gradients();

	// SummedSquare's stored gradient corresponds to loss (1/2N) Σ (t - y)²
	// (the implicit 1/2 cancels the 2 from differentiating the square).
	// outputError() returns the un-halved version, so use the halved form
	// here for a consistent gradient check.
	auto evalLoss = [&]() {
		double s = 0.0;
		for (uint i = 0; i < data.size(); ++i) {
			vector<double> y = mlp.propagate(data.pattern(i).input());
			double d = y[0] - data.pattern(i).output()[0];
			s += d * d;
		}
		return 0.5 * s / static_cast<double>(data.size());
	};

	vector<double> w = mlp.weights();
	const double h = 1e-6;
	double maxRel = 0.0;
	for (uint k = 0; k < w.size(); ++k) {
		double saved = w[k];
		w[k] = saved + h; mlp.weights(w);
		double Lp = evalLoss();
		w[k] = saved - h; mlp.weights(w);
		double Lm = evalLoss();
		w[k] = saved; mlp.weights(w);
		double g_num = (Lp - Lm) / (2.0 * h);
		double g_an = g_analytic[k];
		double denom = std::max(1.0, std::fabs(g_num) + std::fabs(g_an));
		double rel = std::fabs(g_num - g_an) / denom;
		if (rel > maxRel) maxRel = rel;
	}
	std::cout << "max relative grad error = " << maxRel << std::endl;
	EXPECT(maxRel < 1e-4, "analytic gradient matches numerical gradient");
}

int main() {
	testForwardIdentity();
	testResidualXor();
	testGradientCheckBaseline();
	testGradientCheck();
	if (fails == 0) std::cout << "All residual tests passed." << std::endl;
	return fails == 0 ? 0 : 1;
}
