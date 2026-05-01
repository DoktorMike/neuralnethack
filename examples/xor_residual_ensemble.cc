// Residual MLP ensemble on XOR.
//
// Trains an ensemble of small residual networks on the XOR problem and
// reports per-member predictions plus the ensemble's averaged output.
// Each member is the same architecture (2-4-4-1 with a skip from layer 0
// to layer 1) but starts from a different random init, so the ensemble
// averages across diverse local optima.
//
// Build:   cmake --build build --target xor_residual_ensemble
// Run:     ./build/xor_residual_ensemble

#include "Ensemble.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/Mlp.hh"
#include "mlp/SummedSquare.hh"

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using NeuralNetHack::Ensemble;

namespace {

std::unique_ptr<Mlp> trainResidualXor(DataSet& data, uint seed, uint epochs) {
	srand(seed);
	srand48(seed);

	// 2-4-4-1 with two width-4 hidden layers. Skip from layer 0 to layer 1
	// so the second hidden layer's pre-activation gets the first hidden
	// layer's output added in (residual block).
	std::vector<uint> arch = {2, 4, 4, 1};
	std::vector<std::string> types = {"tansig", "tansig", "logsig"};
	auto mlp = std::make_unique<Mlp>(arch, types, false);
	mlp->skipFrom(1, 0);

	SummedSquare error(*mlp, data);
	Adam trainer(*mlp, data, error, 0.0, 4, 0.05);
	trainer.numEpochs(epochs);
	std::ostringstream sink; // swallow per-member training noise
	trainer.train(sink);
	return mlp;
}

} // namespace

int main() {
	const uint nMembers = 5;
	const uint epochs = 3000;

	// CoreDataSet must outlive the DataSet (DataSet stores a raw pointer
	// into it), so build both here and let main own them.
	CoreDataSet core;
	double xor_in[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double xor_out[][1] = {{0}, {1}, {1}, {0}};
	for (int i = 0; i < 4; ++i) {
		std::vector<double> in(xor_in[i], xor_in[i] + 2);
		std::vector<double> out(xor_out[i], xor_out[i] + 1);
		core.addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet data;
	data.coreDataSet(core);

	Ensemble ensemble;
	for (uint i = 0; i < nMembers; ++i) {
		auto mlp = trainResidualXor(data, /*seed=*/100 + i, epochs);
		ensemble.addMlp(std::move(mlp)); // uniform 1/N weighting
	}

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "Residual XOR ensemble (" << nMembers << " members, "
	          << "arch 2-4-4-1, skip 0->1, " << epochs << " epochs each)\n\n";

	std::cout << "input        target   ";
	for (uint i = 0; i < nMembers; ++i) std::cout << "  m" << i << "    ";
	std::cout << "  ensemble  pred\n";
	std::cout << std::string(20 + nMembers * 8 + 18, '-') << "\n";

	int correct = 0;
	for (uint p = 0; p < data.size(); ++p) {
		Pattern& pat = data.pattern(p);
		std::vector<double>& in = pat.input();
		const double tgt = pat.output()[0];

		std::cout << "(" << in[0] << ", " << in[1] << ")    "
		          << static_cast<int>(tgt) << "      ";
		for (uint i = 0; i < ensemble.size(); ++i) {
			Mlp& m = const_cast<Mlp&>(ensemble.mlp(i));
			std::cout << " " << m.propagate(in)[0];
		}
		std::vector<double> y = ensemble.propagate(in);
		int pred = (y[0] >= 0.5) ? 1 : 0;
		std::cout << "   " << y[0] << "    " << pred << "\n";
		if (pred == static_cast<int>(tgt)) ++correct;
	}

	std::cout << "\nEnsemble accuracy: " << correct << "/" << data.size() << "\n";
	return correct == static_cast<int>(data.size()) ? 0 : 1;
}
