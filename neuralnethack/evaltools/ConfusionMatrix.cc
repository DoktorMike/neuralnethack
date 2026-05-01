#include "ConfusionMatrix.hh"

#include <cassert>
#include <iomanip>

using namespace EvalTools;
using namespace NeuralNetHack;
using namespace DataTools;
using namespace MultiLayerPerceptron;
using std::vector;

ConfusionMatrix::ConfusionMatrix(uint nClasses)
    : n(nClasses), m(nClasses, vector<uint>(nClasses, 0)) {
	assert(nClasses >= 2);
}

void ConfusionMatrix::add(uint actual, uint predicted) {
	assert(actual < n && predicted < n);
	++m[actual][predicted];
}

void ConfusionMatrix::reset() {
	for (auto& row : m)
		std::fill(row.begin(), row.end(), 0u);
}

uint ConfusionMatrix::total() const {
	uint t = 0;
	for (const auto& row : m)
		for (uint v : row)
			t += v;
	return t;
}

uint ConfusionMatrix::correct() const {
	uint c = 0;
	for (uint i = 0; i < n; ++i)
		c += m[i][i];
	return c;
}

uint ConfusionMatrix::actualTotal(uint cls) const {
	assert(cls < n);
	uint s = 0;
	for (uint v : m[cls])
		s += v;
	return s;
}

uint ConfusionMatrix::predictedTotal(uint cls) const {
	assert(cls < n);
	uint s = 0;
	for (uint i = 0; i < n; ++i)
		s += m[i][cls];
	return s;
}

uint ConfusionMatrix::tp() const {
	assert(n == 2);
	return m[1][1];
}
uint ConfusionMatrix::fp() const {
	assert(n == 2);
	return m[0][1];
}
uint ConfusionMatrix::fn() const {
	assert(n == 2);
	return m[1][0];
}
uint ConfusionMatrix::tn() const {
	assert(n == 2);
	return m[0][0];
}

void ConfusionMatrix::print(std::ostream& os) const {
	os << "Confusion matrix (" << n << " classes, rows=actual, cols=predicted):\n";
	os << std::setw(8) << "";
	for (uint j = 0; j < n; ++j)
		os << std::setw(8) << j;
	os << "\n";
	for (uint i = 0; i < n; ++i) {
		os << std::setw(8) << i;
		for (uint j = 0; j < n; ++j)
			os << std::setw(8) << m[i][j];
		os << "\n";
	}
}

ConfusionMatrix ConfusionMatrix::fromBinary(const vector<double>& output,
                                            const vector<uint>& target, double cut) {
	assert(output.size() == target.size());
	ConfusionMatrix cm(2);
	for (size_t i = 0; i < output.size(); ++i) {
		uint pred = (output[i] >= cut) ? 1u : 0u;
		cm.add(target[i], pred);
	}
	return cm;
}

static uint argmax(const vector<double>& v) {
	uint best = 0;
	for (uint i = 1; i < v.size(); ++i)
		if (v[i] > v[best]) best = i;
	return best;
}

ConfusionMatrix ConfusionMatrix::fromMulticlass(const vector<vector<double>>& output,
                                                const vector<vector<double>>& target) {
	assert(output.size() == target.size());
	assert(!output.empty());
	uint nc = output.front().size();
	ConfusionMatrix cm(nc);
	for (size_t i = 0; i < output.size(); ++i) {
		assert(output[i].size() == nc && target[i].size() == nc);
		cm.add(argmax(target[i]), argmax(output[i]));
	}
	return cm;
}

ConfusionMatrix ConfusionMatrix::fromEnsemble(Ensemble& committee, DataSet& data, double cut) {
	assert(data.size() > 0);
	// Probe output dimension.
	vector<double> first = committee.propagate(data.pattern(0).input());
	if (first.size() == 1) {
		ConfusionMatrix cm(2);
		uint actual0 = (data.pattern(0).output().front() >= 0.5) ? 1u : 0u;
		cm.add(actual0, (first[0] >= cut) ? 1u : 0u);
		for (uint i = 1; i < data.size(); ++i) {
			Pattern& pat = data.pattern(i);
			vector<double> y = committee.propagate(pat.input());
			uint actual = (pat.output().front() >= 0.5) ? 1u : 0u;
			cm.add(actual, (y[0] >= cut) ? 1u : 0u);
		}
		return cm;
	}
	ConfusionMatrix cm(static_cast<uint>(first.size()));
	cm.add(argmax(data.pattern(0).output()), argmax(first));
	for (uint i = 1; i < data.size(); ++i) {
		Pattern& pat = data.pattern(i);
		vector<double> y = committee.propagate(pat.input());
		cm.add(argmax(pat.output()), argmax(y));
	}
	return cm;
}
