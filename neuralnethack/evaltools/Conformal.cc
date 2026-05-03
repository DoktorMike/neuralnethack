#include "Conformal.hh"

#include "datatools/Pattern.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace EvalTools {

namespace {

// Finite-sample-corrected (1-alpha) quantile.
// k = ceil((n+1) * level), 1-indexed. If k > n, return +inf
// (calibration set too small for the requested level → universal set).
// Mutates `s` (uses nth_element).
double finiteQuantile(std::vector<double>& s, double level) {
	const std::size_t n = s.size();
	if (n == 0) return std::numeric_limits<double>::infinity();
	const double k_real = std::ceil((static_cast<double>(n) + 1.0) * level);
	if (k_real > static_cast<double>(n)) return std::numeric_limits<double>::infinity();
	const std::size_t k = (k_real < 1.0) ? 1 : static_cast<std::size_t>(k_real);
	std::nth_element(s.begin(), s.begin() + (k - 1), s.end());
	return s[k - 1];
}

// Convert ensemble output + target to (K-vector probs, true class index).
// Single-output → binary [1-y, y] with target rounded.
// Multi-output  → output as probs, argmax of one-hot target.
void toProbsAndCls(const std::vector<double>& out, const std::vector<double>& tgt,
                   std::vector<double>& probs, uint& cls) {
	if (out.size() == 1) {
		const double y = out[0];
		probs = {1.0 - y, y};
		cls = (tgt[0] >= 0.5) ? 1u : 0u;
	} else {
		probs = out;
		uint a = 0;
		for (uint i = 1; i < tgt.size(); ++i)
			if (tgt[i] > tgt[a]) a = i;
		cls = a;
	}
}

std::vector<double> probsFromOutput(const std::vector<double>& out) {
	if (out.size() == 1) return {1.0 - out[0], out[0]};
	return out;
}

} // namespace

Conformal::Conformal(Mode mode, double alpha) : theMode(mode), theAlpha(alpha) {
	if (!(alpha > 0.0 && alpha < 1.0))
		throw std::invalid_argument("Conformal: alpha must be in (0,1)");
}

void Conformal::calibrate(NeuralNetHack::Ensemble& e, DataTools::DataSet& cal) {
	const uint n = cal.size();
	if (n == 0) throw std::runtime_error("Conformal::calibrate: empty calibration set");

	const auto& y0 = e.propagate(cal.pattern(0).input());
	theNOutput = static_cast<uint>(y0.size());

	if (theMode == Mode::Regression) {
		std::vector<std::vector<double>> scores(theNOutput);
		for (auto& s : scores)
			s.reserve(n);
		for (uint i = 0; i < n; ++i) {
			auto& pat = cal.pattern(i);
			const auto& y = e.propagate(pat.input());
			const auto& t = pat.output();
			for (uint d = 0; d < theNOutput; ++d)
				scores[d].push_back(std::fabs(t[d] - y[d]));
		}
		theQ.assign(theNOutput, 0.0);
		for (uint d = 0; d < theNOutput; ++d)
			theQ[d] = finiteQuantile(scores[d], 1.0 - theAlpha);
	} else {
		std::vector<double> scores;
		scores.reserve(n);
		for (uint i = 0; i < n; ++i) {
			auto& pat = cal.pattern(i);
			const auto& out = e.propagate(pat.input());
			const auto& t = pat.output();
			std::vector<double> probs;
			uint cls;
			toProbsAndCls(out, t, probs, cls);
			scores.push_back(1.0 - probs[cls]);
		}
		theQ = {finiteQuantile(scores, 1.0 - theAlpha)};
	}
	theCalibrated = true;
}

std::vector<Conformal::Interval> Conformal::interval(NeuralNetHack::Ensemble& e,
                                                     const std::vector<double>& x) const {
	if (!theCalibrated) throw std::runtime_error("Conformal::interval: not calibrated");
	if (theMode != Mode::Regression)
		throw std::runtime_error("Conformal::interval: not regression mode");
	const auto& y = e.propagate(x);
	std::vector<Interval> out(theNOutput);
	for (uint d = 0; d < theNOutput; ++d) {
		out[d].lo = y[d] - theQ[d];
		out[d].hi = y[d] + theQ[d];
	}
	return out;
}

std::vector<uint> Conformal::set(NeuralNetHack::Ensemble& e, const std::vector<double>& x) const {
	if (!theCalibrated) throw std::runtime_error("Conformal::set: not calibrated");
	if (theMode != Mode::Classification)
		throw std::runtime_error("Conformal::set: not classification mode");
	const auto probs = probsFromOutput(e.propagate(x));
	const double thresh = 1.0 - theQ[0];
	std::vector<uint> idx;
	for (uint k = 0; k < probs.size(); ++k)
		if (probs[k] >= thresh) idx.push_back(k);
	return idx;
}

double Conformal::coverage(NeuralNetHack::Ensemble& e, DataTools::DataSet& tst) const {
	if (!theCalibrated) throw std::runtime_error("Conformal::coverage: not calibrated");
	const uint n = tst.size();
	if (n == 0) return std::numeric_limits<double>::quiet_NaN();

	if (theMode == Mode::Regression) {
		std::size_t inCount = 0, total = 0;
		for (uint i = 0; i < n; ++i) {
			auto& pat = tst.pattern(i);
			const auto& y = e.propagate(pat.input());
			const auto& t = pat.output();
			for (uint d = 0; d < theNOutput; ++d) {
				if (std::fabs(t[d] - y[d]) <= theQ[d]) ++inCount;
				++total;
			}
		}
		return total ? static_cast<double>(inCount) / static_cast<double>(total)
		             : std::numeric_limits<double>::quiet_NaN();
	}

	const double thresh = 1.0 - theQ[0];
	std::size_t covered = 0;
	for (uint i = 0; i < n; ++i) {
		auto& pat = tst.pattern(i);
		const auto& out = e.propagate(pat.input());
		const auto& t = pat.output();
		std::vector<double> probs;
		uint cls;
		toProbsAndCls(out, t, probs, cls);
		if (probs[cls] >= thresh) ++covered;
	}
	return static_cast<double>(covered) / static_cast<double>(n);
}

} // namespace EvalTools
