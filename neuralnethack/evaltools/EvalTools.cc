#include "EvalTools.hh"
#include "Roc.hh"
#include "Gof.hh"

#include <cassert>
#include <cmath>

using namespace EvalTools;
using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

double ErrorMeasures::crossEntropy(Ensemble& committee, DataSet& data) {
	double err = 0;
	uint bs = data.size();

	for (uint i = 0; i < bs; ++i) {
		Pattern& p = data.pattern(i);
		vector<double> output = committee.propagate(p.input());
		vector<double>& target = p.output();
		err += crossEntropy(output, target);
	}
	return -err / (double)bs;
}

double ErrorMeasures::crossEntropy(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());

	double power = -20;
	double tiny = exp(power);

	// return target[0] * log(output[0]) + (1.0 - target[0]) * log(1.0 - output[0]);
	double e = 0;
	if (target.size() == 1) {
		if (target[0] == 0.0)
			e = (1.0 - output[0] > tiny) ? log(1.0 - output[0]) : power;
		else if (target[0] == 1.0)
			e = (output[0] > tiny) ? log(output[0]) : power;
		else
			cerr << "Target is neither 1.0 nor 0.0 in the single class case!" << endl;
	} else {
		for (uint i = 0; i < target.size(); ++i)
			if (target[i] == 0.0)
				e += 0;
			else if (target[i] == 1.0)
				e += (output[i] > tiny) ? log(output[i]) : power;
	}
	return e;
}

double ErrorMeasures::summedSquare(Ensemble& committee, DataSet& data) {
	double err = 0;
	uint bs = data.size();

	for (uint i = 0; i < bs; ++i) {
		Pattern& p = data.pattern(i);
		vector<double> output = committee.propagate(p.input());
		vector<double>& target = p.output();
		err += summedSquare(output, target);
	}
	return 0.5 * err / (double)bs;
}

double ErrorMeasures::summedSquare(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	double e = 0;
	for (uint i = 0; i < target.size(); ++i)
		e += pow(target[i] - output[i], 2);
	return e;
}

double ErrorMeasures::auc(Ensemble& committee, DataSet& data) {
	using EvalTools::Roc;
	vector<double> output;
	vector<uint> target;
	buildOutputTargetVectors(committee, data, output, target);
	Roc roc;
	// cout<<roc.calcAucWmw(output, target)<<endl;
	// cout<<roc.calcAucWmwFast(output, target)<<endl;
	return roc.calcAucWmwFast(output, target);
	// return roc.calcAucTrapezoidal(output, target);
}

double ErrorMeasures::accuracy(Ensemble& committee, DataSet& data) {
	uint correct = 0;
	for (uint i = 0; i < data.size(); ++i) {
		Pattern& p = data.pattern(i);
		vector<double> out = committee.propagate(p.input());
		const vector<double>& tgt = p.output();
		if (out.size() == 1) {
			const uint pred = (out[0] >= 0.5) ? 1u : 0u;
			const uint t = (tgt.front() >= 0.5) ? 1u : 0u;
			if (pred == t) ++correct;
		} else {
			uint argP = 0, argT = 0;
			for (uint j = 1; j < out.size(); ++j) {
				if (out[j] > out[argP]) argP = j;
				if (tgt[j] > tgt[argT]) argT = j;
			}
			if (argP == argT) ++correct;
		}
	}
	return static_cast<double>(correct) / data.size();
}

double ErrorMeasures::gof(Ensemble& committee, DataSet& data) {
	using EvalTools::Gof;
	vector<double> output;
	vector<uint> target;
	buildOutputTargetVectors(committee, data, output, target);
	Gof gof(10);
	return gof.goodnessOfFit(output, target);
}

void ErrorMeasures::buildOutputTargetVectors(Ensemble& committee, DataSet& data,
                                             vector<double>& output, vector<uint>& target) {
	output.clear();
	target.clear();

	for (uint i = 0; i < data.size(); ++i) {
		Pattern& pat = data.pattern(i);
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
	}
}

void ErrorMeasures::buildOutputTargetVectors(Ensemble& committee, DataSet& data,
                                             vector<vector<double>>& output,
                                             vector<vector<double>>& target) {
	output.clear();
	target.clear();

	for (uint i = 0; i < data.size(); ++i) {
		Pattern& pat = data.pattern(i);
		output.push_back(committee.propagate(pat.input()));
		target.push_back(pat.output());
	}
}

void ErrorMeasures::buildFlatRegressionVectors(Ensemble& committee, DataSet& data,
                                               vector<double>& output, vector<double>& target) {
	output.clear();
	target.clear();
	for (uint i = 0; i < data.size(); ++i) {
		Pattern& pat = data.pattern(i);
		vector<double> y = committee.propagate(pat.input());
		const vector<double>& t = pat.output();
		output.insert(output.end(), y.begin(), y.end());
		target.insert(target.end(), t.begin(), t.end());
	}
}

// ---- Regression metrics --------------------------------------------------

double ErrorMeasures::mae(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	if (target.empty()) return std::nan("");
	double s = 0.0;
	for (size_t i = 0; i < target.size(); ++i)
		s += std::fabs(target[i] - output[i]);
	return s / static_cast<double>(target.size());
}

double ErrorMeasures::mae(Ensemble& committee, DataSet& data) {
	vector<double> y, t;
	buildFlatRegressionVectors(committee, data, y, t);
	return mae(y, t);
}

double ErrorMeasures::mape(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	const double eps = 1e-12;
	double s = 0.0;
	uint n = 0;
	for (size_t i = 0; i < target.size(); ++i) {
		double a = std::fabs(target[i]);
		if (a < eps) continue; // undefined at zero
		s += std::fabs(target[i] - output[i]) / a;
		++n;
	}
	if (n == 0) return std::nan("");
	return 100.0 * s / static_cast<double>(n);
}

double ErrorMeasures::mape(Ensemble& committee, DataSet& data) {
	vector<double> y, t;
	buildFlatRegressionVectors(committee, data, y, t);
	return mape(y, t);
}

double ErrorMeasures::smape(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	const double eps = 1e-12;
	double s = 0.0;
	uint n = 0;
	for (size_t i = 0; i < target.size(); ++i) {
		double denom = std::fabs(target[i]) + std::fabs(output[i]);
		if (denom < eps) continue; // both target and output are zero
		s += std::fabs(target[i] - output[i]) / denom;
		++n;
	}
	if (n == 0) return std::nan("");
	return 200.0 * s / static_cast<double>(n);
}

double ErrorMeasures::smape(Ensemble& committee, DataSet& data) {
	vector<double> y, t;
	buildFlatRegressionVectors(committee, data, y, t);
	return smape(y, t);
}

double ErrorMeasures::rmse(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	if (target.empty()) return std::nan("");
	double s = 0.0;
	for (size_t i = 0; i < target.size(); ++i) {
		double d = target[i] - output[i];
		s += d * d;
	}
	return std::sqrt(s / static_cast<double>(target.size()));
}

double ErrorMeasures::rmse(Ensemble& committee, DataSet& data) {
	vector<double> y, t;
	buildFlatRegressionVectors(committee, data, y, t);
	return rmse(y, t);
}

double ErrorMeasures::r2(const vector<double>& output, const vector<double>& target) {
	assert(output.size() == target.size());
	if (target.empty()) return std::nan("");
	double mean_t = 0.0;
	for (double v : target)
		mean_t += v;
	mean_t /= static_cast<double>(target.size());

	double ss_res = 0.0, ss_tot = 0.0;
	for (size_t i = 0; i < target.size(); ++i) {
		double r = target[i] - output[i];
		double d = target[i] - mean_t;
		ss_res += r * r;
		ss_tot += d * d;
	}
	if (ss_tot == 0.0) return std::nan(""); // constant target
	return 1.0 - ss_res / ss_tot;
}

double ErrorMeasures::r2(Ensemble& committee, DataSet& data) {
	vector<double> y, t;
	buildFlatRegressionVectors(committee, data, y, t);
	return r2(y, t);
}

// ---- Confusion-matrix derived metrics -----------------------------------

double ErrorMeasures::accuracy(const ConfusionMatrix& cm) {
	uint total = cm.total();
	if (total == 0) return std::nan("");
	return static_cast<double>(cm.correct()) / static_cast<double>(total);
}

double ErrorMeasures::precision(const ConfusionMatrix& cm, uint cls) {
	uint pred = cm.predictedTotal(cls);
	if (pred == 0) return std::nan("");
	return static_cast<double>(cm.count(cls, cls)) / static_cast<double>(pred);
}

double ErrorMeasures::recall(const ConfusionMatrix& cm, uint cls) {
	uint actual = cm.actualTotal(cls);
	if (actual == 0) return std::nan("");
	return static_cast<double>(cm.count(cls, cls)) / static_cast<double>(actual);
}

double ErrorMeasures::f1(const ConfusionMatrix& cm, uint cls) {
	double p = precision(cm, cls);
	double r = recall(cm, cls);
	if (std::isnan(p) || std::isnan(r)) return std::nan("");
	if (p + r == 0.0) return std::nan("");
	return 2.0 * p * r / (p + r);
}

double ErrorMeasures::mcc(const ConfusionMatrix& cm) {
	if (cm.nClasses() != 2) return std::nan("");
	double tp = cm.tp(), tn = cm.tn(), fp = cm.fp(), fn = cm.fn();
	double num = tp * tn - fp * fn;
	double den = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
	if (den == 0.0) return std::nan("");
	return num / den;
}

double ErrorMeasures::balancedAccuracy(const ConfusionMatrix& cm) {
	double s = 0.0;
	uint counted = 0;
	for (uint c = 0; c < cm.nClasses(); ++c) {
		double r = recall(cm, c);
		if (std::isnan(r)) continue;
		s += r;
		++counted;
	}
	if (counted == 0) return std::nan("");
	return s / static_cast<double>(counted);
}

static double macroAvg(const ConfusionMatrix& cm, double (*metric)(const ConfusionMatrix&, uint)) {
	double s = 0.0;
	uint counted = 0;
	for (uint c = 0; c < cm.nClasses(); ++c) {
		double v = metric(cm, c);
		if (std::isnan(v)) continue;
		s += v;
		++counted;
	}
	if (counted == 0) return std::nan("");
	return s / static_cast<double>(counted);
}

double ErrorMeasures::macroPrecision(const ConfusionMatrix& cm) {
	return macroAvg(cm, precision);
}
double ErrorMeasures::macroRecall(const ConfusionMatrix& cm) {
	return macroAvg(cm, recall);
}
double ErrorMeasures::macroF1(const ConfusionMatrix& cm) {
	return macroAvg(cm, f1);
}
