#include "Roc.hh"
#include "Evaluator.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <ostream>
#include <iterator>
#include <random>

using namespace EvalTools;
using std::cerr;
using std::copy;
using std::cout;
using std::endl;
using std::ostream;
using std::ostream_iterator;
using std::pair;
using std::setprecision;
using std::vector;

Roc::Roc() : theRoc(), theAuc(0), theEval(std::make_unique<Evaluator>()) {}

Roc::Roc(const Roc& roc)
    : theRoc(roc.theRoc), theAuc(roc.theAuc), theEval(std::make_unique<Evaluator>(*roc.theEval)) {}

Roc::~Roc() = default;

Roc& Roc::operator=(const Roc& roc) {
	if (this != &roc) {
		theRoc = roc.theRoc;
		theAuc = roc.theAuc;
		theEval = std::make_unique<Evaluator>(*roc.theEval);
	}
	return *this;
}

double Roc::calcAucWmw(vector<double>& out, vector<uint>& dout) {
	if (out.size() != dout.size()) {
		cerr << "Error: output and target vectors must have the same size" << endl;
		abort();
	}
	vector<double> posOut(0);
	vector<double> negOut(0);
	for (uint i = 0; i < out.size(); ++i)
		if (dout[i] > 0)
			posOut.push_back(out[i]);
		else
			negOut.push_back(out[i]);
	uint m = posOut.size();
	uint n = negOut.size();
	double r = 0;
	for (uint i = 0; i < m; ++i)
		for (uint j = 0; j < n; ++j)
			if (posOut[i] > negOut[j]) r += 1.0;
	// else if(posOut[i] == negOut[i])
	//	r += 0.5; //Try to compensate for similar outputs.

	//	cout<<"The n is: "<<n<<"\nThe m is: "<<m<<"\nThe rank is: "<<r<<endl;
	return theAuc = r / (m * n);
}

double Roc::calcAucWmwFast(vector<double>& out, vector<uint>& dout) {
	if (out.size() != dout.size()) {
		cerr << "Error: output and target vectors must have the same size" << endl;
		abort();
	}
	uint m = 0;
	uint n = 0;
	vector<pair<double, uint>> rank(0);
	for (uint i = 0; i < out.size(); ++i) {
		if (dout[i] > 0)
			m++;
		else
			n++;
		rank.push_back(pair<double, uint>(out[i], dout[i]));
	}
	sort(rank.begin(), rank.end());

	uint r = 0;
	for (uint i = 0; i < rank.size(); ++i)
		if (rank[i].second > 0) r += i;

	//	cout<<"The n is: "<<n<<"\nThe m is: "<<m<<"\nThe rank is: "<<r<<endl;
	return theAuc = (r - m * (m - 1.0) * 0.5) / (double)(m * n);
}

double Roc::calcAucTrapezoidal(vector<double>& out, vector<uint>& dout) {
	double area = 0;
	calcRoc(out, dout);
	vector<pair<double, double>>::iterator it;
	for (it = theRoc.begin() + 1; it != theRoc.end(); ++it) {
		double x1 = (it - 1)->first;
		double y1 = (it - 1)->second;
		double x2 = it->first;
		double y2 = it->second;
		area += (x2 - x1) * 0.5 * (y1 + y2);
	}
	return theAuc = area;
}

double Roc::aucWmwFastSample(const vector<double>& out, const vector<uint>& dout,
                            const vector<uint>& idx) {
	uint m = 0;
	uint n = 0;
	vector<pair<double, uint>> rank;
	rank.reserve(idx.size());
	for (uint k = 0; k < idx.size(); ++k) {
		const uint i = idx[k];
		if (dout[i] > 0)
			m++;
		else
			n++;
		rank.push_back(pair<double, uint>(out[i], dout[i]));
	}
	if (m == 0 || n == 0) return std::nan(""); // degenerate resample
	sort(rank.begin(), rank.end());
	double r = 0;
	for (uint i = 0; i < rank.size(); ++i)
		if (rank[i].second > 0) r += i;
	return (r - m * (m - 1.0) * 0.5) / (double)(m * n);
}

Roc::AucCI Roc::aucBootstrapCI(vector<double>& out, vector<uint>& dout, uint nBoot, double alpha,
                               std::uint64_t seed) {
	if (out.size() != dout.size()) {
		cerr << "Error: output and target vectors must have the same size" << endl;
		abort();
	}
	const uint N = out.size();

	AucCI res;
	res.alpha = alpha;
	res.auc = calcAucWmwFast(out, dout); // full-sample point estimate

	std::mt19937_64 gen(seed);
	std::uniform_int_distribution<uint> pick(0, N - 1);

	vector<double> boot;
	boot.reserve(nBoot);
	vector<uint> idx(N);
	uint nBelow = 0; // resamples with AUC <= 0.5, for the one-sided p-value
	for (uint b = 0; b < nBoot; ++b) {
		for (uint i = 0; i < N; ++i)
			idx[i] = pick(gen);
		const double a = aucWmwFastSample(out, dout, idx);
		if (std::isnan(a)) continue;
		boot.push_back(a);
		if (a <= 0.5) ++nBelow;
	}

	res.nBoot = boot.size();
	if (boot.empty()) { // pathological: every resample dropped a class
		res.lower = res.upper = res.auc;
		res.pValue = 1.0;
		return res;
	}
	sort(boot.begin(), boot.end());
	auto pct = [&boot](double q) {
		const double pos = q * (boot.size() - 1);
		const uint lo = (uint)std::floor(pos);
		const uint hi = (uint)std::ceil(pos);
		const double frac = pos - lo;
		return boot[lo] * (1.0 - frac) + boot[hi] * frac;
	};
	res.lower = pct(alpha / 2.0);
	res.upper = pct(1.0 - alpha / 2.0);
	// +1 smoothing so the p-value is never exactly 0
	res.pValue = (nBelow + 1.0) / (res.nBoot + 1.0);
	return res;
}

void Roc::calcRoc(vector<double>& out, vector<uint>& dout) {
	theRoc = vector<pair<double, double>>(0);
	pair<double, double> tmp;
	for (uint i = 0; i < out.size(); ++i) {
		theEval->cut(out[i]);
		theEval->evaluate(out, dout);
		tmp.first = theEval->fpf();
		tmp.second = theEval->tpf();
		theRoc.push_back(tmp);
	}
	sort(theRoc.begin(), theRoc.end());
}

void Roc::print(ostream& os) {
	if (!os) {
		cerr << "Output stream error.\n";
		return;
	}
	os << "#Spec\tSens" << endl;
	vector<pair<double, double>>::iterator it;
	for (it = theRoc.begin(); it != theRoc.end(); ++it)
		os << setprecision(6) << it->first << "\t" << it->second << endl;
}

// PRIVATE---------------------------------------------------------------------//

template <class T> void Roc::printVector(vector<T>& vec) {
	typename vector<T>::iterator it;
	copy(vec.begin(), vec.end(), ostream_iterator<T>(cout, " "));
	for (it = vec.begin(); it != vec.end(); ++it)
		cout << *it << " ";
	cout << endl;
}
