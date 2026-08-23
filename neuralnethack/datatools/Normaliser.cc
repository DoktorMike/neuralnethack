#include "Normaliser.hh"

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <algorithm>

using namespace DataTools;
using namespace std;

Normaliser::Normaliser() : theStdDev(0), theMean(0), theSkip(0) {}

Normaliser::Normaliser(std::vector<double>& stds, std::vector<double>& means,
                       std::vector<bool>& skips)
    : theStdDev(stds), theMean(means), theSkip(skips) {}

Normaliser::Normaliser(const Normaliser& n) {
	*this = n;
}

Normaliser::~Normaliser() {}

Normaliser& Normaliser::operator=(const Normaliser& n) {
	if (this != &n) {
		theStdDev = n.theStdDev;
		theMean = n.theMean;
		theSkip = n.theSkip;
	}
	return *this;
}

DataSet& Normaliser::normalise(DataSet& d) {
	if (d.nInput() + d.nOutput() != theMean.size()) {
		cerr << "Error: mean vector length and input data length differ!" << endl;
		abort();
	}
	for (uint i = 0; i < d.size(); ++i)
		normalise(d.pattern(i));
	if (theSkip.size() > 0) transformBinaryCoding(d);
	return d;
}

DataSet& Normaliser::calcAndNormalise(DataSet& d, bool doSkip) {
	theMean = vector<double>(d.nInput() + d.nOutput(), 0);
	theStdDev = vector<double>(d.nInput() + d.nOutput(), 0);
	calcMean(d);
	calcStdDev(d);

	if (doSkip) {
		theSkip = vector<bool>(d.nInput() + d.nOutput(), true);
		findSkip(d);
		transformBinaryCoding(d);
		for (uint i = 0; i < theSkip.size(); ++i)
			if (theSkip[i]) {
				theMean[i] = 0;
				theStdDev[i] = 1;
			}
	}

	for (uint i = 0; i < d.size(); ++i)
		normalise(d.pattern(i));
	return d;
}

DataSet& Normaliser::calcAndNormaliseMaxAbs(DataSet& d, const vector<uint>& colGroup) {
	const uint n = d.nInput() + d.nOutput();
	assert(colGroup.empty() || colGroup.size() == d.nInput());
	theMean.assign(n, 0.0);
	theStdDev.assign(n, 0.0);
	theSkip.clear();

	const uint nIn = d.nInput();

	// INPUTS: per-column max|x| — the max-abs rationale (zero
	// preservation, Hill domain) is input-side.
	// OUTPUTS: divided by their train standard deviation, no centering.
	// Training is only stable when the target sd is O(1): measured on the
	// mmm dataset, sd ~1.6 gives holdout R^2 0.90, sd 0.1-0.2 collapses
	// to ~0.3 (Adam's absolute step noise swamps the signal), sd ~17
	// diverges to NaN, sd ~1600 never gets off the ground. Dividing by
	// sd makes the mode robust to targets on any scale while keeping
	// zero meaningful (no shift).
	for (uint i = 0; i < d.size(); ++i) {
		Pattern& p = d.pattern(i);
		for (uint j = 0; j < p.nInput(); ++j)
			theStdDev[j] = max(theStdDev[j], fabs(p.input()[j]));
	}
	{
		const uint nOut = d.nOutput();
		vector<double> mean(nOut, 0.0), var(nOut, 0.0);
		for (uint i = 0; i < d.size(); ++i)
			for (uint j = 0; j < nOut; ++j)
				mean[j] += d.pattern(i).output()[j];
		for (auto& m : mean)
			m /= d.size();
		for (uint i = 0; i < d.size(); ++i)
			for (uint j = 0; j < nOut; ++j) {
				const double dv = d.pattern(i).output()[j] - mean[j];
				var[j] += dv * dv;
			}
		for (uint j = 0; j < nOut; ++j)
			theStdDev[nIn + j] = sqrt(var[j] / d.size());
	}

	// Share one scale per group: the max over the group's columns
	if (!colGroup.empty()) {
		vector<double> groupMax;
		for (uint j = 0; j < nIn; ++j) {
			if (colGroup[j] >= groupMax.size()) groupMax.resize(colGroup[j] + 1, 0.0);
			groupMax[colGroup[j]] = max(groupMax[colGroup[j]], theStdDev[j]);
		}
		for (uint j = 0; j < nIn; ++j)
			theStdDev[j] = groupMax[colGroup[j]];
	}

	// All-zero columns/groups keep scale 1
	for (auto& s : theStdDev)
		if (s <= 0.0) s = 1.0;

	for (uint i = 0; i < d.size(); ++i)
		normalise(d.pattern(i));
	return d;
}

/** Subtracts the mean from each variable and then divides it with the corresponding standard
 * deviation.
 */
struct SubtractAndDivide {
	vector<double>::const_iterator itm;
	vector<double>::const_iterator its;
	SubtractAndDivide(vector<double>::const_iterator m, vector<double>::const_iterator s)
	    : itm(m), its(s) {}
	void operator()(double& x) {
		double diff = abs(x - *itm);
		if (diff > 1e-15)
			x = (x - *itm) / *its;
		else
			x = 0;
		++itm;
		++its;
	}
};

/** Multiplies each variable with its standard deviation and then adds its
 * mean.
 */
struct MultiplyAndAdd {
	vector<double>::const_iterator itm;
	vector<double>::const_iterator its;
	MultiplyAndAdd(vector<double>::const_iterator m, vector<double>::const_iterator s)
	    : itm(m), its(s) {}
	void operator()(double& x) {
		x = x * *its++ + *itm++;
		if (fabs(x) < 1e-15) x = 0;
	}
};

Pattern& Normaliser::normalise(Pattern& p) {
	for_each(p.input().begin(), p.input().end(),
	         SubtractAndDivide(theMean.begin(), theStdDev.begin()));
	for_each(p.output().begin(), p.output().end(),
	         SubtractAndDivide(theMean.begin() + p.nInput(), theStdDev.begin() + p.nInput()));
	return p;
}

vector<double>& Normaliser::normaliseInput(vector<double>& i) {
	for_each(i.begin(), i.end(), SubtractAndDivide(theMean.begin(), theStdDev.begin()));
	transformBinaryCoding(i);
	return i;
}

DataSet& Normaliser::unnormalise(DataSet& d) {
	assert(theMean.size() == theStdDev.size());
	assert(theMean.size() == (d.nInput() + d.nOutput()));

	for (uint i = 0; i < d.size(); ++i)
		unnormalise(d.pattern(i));
	return d;
}

Pattern& Normaliser::unnormalise(Pattern& p) {
	for_each(p.input().begin(), p.input().end(),
	         MultiplyAndAdd(theMean.begin(), theStdDev.begin()));
	for_each(p.output().begin(), p.output().end(),
	         MultiplyAndAdd(theMean.begin() + p.nInput(), theStdDev.begin() + p.nInput()));
	return p;
}

vector<double>& Normaliser::stdDev() {
	return theStdDev;
}

void Normaliser::stdDev(vector<double>& s) {
	theStdDev = s;
}

vector<double>& Normaliser::mean() {
	return theMean;
}

void Normaliser::mean(vector<double>& m) {
	theMean = m;
}

vector<bool>& Normaliser::skip() {
	return theSkip;
}

void Normaliser::skip(vector<bool>& m) {
	theSkip = m;
}

// PRIVATE---------------------------------------------------------------------//

void Normaliser::calcMean(DataSet& d) {
	for (uint i = 0; i < d.size(); ++i) {
		Pattern& p = d.pattern(i);
		vector<double>& in = p.input();
		vector<double>& out = p.output();
		uint k = 0;
		for (uint j = 0; j < in.size(); ++j, ++k)
			theMean[k] += in[j];
		for (uint j = 0; j < out.size(); ++j, ++k)
			theMean[k] += out[j];
	}
	double n = d.size();
	for (uint i = 0; i < theMean.size(); ++i)
		theMean[i] = theMean[i] / n;
}

void Normaliser::calcStdDev(DataSet& d) {
	for (uint i = 0; i < d.size(); ++i) {
		Pattern& p = d.pattern(i);
		vector<double>& in = p.input();
		vector<double>& out = p.output();
		uint k = 0;
		for (uint j = 0; j < in.size(); ++j, ++k)
			theStdDev[k] += pow(in[j] - theMean[k], 2);
		for (uint j = 0; j < out.size(); ++j, ++k)
			theStdDev[k] += pow(out[j] - theMean[k], 2);
	}
	double n = d.size();
	for (uint i = 0; i < theStdDev.size(); ++i)
		theStdDev[i] = sqrt(theStdDev[i] / n);
}

void Normaliser::findSkip(DataSet& d) {
	for (uint i = 0; i < d.size(); ++i) {
		Pattern& p = d.pattern(i);
		vector<double>& in = p.input();
		vector<double>& out = p.output();
		uint k = 0;
		for (uint j = 0; j < in.size(); ++j, ++k)
			theSkip[k] = (theSkip[k] && skipBin(in[j])) || (theSkip[k] && skipSig(in[j]));
		for (uint j = 0; j < out.size(); ++j, ++k)
			theSkip[k] = (theSkip[k] && skipBin(out[j])) || (theSkip[k] && skipSig(out[j]));
	}
}

bool Normaliser::skipBin(double val) const {
	double e = 1e-15;
	return ((fabs(val - 1) <= e) || (fabs(val) <= e));
}

bool Normaliser::skipSig(double val) const {
	double e = 1e-15;
	return ((fabs(val - 1) <= e) || (fabs(val + 1) <= e));
}

void Normaliser::transformBinaryCoding(DataSet& data) {
	for (uint i = 0; i < data.size(); ++i) {
		vector<double>& in = data.pattern(i).input();
		uint k = 0;
		for (uint j = 0; j < in.size(); ++j, ++k)
			if (theSkip[k] == true && in[j] == 0) in[j] = -1;
	}
}

void Normaliser::transformBinaryCoding(vector<double>& input) {
	for (uint j = 0; j < input.size(); ++j)
		if (theSkip[j] == true && input[j] == 0) input[j] = -1;
}
