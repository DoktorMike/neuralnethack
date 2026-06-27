#include "Error.hh"
#include "../datatools/Pattern.hh"

#include <vector>
#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

// PUBLIC

Error::~Error() {}

Mlp& Error::mlp() {
	return *theMlp;
}

void Error::mlp(Mlp& mlp) {
	// Rebinding to a borrowed Mlp; release any previously-owned one.
	theOwnedMlp.reset();
	theMlp = &mlp;
}

DataSet& Error::dset() {
	return *theDset;
}

void Error::dset(DataSet& dset) {
	theDset = &dset;
}

bool Error::weightElimOn() const {
	return theWeightElimOn;
}

void Error::weightElimOn(bool on) {
	theWeightElimOn = on;
}

double Error::weightElimAlpha() const {
	return theWeightElimAlpha;
}

void Error::weightElimAlpha(double alpha) {
	theWeightElimAlpha = alpha;
}

double Error::weightElimW0() const {
	return theWeightElimW0;
}

void Error::weightElimW0(double w0) {
	theWeightElimW0 = w0;
}

void Error::classWeights(const vector<double>& w) {
	theClassWeights = w;
}

const vector<double>& Error::classWeights() const {
	return theClassWeights;
}

// PROTECTED

Error::Error(Mlp& mlp, DataSet& dset)
    : theMlp(&mlp), theOwnedMlp(nullptr), theDset(&dset), theWeightElimOn(false),
      theWeightElimAlpha(0), theWeightElimW0(0) {}

Error::Error(unique_ptr<Mlp> mlp, DataSet& dset)
    : theMlp(mlp.get()), theOwnedMlp(std::move(mlp)), theDset(&dset), theWeightElimOn(false),
      theWeightElimAlpha(0), theWeightElimW0(0) {}

double Error::weightElimGrad(double wi) const {
	double alpha = theWeightElimAlpha;
	double w0 = theWeightElimW0;
	return alpha * ((2 * wi * pow(w0, 2)) / pow(pow(w0, 2) + pow(wi, 2), 2));
}

void Error::weightElimGrad(vector<double>& gradients, const vector<double>& weights, uint offset,
                           uint length) const {
	for (uint i = offset; i < offset + length; ++i)
		gradients[i] += weightElimGrad(weights[i]);
}

void Error::weightElimGradLayer(vector<double>& gradients, const vector<double>& weights,
                                uint ncurr, uint nprev) const {
	uint offset = 0;
	for (uint i = 0; i < ncurr; ++i) {
		weightElimGrad(gradients, weights, offset, nprev);
		offset += nprev + 1;
	}
}

void Error::weightElimGradMlp(vector<double>& gradients, const vector<double>& weights,
                              const vector<uint>& arch) const {
	uint offset = 0;
	for (uint i = 1; i < arch.size(); ++i) {
		for (uint j = 0; j < arch[i]; ++j) {
			weightElimGrad(gradients, weights, offset, arch[i - 1]);
			offset += arch[i - 1] + 1; // avoid the bias
		}
	}
}

void Error::weightElimGrad() {
	assert(theMlp != 0);

	if (weightElimOn() == true) {
		for (uint i = 0; i < theMlp->nLayers(); ++i) {
			Layer& l = theMlp->layer(i);
			weightElimGradLayer(l.gradients(), l.weights(), l.nNeurons(), l.nPrevious());
		}
	}
}

double Error::weightElim() const {
	vector<double> weights = theMlp->weights();
	double we = 0;
	for (vector<double>::iterator itw = weights.begin(); itw != weights.end(); ++itw) {
		double wisqr = pow(*itw, 2);
		double w0sqr = pow(weightElimW0(), 2);
		we += wisqr / (w0sqr + wisqr);
	}
	return we;
}

void Error::packBatch(DataSet& dset) const {
	const uint B = dset.size();
	const uint n_in = dset.nInput();
	const uint n_out = dset.nOutput();

	theInputMatrix.resize(B * n_in);
	theTargetMatrix.resize(B * n_out);

	for (uint b = 0; b < B; ++b) {
		Pattern& p = dset.pattern(b);
		const vector<double>& inp = p.input();
		copy(inp.begin(), inp.end(), theInputMatrix.data() + b * n_in);
		const vector<double>& out = p.output();
		copy(out.begin(), out.end(), theTargetMatrix.data() + b * n_out);
	}
}

void Error::patternWeights(uint bs, uint nOut, vector<double>& pw, double& denom) const {
	if (theClassWeights.empty()) {
		pw.clear();
		denom = (double)bs;
		return;
	}
	const uint nClasses = (nOut == 1) ? 2 : nOut;
	assert(theClassWeights.size() == nClasses);
	pw.resize(bs);
	denom = 0;
	const double* t = theTargetMatrix.data();
	for (uint b = 0; b < bs; ++b) {
		uint cls;
		if (nOut == 1) {
			cls = (t[b] != 0.0) ? 1u : 0u;
		} else {
			cls = 0;
			double best = t[b * nOut];
			for (uint j = 1; j < nOut; ++j)
				if (t[b * nOut + j] > best) {
					best = t[b * nOut + j];
					cls = j;
				}
		}
		pw[b] = theClassWeights[cls];
		denom += pw[b];
	}
	if (denom <= 0.0) denom = (double)bs; // guard degenerate weights
}

// PRIVATE--------------------------------------------------------------------//

Error::Error(const Error& err) {
	*this = err;
}

Error& Error::operator=(const Error& err) {
	if (this != &err) {
		theMlp = err.theMlp;
		theDset = err.theDset;
		theClassWeights = err.theClassWeights;
	}
	return *this;
}
