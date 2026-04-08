#include "Adam.hh"
#include "Error.hh"

#include <ostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

Adam::Adam(Mlp& mlp, DataSet& data, Error& error, double te, uint bs, double lr, double beta1,
           double beta2, double eps, double weightDecay)
    : Trainer(mlp, data, error, te, bs), theLearningRate(lr), theBeta1(beta1), theBeta2(beta2),
      theEpsilon(eps), theWeightDecay(weightDecay), theTimestep(0) {}

Adam::~Adam() {}

double Adam::learningRate() const {
	return theLearningRate;
}
void Adam::learningRate(double lr) {
	theLearningRate = lr;
}

void Adam::train(ostream& os) {
	if (theBatchSize > theData->size()) {
		cerr << "Warning: Batch size larger than DataSet, reseting to DataSet size." << endl;
		theBatchSize = theData->size();
	}

	// Initialize moment vectors (weights + norm params)
	uint nTotal = theMlp->nWeights();
	for (uint i = 0; i < theMlp->nLayers(); ++i)
		nTotal += theMlp->layer(i).nNormParams();
	if (theM.size() != nTotal) {
		theM.assign(nTotal, 0.0);
		theV.assign(nTotal, 0.0);
		theTimestep = 0;
	}

	double err = theError->outputError(*theMlp, *theData);
	double prevErr = err + 1;
	uint cntr = theNumEpochs;
	const uint w = 14;
	const uint maxRounds = 2;
	uint nrounds = 0;
	DataSet blockData(*theData);
	uint index = 0;

	os.setf(ios::left);
	os << setw(w) << "# Epoch" << setw(w) << "TrnErr" << setw(w) << "LrnRate" << endl;
	do {
		if (buildBlock(blockData, index) == true) ++nrounds;
		trainEpoch(blockData);
		if (nrounds >= maxRounds) {
			cntr--;
			nrounds = 0;
			prevErr = err;
			err = theError->outputError(*theMlp, *theData);
			if (cntr % 100 == 0)
				os << setw(w) << theNumEpochs - cntr << setw(w) << err << setw(w) << theLearningRate
				   << endl;
		}
	} while (cntr && !hasConverged(err, prevErr));
	os << setw(w) << theNumEpochs - cntr << setw(w) << err << setw(w) << theLearningRate << endl;
}

unique_ptr<Trainer> Adam::clone() const {
	return unique_ptr<Trainer>(new Adam(*this));
}

// PRIVATE--------------------------------------------------------------------//

Adam::Adam(const Adam& a) : Trainer(a) {
	*this = a;
}

Adam& Adam::operator=(const Adam& a) {
	if (this != &a) {
		Trainer::operator=(a);
		theLearningRate = a.theLearningRate;
		theBeta1 = a.theBeta1;
		theBeta2 = a.theBeta2;
		theEpsilon = a.theEpsilon;
		theWeightDecay = a.theWeightDecay;
		theM = a.theM;
		theV = a.theV;
		theTimestep = a.theTimestep;
	}
	return *this;
}

double Adam::trainEpoch(DataSet& dset) {
	theError->mlp(*theMlp);
	theError->dset(dset);
	double err = theError->gradient();

	theTimestep++;
	const double lr = theLearningRate;
	const double b1 = theBeta1;
	const double b2 = theBeta2;
	const double eps = theEpsilon;
	const double wd = theWeightDecay;
	const double bc1 = 1.0 / (1.0 - pow(b1, theTimestep));
	const double bc2 = 1.0 / (1.0 - pow(b2, theTimestep));

	uint offset = 0;
	for (uint layer = 0; layer < theMlp->nLayers(); ++layer) {
		Layer& l = theMlp->layer(layer);
		const uint nw = l.nWeights();
		double* __restrict__ wt = l.weights().data();
		double* __restrict__ g = l.gradients().data();
		double* __restrict__ m = theM.data() + offset;
		double* __restrict__ v = theV.data() + offset;

		for (uint j = 0; j < nw; ++j) {
			m[j] = b1 * m[j] + (1.0 - b1) * g[j];
			v[j] = b2 * v[j] + (1.0 - b2) * g[j] * g[j];
			double mHat = m[j] * bc1;
			double vHat = v[j] * bc2;
			if (wd > 0.0) wt[j] -= lr * wd * wt[j];
			wt[j] -= lr * mHat / (sqrt(vHat) + eps);
		}
		offset += nw;
		// Update norm params (no weight decay applied)
		if (l.normType() != NormType::None) {
			const uint nn = l.nNeurons();
			double* gm = l.gamma().data();
			double* gg = l.gammaGradients().data();
			double* bt = l.beta().data();
			double* bg = l.betaGradients().data();
			double* m_ptr = theM.data() + offset;
			double* v_ptr = theV.data() + offset;
			for (uint j = 0; j < nn; ++j) {
				m_ptr[j] = b1 * m_ptr[j] + (1 - b1) * gg[j];
				v_ptr[j] = b2 * v_ptr[j] + (1 - b2) * gg[j] * gg[j];
				gm[j] -= lr * (m_ptr[j] * bc1) / (sqrt(v_ptr[j] * bc2) + eps);
			}
			m_ptr += nn;
			v_ptr += nn;
			for (uint j = 0; j < nn; ++j) {
				m_ptr[j] = b1 * m_ptr[j] + (1 - b1) * bg[j];
				v_ptr[j] = b2 * v_ptr[j] + (1 - b2) * bg[j] * bg[j];
				bt[j] -= lr * (m_ptr[j] * bc1) / (sqrt(v_ptr[j] * bc2) + eps);
			}
			offset += 2 * nn;
		}
	}
	return err;
}

bool Adam::buildBlock(DataSet& blockData, uint& cntr) const {
	bool roundabout = false;
	vector<uint> indices(theBatchSize, 0);
	for (uint i = 0; i < theBatchSize; ++i, ++cntr) {
		if (cntr >= theData->indices().size()) {
			cntr = 0;
			roundabout = true;
		}
		indices.at(i) = theData->indices().at(cntr);
	}
	blockData.indices(indices);
	return roundabout;
}
