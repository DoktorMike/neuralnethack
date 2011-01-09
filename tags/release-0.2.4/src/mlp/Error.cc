#include "Error.hh"

#include <vector>
#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

Error::Error(Mlp* mlp, DataSet* dset)
	:theMlp(mlp),theDset(dset),
	theWeightElimOn(false), theWeightElimAlpha(0), theWeightElimW0(0){}

Error::~Error(){}

Mlp* Error::mlp(){return theMlp;}

void Error::mlp(Mlp* mlp){theMlp = mlp;}

DataSet* Error::dset(){return theDset;}

void Error::dset(DataSet* dset){theDset = dset;}

bool Error::weightElimOn() const {return theWeightElimOn;}

void Error::weightElimOn(bool on){theWeightElimOn = on;}

double Error::weightElimAlpha() const {return theWeightElimAlpha;}

void Error::weightElimAlpha(double alpha){theWeightElimAlpha = alpha;}

double Error::weightElimW0() const {return theWeightElimW0;}

void Error::weightElimW0(double w0){theWeightElimW0 = w0;}

double Error::weightElimGrad(double wi)
{
	double alpha = theWeightElimAlpha;
	double w0 = theWeightElimW0;
	return alpha*( (2*wi*pow(w0,2))/pow( pow(w0,2)+pow(wi,2), 2) );
}

void Error::weightElimGrad(vector<double>& gradients, uint offset, uint length)
{
	for(uint i=offset; i<offset+length; ++i)
		gradients[i] += weightElimGrad(gradients[i]);
}

void Error::weightElimGradLayer(vector<double>& gradients, uint ncurr, uint nprev)
{
	uint offset = 0;
	for(uint i = 0; i<ncurr; ++i){
		weightElimGrad(gradients, offset, nprev);
		offset += nprev + 1;
	}
}

void Error::weightElimGradMlp(vector<double>& gradients, vector<uint>& arch)
{
	uint offset = 0;
	for(uint i=1; i<arch.size(); ++i){
		for(uint j=0; j<arch[i]; ++j){
			weightElimGrad(gradients, offset, arch[i-1]);
			offset += arch[i-1]+1; //avoid the bias
		}
	}
}

void Error::weightElimGrad()
{
	assert(theMlp != 0);
	
	if(weightElimOn() == true){
		vector<Layer*>& layers = theMlp->layers();
		for(uint i=0; i<layers.size(); ++i){
			Layer* l = layers[i];
			weightElimGradLayer(l->gradients(), l->nNeurons(), l->nPrevious());
		}
	}
}

double Error::weightElim() const
{
	vector<double> weights = theMlp->weights();
	double we = 0;
	for(vector<double>::iterator itw = weights.begin(); itw != weights.end(); ++itw){
		double wisqr = pow(*itw, 2);	
		double w0sqr = pow(weightElimW0(), 2);
		we += wisqr/(w0sqr + wisqr);
	}
	return we;
}

//PRIVATE--------------------------------------------------------------------//

Error::Error(const Error& err)
{*this = err;}

Error& Error::operator=(const Error& err)
{
	if(this != &err){
		theMlp = err.theMlp;
		theDset = err.theDset;
	}
	return *this;
}

