#include "TanHypLayer.hh"

#include <cmath>

using namespace MultiLayerPerceptron;

TanHypLayer::TanHypLayer(uint nc, uint np):Layer(nc, np, TANHYP)
{}

TanHypLayer::~TanHypLayer()
{}

//ACCESSOR AND MUTATOR FUNCTIONS

//ACCESSOR FUNCTIONS

//COUNTS AND CRAP

//PRINTS

//UTILS

double TanHypLayer::fire(double lif)
{return tanh(lif);}

double TanHypLayer::fire(uint i)
{
	assert(i<theOutputs.size());
	return theOutputs[i];
}

double TanHypLayer::firePrime(double lif)
{
	double tmp = fire(lif);
	return 1.0 - pow(tmp, 2);
}

double TanHypLayer::firePrime(uint i)
{
	assert(i<theOutputs.size());
	return 1.0 - pow(theOutputs[i], 2);
}

//PRIVATE--------------------------------------------------------------------//

