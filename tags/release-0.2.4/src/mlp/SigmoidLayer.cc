#include "SigmoidLayer.hh"

#include <cmath>
#include <cassert>

using namespace MultiLayerPerceptron;

SigmoidLayer::SigmoidLayer(uint nc, uint np):Layer(nc, np, SIGMOID)
{}

SigmoidLayer::~SigmoidLayer()
{}

//ACCESSOR AND MUTATOR FUNCTIONS

//ACCESSOR FUNCTIONS

//COUNTS AND CRAP

//PRINTS

//UTILS

double SigmoidLayer::fire(double lif)
{return 1.0/(1.0+exp(-lif));}

double SigmoidLayer::fire(uint i)
{
	assert(i<theOutputs.size());
	return theOutputs[i];
}

double SigmoidLayer::firePrime(double lif)
{
	double tmp = fire(lif);
	return tmp*(1-tmp);
}

double SigmoidLayer::firePrime(uint i)
{
	assert(i<theOutputs.size());
	return theOutputs[i]*(1-theOutputs[i]);
}

//PRIVATE--------------------------------------------------------------------//

