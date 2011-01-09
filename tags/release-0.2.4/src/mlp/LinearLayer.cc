#include "LinearLayer.hh"

#include <cassert>

using namespace MultiLayerPerceptron;

LinearLayer::LinearLayer(uint nc, uint np):Layer(nc, np, LINEAR)
{}

LinearLayer::~LinearLayer()
{}

//ACCESSOR AND MUTATOR FUNCTIONS

//ACCESSOR FUNCTIONS

//COUNTS AND CRAP

//PRINTS

//UTILS

double LinearLayer::fire(double lif)
{return lif;}

double LinearLayer::fire(uint i)
{
	assert(i<theOutputs.size());
	return theOutputs[i];
}

double LinearLayer::firePrime(double lif)
{return 1.0;}

double LinearLayer::firePrime(uint i)
{
	assert(i<theOutputs.size());
	return 1.0;
}

//PRIVATE--------------------------------------------------------------------//

