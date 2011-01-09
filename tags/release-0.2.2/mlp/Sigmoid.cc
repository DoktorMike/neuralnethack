#include "Sigmoid.hh"

#include <ctime>
#include <cmath>

using namespace MultiLayerPerceptron;

Sigmoid::Sigmoid(uint nprev):Neuron(nprev)
{
    theType = SIGMOID;
}

Sigmoid::Sigmoid(const Sigmoid& n):Neuron(n)
{*this = n;}

Sigmoid::~Sigmoid(){}

Sigmoid& Sigmoid::operator=(const Sigmoid& n)
{
    if(this != &n){
    }
    return *this;
}

double Sigmoid::fire(vector<double>& input)
{
    potential(input);
    return theActivation = 1.0/(1.0+exp(-theLif));
}

double Sigmoid::firePrime(vector<double>& input)
{
    double tmp = fire(input);
    return tmp*(1-tmp);
}

double Sigmoid::firePrime()
{return theActivation*(1-theActivation);}

//PRIVATE--------------------------------------------------------------------//


