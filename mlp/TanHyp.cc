#include "TanHyp.hh"

#include <ctime>
#include <cmath>

using namespace MultiLayerPerceptron;

TanHyp::TanHyp(uint nprev):Neuron(nprev)
{
    theType = TANHYP;
}

TanHyp::TanHyp(const TanHyp& n):Neuron(n)
{*this = n;}

TanHyp::~TanHyp(){}

TanHyp& TanHyp::operator=(const TanHyp& n)
{
    if(this != &n){
    }
    return *this;
}

double TanHyp::fire(vector<double>& input)
{
    potential(input);
    return theActivation = tanh(theLif);
}

double TanHyp::firePrime(vector<double>& input)
{
    fire(input);
    return 1-pow(theActivation,2);
}

double TanHyp::firePrime()
{return 1-pow(theActivation,2);}

//PRIVATE--------------------------------------------------------------------//


