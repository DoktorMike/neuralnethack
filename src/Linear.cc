#include "Linear.hh"

#include <ctime>
#include <cmath>

using namespace NetHack;

Linear::Linear(uint nprev):Neuron(nprev)
{
    theType = LINEAR;
}

Linear::Linear(const Linear& n):Neuron(n)
{*this = n;}

Linear::~Linear(){}

Linear& Linear::operator=(const Linear& n)
{
    if(this != &n){
    }
    return *this;
}

double Linear::fire(vector<double>& input)
{return theActivation = potential(input);}

double Linear::firePrime(vector<double>& input)
{
    fire(input);
    return 1;
}

double Linear::firePrime()
{return 1;}

//PRIVATE--------------------------------------------------------------------//


