#include "Layer.hh"

using namespace MultiLayerPerceptron;

Layer::Layer(uint nc, uint np, string t):
	ncurr(nc),
	nprev(np),
	theType(t),
	theWeights(ncurr*(nprev+1), 0),
	theOutputs(ncurr,0),
	theLocalGradients(ncurr,0),
	theGradients(ncurr*(nprev+1), 0),
	theWeightUpdates(ncurr*(nprev+1), 0)
{
	regenerateWeights();
}

Layer::Layer(const Layer& layer)
{*this = layer;}

Layer::~Layer()
{
}

Layer& Layer::operator=(const Layer& layer)
{
	if(this != &layer){
		ncurr=layer.ncurr;
		nprev=layer.nprev;
		theType=layer.theType;
		theWeights=layer.theWeights;
		theOutputs=layer.theOutputs;
		theLocalGradients=layer.theLocalGradients;
		theGradients=layer.theGradients;
		theWeightUpdates=layer.theWeightUpdates;
	}
	return *this;
}

double& Layer::operator[](const uint i)
{
	assert(i < theOutputs.size());
	return theOutputs[i];
}

//ACCESSOR AND MUTATOR FUNCTIONS

vector<double>& Layer::weights()
{return theWeights;}

void Layer::weights(vector<double>& w)
{theWeights = w;}

vector<double>& Layer::outputs()
{return theOutputs;}

vector<double>& Layer::localGradients()
{return theLocalGradients;}

vector<double>& Layer::gradients()
{return theGradients;}

vector<double>& Layer::weightUpdates()
{return theWeightUpdates;}

//ACCESSOR FUNCTIONS
double& Layer::weights(uint i, uint j)
{return theWeights[index(i,j)];}

double& Layer::weights(uint i)
{
	assert(i < theWeights.size());
	return theWeights[i];
}

double& Layer::outputs(uint i)
{
	assert(i < theOutputs.size());
	return theOutputs[i];
}

double& Layer::localGradients(uint i)
{
	assert(i < theLocalGradients.size());
	return theLocalGradients[i];
}

double& Layer::gradients(uint i, uint j)
{return theGradients[index(i,j)];}

double& Layer::gradients(uint i)
{
	assert(i < theGradients.size());
	return theGradients[i];
}

double& Layer::weightUpdates(uint i, uint j)
{return theWeightUpdates[index(i,j)];}

double& Layer::weightUpdates(uint i)
{
	assert(i < theWeightUpdates.size());
	return theWeightUpdates[i];
}

//COUNTS AND CRAP

uint Layer::nWeights()
{return theWeights.size();}

uint Layer::nNeurons()
{return ncurr;}

uint Layer::size()
{return nNeurons();}

//PRINTS

void Layer::printWeights()
{
	vector<double>::iterator it = theWeights.begin();
	for(; it != theWeights.end(); ++it)
		cout<<*it<<" ";
	cout<<"\n";
}

void Layer::printGradients()
{
	vector<double>::iterator it = theGradients.begin();
	for(; it != theGradients.end(); ++it)
		cout<<*it<<" ";
	cout<<"\n";
}

//UTILS

void Layer::regenerateWeights()
{
	//srand(1);//time(0));
	vector<double>::iterator it;
	for(it = theWeights.begin(); it != theWeights.end(); ++it)
		*it = 0.5-((double)rand()/RAND_MAX);
}

vector<double>& Layer::propagate(vector<double>& input)
{
	vector<double>::iterator itw = theWeights.begin();
	vector<double>::iterator ito, iti;
	for(ito = theOutputs.begin(); ito != theOutputs.end(); ++ito){
		*ito = 0;
		for(iti = input.begin(); iti != input.end(); ++iti, ++itw)
			*ito += *itw * *iti;
		*ito += *itw;
		itw++;
		*ito = fire(*ito);
	}
	return theOutputs;
}


//PRIVATE--------------------------------------------------------------------//

uint Layer::index(uint i, uint j)
{
	assert(i < ncurr && j < nprev+1);
	return i*(nprev+1)+j;
}

