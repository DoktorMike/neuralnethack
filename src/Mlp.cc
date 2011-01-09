#include "Mlp.hh"

using namespace NetHack;

Mlp::Mlp(vector<uint>& theArch, vector<string>& theTypes, 
	bool softmax)
{
    this->softmax = softmax;
    this->theArch = theArch;
    this->theTypes = theTypes;

    createLayers();
}

Mlp::Mlp(const Mlp& mlp)
{*this = mlp;}

Mlp::~Mlp()
{
    vector<Layer*>::iterator it = theLayers.begin();
    for(; it!=theLayers.end(); it++)
	delete (*it);
}

Mlp& Mlp::operator=(const Mlp& mlp)
{
    if(this != &mlp){
	theArch = mlp.theArch;
	theTypes = mlp.theTypes;
	softmax = mlp.softmax;
	theLayers = vector<Layer*>(0);
	createLayers();
	assert(theLayers.size()==mlp.theLayers.size());
	for(uint i=0; i<theTypes.size(); ++i)
	    *(theLayers[i]) = *(mlp.theLayers[i]);
    }
    return *this;
}


Layer& Mlp::operator[](const uint i)
{
    assert(i < theLayers.size());
    return *(theLayers[i]);
}

void Mlp::printWeights()
{
    vector<Layer*>::iterator itl;
    for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
	(*itl)->printWeights();
}

vector<double> Mlp::weights()
{
    vector<double> w(0);
    vector<Layer*>::iterator it;

    for(it=theLayers.begin(); it!=theLayers.end(); ++it){
	vector<double> tmp = (*it)->weights();
	w.insert(w.end(),tmp.begin(),tmp.end());
    }
    return w;
}

vector<double> Mlp::gradients()
{
    vector<double> g(0);
    vector<Layer*>::iterator it;

    for(it=theLayers.begin(); it!=theLayers.end(); ++it){
	vector<double> tmp = (*it)->gradients();
	g.insert(g.end(),tmp.begin(),tmp.end());
    }
    return g;
}

void Mlp::updateWeights(vector<double>& dw)
{
    assert(dw.size()==nWeights());
    vector<Layer*>::iterator itl;
    vector<double>::iterator f = dw.begin();
    for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
	f=(*itl)->updateWeights(f);
}

void Mlp::weights(vector<double>& w)
{
    assert(w.size()==nWeights());
    vector<Layer*>::iterator itl;
    vector<double>::iterator f = w.begin();
    for(itl=theLayers.begin(); itl!=theLayers.end(); ++itl)
	f=(*itl)->weights(f);
}

vector<double> Mlp::propagate(vector<double>& input)
{
    vector<Layer*>::iterator it; 
    vector<double> output = input;

    for(it=theLayers.begin(); it!=theLayers.end(); ++it)
	output = (*it)->propagate(output);
    return output;
}

uint Mlp::nWeights()
{
    vector<Layer*>::iterator it;
    uint tmp=0;
    for(it=theLayers.begin(); it!=theLayers.end(); ++it)
	tmp+=(*it)->nWeights();
    return tmp;
}

uint Mlp::nLayers()
{return theLayers.size();}

uint Mlp::size()
{return nLayers();}

vector<uint>& Mlp::arch()
{return theArch;}

vector<Layer*>& Mlp::layers()
{return theLayers;}

Layer& Mlp::layer(uint index)
{
    assert(index < (theArch.size()-2));
    return *(theLayers[index]);
}

void Mlp::createLayers()
{
    vector<uint>::iterator it;
    int i=0;

    for(it=theArch.begin()+1; it!=theArch.end(); ++it, ++i){
	Layer* l = new Layer(*(it),*(it-1), theTypes.at(i));
	theLayers.push_back(l);	
    }
}

