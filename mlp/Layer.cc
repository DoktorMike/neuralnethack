#include "Layer.hh"
#include "Sigmoid.hh"
#include "TanHyp.hh"
#include "Linear.hh"

using namespace MultiLayerPerceptron;

Layer::Layer(int ncurr, int nprev, string theType):theOutput(ncurr,0),
theBOutput(ncurr+1,0)
{
	this->ncurr = ncurr;
	this->nprev = nprev;
	this->theType = theType;

	createNeurons();
}

Layer::Layer(const Layer& layer)
{*this = layer;}

Layer::~Layer()
{
	vector<Neuron*>::iterator it = theNeurons.begin();
	for(; it!=theNeurons.end(); it++)
		delete (*it);
}

Layer& Layer::operator=(const Layer& layer)
{
	if(this != &layer){
		ncurr=layer.ncurr;
		nprev=layer.nprev;
		theType=layer.theType;
		theNeurons = vector<Neuron*>(0);

		createNeurons();
		assert(theNeurons.size() == layer.theNeurons.size());
		for(int i=0; i<ncurr; ++i)
			*(theNeurons[i]) = *(layer.theNeurons[i]);

		theOutput=layer.theOutput;
		theBOutput=layer.theBOutput;
	}
	return *this;
}

Neuron& Layer::operator[](const uint i)
{return neuron(i);}

vector<Neuron*>& Layer::neurons(){return theNeurons;}

Neuron& Layer::neuron(uint index)
{
	assert(index < theNeurons.size());
	return *(theNeurons[index]);
}

vector<double>& Layer::propagate(vector<double>& input)
{
	vector<Neuron*>::iterator itn;
	vector<double>::iterator ito = theOutput.begin();

	for(itn=theNeurons.begin(); itn!=theNeurons.end(); ++itn, ++ito){
		*ito = (*itn)->fire(input);
		//cout<<"Layer output: "<<*ito<<endl;
	}
	return theOutput;
}

vector<double>& Layer::output()
{return theOutput;}

vector<double>& Layer::bOutput()
{
	theBOutput = theOutput;
	theBOutput.assign(theOutput.begin(), theOutput.end());
	theBOutput.push_back(1.0);
	return theBOutput;
}

vector<double> Layer::weights()
{
	vector<double> w(0);
	vector<Neuron*>::iterator it;

	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it){
		vector<double> tmp = (*it)->weights();
		w.insert(w.end(),tmp.begin(),tmp.end());
	}
	return w;
}

vector<double> Layer::gradients()
{
	vector<double> g(0);
	vector<Neuron*>::iterator it;

	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it){
		vector<double> tmp = (*it)->gradients();
		g.insert(g.end(),tmp.begin(),tmp.end());
	}
	return g;
}

vector<double>::iterator Layer::weights(vector<double>::iterator f)
{
	vector<Neuron*>::iterator it;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		f=(*it)->weights(f);
	return f;
}

vector<double>::iterator Layer::updateWeights(vector<double>::iterator f)
{
	vector<Neuron*>::iterator it;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		f=(*it)->updateWeights(f);
	return f;
}

void Layer::regenerateWeights()
{
	vector<Neuron*>::iterator it;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		(*it)->regenerateWeights();
}

void Layer::weights(vector<double>& w)
{
	vector<Neuron*>::iterator it;
	vector<double>::iterator f = w.begin();
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		f=(*it)->weights(f);
}

void Layer::updateWeights(vector<double>& w)
{
	vector<Neuron*>::iterator it;
	vector<double>::iterator f = w.begin();
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		f=(*it)->updateWeights(f);
}

uint Layer::nWeights()
{
	vector<Neuron*>::iterator it;
	uint tmp = 0;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		tmp += (*it)->nWeights();
	return tmp;
}

uint Layer::nNeurons()
{return theNeurons.size();}

uint Layer::size()
{return nNeurons();}

void Layer::printWeights()
{
	vector<Neuron*>::iterator it;

	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		(*it)->printWeights();
	cout<<endl;
}

void Layer::printLocalGradients()
{
	cout<<"Printing local gradients.\n";
	vector<Neuron*>::iterator it;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it){
		cout<<(*it)->localGradient()<<" ";
	}
}

void Layer::printGradients()
{
	cout<<"Printing gradients.\n";
	vector<Neuron*>::iterator it;
	for(it=theNeurons.begin(); it!=theNeurons.end(); ++it)
		(*it)->printGradients();
}


//PRIVATE--------------------------------------------------------------------//

void Layer::createNeurons()
{
	for(int i=0; i<ncurr; i++){
		if(theType==SIGMOID){
			//cout<<"Sigmoid Selected!\n";
			theNeurons.push_back(new Sigmoid(nprev));
		}else if(theType == TANHYP){
			//cout<<"TanHyp Selected!\n";
			theNeurons.push_back(new TanHyp(nprev));
		}else if(theType==LINEAR){
			//cout<<"Linear Selected!\n";
			theNeurons.push_back(new Linear(nprev));
		}
	}
}
