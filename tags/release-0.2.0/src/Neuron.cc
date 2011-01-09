#include "Neuron.hh"

using namespace NetHack;

Neuron::Neuron(uint nprev)
{
    theWeights = vector<double>(nprev+1);
    theLif = 0;
    theActivation = 0;
    theLocalGradient = 0;
    theGradients = vector<double>(nprev+1);
    thePrevWeightUpd = vector<double>(nprev+1);
    //srand(1);//time(0));
    for(uint i=0; i<nprev+1; i++)
	theWeights[i] = 0.5-((double)rand()/RAND_MAX);//rand()%10;
}

Neuron::Neuron(const Neuron& n)
{
    if(this!=&n){
	theWeights = n.theWeights;
	theType = n.theType;
	theLif = n.theLif;
	theActivation = n.theActivation;
	theLocalGradient = n.theLocalGradient;
	theGradients = n.theGradients;
	thePrevWeightUpd = n.thePrevWeightUpd;
    }
}

Neuron::~Neuron(){}

double& Neuron::operator[](const uint i)
{
    assert(theWeights.begin()+i < theWeights.end());
    return theWeights[i];
}

double Neuron::potential(vector<double>& input)
{
    //cout<<"Entering Neuron::potential\n";
    vector<double>::iterator itw = theWeights.begin();
    vector<double>::iterator ita = input.begin();
    theLif = 0;

    for(; ita!=input.end(); ++ita, ++itw){
	theLif+=(*ita)*(*itw);	
	//cout<<"ita: "<<*ita<<" itw: "
	//	<<*itw<<" lif: "<<theLif<<endl;
    }
    theLif += *itw; //bias
    return theLif;
}

void Neuron::printWeights()
{
    vector<double>::iterator it;
    for(it=theWeights.begin(); it!=theWeights.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
}

uint Neuron::nWeights(){return theWeights.size();}

vector<double>& Neuron::weights(){return theWeights;}

double Neuron::weights(uint prev)
{
    assert(theWeights.begin()+prev < theWeights.end());
    return theWeights[prev];
}

void Neuron::weights(vector<double>& w)
{
    assert(theWeights.size()==w.size());
    theWeights.assign(w.begin(), w.end());
}

vector<double>::iterator Neuron::weights(vector<double>::iterator f)
{
    vector<double>::iterator it1=theWeights.begin();
    vector<double>::iterator it2=f;
    for(;it1!=theWeights.end(); ++it1, ++it2)
	*it1 = *it2;
    return it2;
}

void Neuron::updateWeights(vector<double>& w)
{
    assert(theWeights.size()==w.size());
    vector<double>::iterator it1=theWeights.begin();
    vector<double>::iterator it2=w.begin();
    for(;it1!=theWeights.end(); ++it1, ++it2)
	*it1 += *it2;
}

vector<double>::iterator Neuron::updateWeights(vector<double>::iterator f)
{
    vector<double>::iterator it1=theWeights.begin();
    vector<double>::iterator it2=f;
    for(;it1!=theWeights.end(); ++it1, ++it2)
	*it1 += *it2;
    return it2;
}

string Neuron::type()
{return theType;}

double Neuron::lif()
{return theLif;}

double Neuron::activation()
{return theActivation;}

double Neuron::localGradient()
{return theLocalGradient;}

void Neuron::localGradient(double e)
{theLocalGradient=e;}

vector<double>& Neuron::gradients()
{return theGradients;}

double Neuron::gradient(uint prev)
{
    assert(theGradients.begin()+prev < theGradients.end());
    return theGradients[prev];
}

void Neuron::gradients(const vector<double>& e)
{theGradients=e;}

void Neuron::gradient(uint index, double g)
{
    assert(index<theGradients.size());
    theGradients[index]=g;
}

void Neuron::printGradients()
{
    vector<double>::iterator it;
    for(it=theGradients.begin(); it!=theGradients.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
}

double Neuron::prevWeightUpd(uint prev)
{
    assert(thePrevWeightUpd.begin()+prev < thePrevWeightUpd.end());
    return thePrevWeightUpd[prev];
}

void Neuron::prevWeightUpd(const vector<double>& u)
{thePrevWeightUpd=u;}

void Neuron::prevWeightUpd(uint index, double u)
{thePrevWeightUpd[index]=u;}

void Neuron::printPrevWeightUpd()
{
    vector<double>::iterator it;
    for(it=thePrevWeightUpd.begin(); it!=thePrevWeightUpd.end(); ++it)
	cout<<*it<<" ";
    cout<<endl;
}


//PRIVATE--------------------------------------------------------------------//


