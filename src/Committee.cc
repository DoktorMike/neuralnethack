#include "Committee.hh"
#include "matrixtools/MatrixTools.hh"

#include <cassert>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace MatrixTools;

Committee::Committee():theCommittee(0), theScales(0){}

Committee::Committee(Mlp& mlp, double s):theCommittee(1, new Mlp(mlp)), theScales(1,s){}

Committee::Committee(const Committee& c){*this=c;}

Committee::~Committee()
{
	vector<Mlp*>::iterator it;
	for(it = theCommittee.begin(); it != theCommittee.end(); ++it)
		delete *it;
}

Committee& Committee::operator=(const Committee& c)
{
	if(this!=&c){
		for(uint i=0; i<c.theCommittee.size(); ++i)
			theCommittee.push_back(new Mlp(*(c.theCommittee[i])));
		theScales=c.theScales;
	}
	return *this;
}

Mlp& Committee::operator[](const uint i)
{return mlp(i);}

Mlp& Committee::mlp(const uint i)
{
	assert(i<theCommittee.size());
	return *(theCommittee[i]);
}

void Committee::delMlp(const uint i)
{
	//I may need to think more here.
	assert(i<theCommittee.size());
	delete theCommittee[i];
	theCommittee.erase(theCommittee.begin()+i);
	theScales.erase(theScales.begin()+i);
}

void Committee::addMlp(Mlp& mlp, double s)
{
	theCommittee.push_back(new Mlp(mlp));
	theScales.push_back(s);
}

void Committee::addMlp(Mlp& mlp)
{
	theCommittee.push_back(new Mlp(mlp));
	theScales.assign(theCommittee.size(), 1.0/theCommittee.size());
}

double Committee::scale(const uint i)
{
	assert(i<theScales.size());
	return theScales[i];
}

void Committee::scale(const uint i, double s)
{
	assert(i<theScales.size());
	theScales[i]=s;
}

uint Committee::size(){return theCommittee.size();}

vector<double> Committee::propagate(vector<double>& input)
{
	assert(theCommittee.size() == theScales.size());
	assert(!theCommittee.empty());
	vector<Mlp*>::iterator itm = theCommittee.begin();
	vector<double>::iterator its = theScales.begin();
	vector<double> output = (*itm)->propagate(input); ++itm;
	mul(output, *its); ++its;
	for(; itm != theCommittee.end(); ++itm, ++its){
		vector<double> tmp = (*itm)->propagate(input);
		mul(tmp, *its);
		add(output, tmp, output);
	}
	return output;
}

