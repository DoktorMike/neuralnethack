#include "Committee.hh"
#include "MatrixTools.hh"

using namespace NetHack;
using namespace MatrixTools;

Committee::Committee(Mlp& mlp, double s):theCommittee(1,mlp), theScales(1,s){}

Committee::Committee(const Committee& c){*this=c;}

Committee::~Committee(){}

Committee& Committee::operator=(const Committee& c)
{
    if(this!=&c){
	theCommittee=c.theCommittee;
	theScales=c.theScales;
    }
    return *this;
}

Mlp& Committee::operator[](const uint i)
{return mlp(i);}

Mlp& Committee::mlp(const uint i)
{
    assert(i<theCommittee.size());
    return theCommittee[i];
}

void Committee::delMlp(const uint i)
{
    assert(i<theCommittee.size());
    vector<Mlp>::iterator it=theCommittee.begin();
    theCommittee.erase(it+i);
}

void Committee::addMlp(Mlp& mlp, double s)
{
    theCommittee.push_back(mlp);
    theScales.push_back(s);
}
	    
void Committee::addMlp(Mlp& mlp)
{
    theCommittee.push_back(mlp);
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
    assert(theCommittee.size()==theScales.size());
    vector<Mlp>::iterator itm=theCommittee.begin();
    vector<double>::iterator its=theScales.begin();
    vector<double> output=itm->propagate(input);
    ++itm;
    mul(output,*its);
    ++its;
    for(; itm!=theCommittee.end(); ++itm, ++its){
	vector<double> tmp=itm->propagate(input);
	mul(tmp,*its);
	add(output, tmp, output);
    }
    return output;
}

