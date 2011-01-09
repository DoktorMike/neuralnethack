#include "Evaluator.hh"
#include <cmath>

using namespace EvalTools;

Evaluator::Evaluator()
    :theTnf(0), theTpf(0), theCut(0), 
    nTp(0), nTn(0), nP(0), nN(0)
{}

Evaluator::Evaluator(const Evaluator& eval){*this=eval;}

Evaluator::~Evaluator(){}

Evaluator& Evaluator::operator=(const Evaluator& eval)
{
    if(this!=&eval){
	theTnf=eval.theTnf;
	theTpf=eval.theTpf;
	theCut=eval.theCut;
	nTp=eval.nTp;
	nTn=eval.nTn;
	nP=eval.nP;
	nN=eval.nN;
    }
    return *this;
}

double Evaluator::tpf(){return theTpf;}

double Evaluator::fnf(){return 1.0 - theTpf;}

double Evaluator::tnf(){return theTnf;}

double Evaluator::fpf(){return 1.0 - theTnf;}

double Evaluator::cut(){return theCut;}

void Evaluator::cut(double c){theCut=c;}

void Evaluator::evaluate(vector<double>& out, vector<uint>& dout)
{
    assert(out.size()==dout.size());
    reset();
    vector<uint> o=cutOutput(out);
    vector<uint>::iterator ito=o.begin();
    vector<uint>::iterator itd=dout.begin();

    for(; itd!=dout.end(); ++itd, ++ito)
	switch(*itd){
	    case NEG:
		++nN;
		if(*ito==*itd)
		    ++nTn;
		break;
	    case POS:
		++nP;
		if(*ito==*itd)
		    ++nTp;
		break;
	}
    calcRates();
}

void Evaluator::print(ostream& os)
{
    if(!os){
	std::cerr<<"Output stream error.";
	abort();
    }
    os<<"\tTrue Positive Fraction: "<<theTpf<<endl;
    os<<"\tTrue Negative Fraction: "<<theTnf<<endl;
}

//PRIVATE--------------------------------------------------------------------//

void Evaluator::reset()
{
    nTp=0; 
    nTn=0; 
    nP=0;
    nN=0;
}

vector<uint> Evaluator::cutOutput(vector<double>& out)
{
    vector<uint> tmp(0);
    vector<double>::iterator it;
    for(it=out.begin(); it!=out.end(); ++it){
	if(*it<theCut)
	    tmp.push_back(NEG);
	else
	    tmp.push_back(POS);
    }
    assert(tmp.size()==out.size());
    return tmp;
}

vector<uint> Evaluator::vectorDoubleToUint(vector<double>& vec)
{
    vector<uint> tmp(0);
    vector<double>::iterator it;
    for(it=vec.begin(); it!=vec.end(); ++it)
	tmp.push_back((uint)round(*it));
    return tmp;
}

void Evaluator::calcRates()
{
    theTnf=nTn/(double)nN;
    theTpf=nTp/(double)nP;
}

