#include "Roc.hh"
#include "Evaluator.hh"

#include <algorithm>
#include <functional>
#include <iomanip>

using namespace EvalTools;

Roc::Roc():theRoc(0), theAuc(0)
{theEval=new Evaluator();}

Roc::Roc(const Roc& roc){*this=roc;}

Roc::~Roc(){delete theEval;}

Roc& Roc::operator=(const Roc& roc)
{
	if(this!=&roc){
		theRoc=roc.theRoc;
		theAuc=roc.theAuc;
		theEval=new Evaluator(*(roc.theEval));
	}
	return *this;
}

vector< pair<double,double> >& Roc::roc(){return theRoc;}

double Roc::auc()
{
	return theAuc;
}

double Roc::calcAucWmw(vector<double>& out, vector<uint>& dout)
{
	/*	
	std::cout<<"output: ";
	printVector(out);
	std::cout<<"Desired output: ";
	printVector(dout);
	*/
	assert(out.size()==dout.size());
	vector<double> posOut(0);
	vector<double> negOut(0);
	for(uint i=0; i<out.size(); ++i){
		if(dout[i] == POS)
			posOut.push_back(out[i]);
		else
			negOut.push_back(out[i]);
	}
	sort(posOut.begin(), posOut.end());
	sort(negOut.begin(), negOut.end());

	/*
	std::cout<<"Positive output: ";
	printVector(posOut);
	std::cout<<"Negative output: ";
	printVector(negOut);
	*/
	vector<double> rank(0);
	uint j=0;
	for(uint i=0; i<posOut.size(); ++i){
		while(posOut[i]>=negOut[j]){
			if(j<negOut.size()){
				j++;
			}
			else
				break;
		}       
		rank.push_back(i+j);
	}

	assert(rank.size()==posOut.size());
	uint m = posOut.size();
	uint n = negOut.size();
	/*
	std::cout<<"The n is: "<<n<<std::endl;
	std::cout<<"The m is: "<<m<<std::endl;
	std::cout<<"The rank is: ";
	printVector(rank);
	*/
	double sum = 0;
	for(uint i=0; i<rank.size(); ++i)
		sum += rank[i];
	theAuc = (sum-(m*(m-1.0))*0.5)/(m*n);
	return theAuc;
}

double Roc::calcAucTrapezoidal(vector<double>& out, vector<uint>& dout)
{
	double area = 0;
	calcRoc(out, dout);
	vector< pair<double, double> >::iterator it;
	for(it=theRoc.begin()+1; it!=theRoc.end(); ++it){
		double x1 = (it-1)->first;
		double y1 = (it-1)->second;
		double x2 = it->first;
		double y2 = it->second;
		area += (x2-x1)*0.5*(y1+y2);
	}
	return theAuc = area;
}

void Roc::calcRoc(vector<double>& out, vector<uint>& dout)
{
	theRoc = vector< pair<double,double> >(0);
	pair<double,double> tmp;
	for(uint i=0; i<out.size(); ++i){
		theEval->cut(out[i]);
		theEval->evaluate(out, dout);
		tmp.first = theEval->fpf();
		tmp.second = theEval->tpf();
		theRoc.push_back(tmp);
	}
	sort(theRoc.begin(), theRoc.end());
}

void Roc::print(ostream& os)
{
	if(!os){
		std::cerr<<"Output stream error.\n";
		return;
	}
	os<<"#Spec\tSens"<<endl;
	vector< pair<double,double> >::iterator it;
	for(it=theRoc.begin(); it!=theRoc.end(); ++it)
		os<<std::setprecision(6)<<it->first<<"\t"<<it->second<<endl;
}

void Roc::printVector(vector<uint>& vec)
{
	vector<uint>::iterator it;
	for(it = vec.begin(); it != vec.end(); ++it)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}

void Roc::printVector(vector<double>& vec)
{
	vector<double>::iterator it;
	for(it = vec.begin(); it != vec.end(); ++it)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}

//PRIVATE---------------------------------------------------------------------//

