/*$Id: Roc.cc 1667 2007-08-23 11:58:24Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#include "Roc.hh"
#include "Evaluator.hh"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <ostream>
#include <iterator>

using namespace EvalTools;
using std::ostream_iterator;
using std::ostream;
using std::setprecision;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::pair;
using std::copy;

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

double Roc::calcAucWmw(vector<double>& out, vector<uint>& dout)
{
	if(out.size()!=dout.size()){
		cerr<<"Error: output and target vectors must have the same size"<<endl; 
		abort();
	}
	vector<double> posOut(0); vector<double> negOut(0);
	for(uint i=0; i<out.size(); ++i)
		if(dout[i] > 0) 
			posOut.push_back(out[i]);
		else 
			negOut.push_back(out[i]);
	uint m = posOut.size(); uint n = negOut.size(); double r = 0;
	for(uint i=0; i<m; ++i)
		for(uint j=0; j<n; ++j)
			if(posOut[i] > negOut[j]) 
				r += 1.0;
			//else if(posOut[i] == negOut[i]) 
			//	r += 0.5; //Try to compensate for similar outputs.

//	cout<<"The n is: "<<n<<"\nThe m is: "<<m<<"\nThe rank is: "<<r<<endl;
	return theAuc = r/(m*n);
}

double Roc::calcAucWmwFast(vector<double>& out, vector<uint>& dout)
{
	if(out.size()!=dout.size()){
		cerr<<"Error: output and target vectors must have the same size"<<endl; 
		abort();
	}
	uint m=0; uint n=0;
	vector< pair<double, uint> > rank(0);
	for(uint i=0; i<out.size(); ++i){
		if(dout[i] > 0) m++; else n++;
		rank.push_back(pair<double, uint>(out[i], dout[i]));
	}
	sort(rank.begin(), rank.end());

	uint r=0;
	for(uint i=0; i<rank.size(); ++i)
		if(rank[i].second > 0) r += i;

//	cout<<"The n is: "<<n<<"\nThe m is: "<<m<<"\nThe rank is: "<<r<<endl;
	return theAuc = (r - m*(m-1.0)*0.5)/(double)(m*n);
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
		cerr<<"Output stream error.\n";
		return;
	}
	os<<"#Spec\tSens"<<endl;
	vector< pair<double,double> >::iterator it;
	for(it=theRoc.begin(); it!=theRoc.end(); ++it)
		os<<setprecision(6)<<it->first<<"\t"<<it->second<<endl;
}

//PRIVATE---------------------------------------------------------------------//

template<class T>
void Roc::printVector(vector<T>& vec)
{
	typename vector<T>::iterator it;
	copy(vec.begin(), vec.end(), ostream_iterator<T>(cout, " "));
	for(it = vec.begin(); it != vec.end(); ++it) 
		cout<<*it<<" ";
	cout<<endl;
}
