/*$Id: Normaliser.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Normaliser.hh"

#include <cmath>
#include <cassert>

using namespace DataTools;
using namespace std;

Normaliser::Normaliser():theStdDev(0), theMean(0), theSkip(0)
{}

Normaliser::Normaliser(const Normaliser& n)
{*this=n;}

Normaliser::~Normaliser()
{}

Normaliser& Normaliser::operator=(const Normaliser& n)
{
	if(this!=&n){
		theStdDev=n.theStdDev;
		theMean=n.theMean;
		theSkip=n.theSkip;
	}
	return *this;
}

DataSet& Normaliser::normalise(DataSet& d, bool doSkip)
{
	theMean = vector<double>(d.nInput()+d.nOutput(), 0);
	theStdDev = vector<double>(d.nInput()+d.nOutput(), 0);
	calcMean(d);
	calcStdDev(d);

	if(doSkip){
		theSkip = vector<bool>(d.nInput()+d.nOutput(), true);
		findSkip(d);
		for(uint i=0; i<theSkip.size(); ++i)
			if(theSkip[i]){
				theMean[i]=0;
				theStdDev[i]=1;
			}
	}

	for(uint i=0; i<d.size(); ++i) normalise(d.pattern(i));
	return d;
}

/** Subtracts the mean from each variable and then divides it with the corresponding standard deviation. 
 */
struct SubtractAndDivide 
{
	vector<double>::const_iterator itm;
	vector<double>::const_iterator its;
	SubtractAndDivide(vector<double>::const_iterator m, vector<double>::const_iterator s):itm(m), its(s) {}
	void operator()(double &x) { x = (x - *itm++) / *its++; }
};

/** Multiplies each variable with its standard deviation and then adds its
 * mean. 
 */
struct MultiplyAndAdd 
{
	vector<double>::const_iterator itm;
	vector<double>::const_iterator its;
	MultiplyAndAdd(vector<double>::const_iterator m, vector<double>::const_iterator s):itm(m), its(s) {}
	void operator()(double &x) { x = x * *its++ + *itm++;}
};

Pattern& Normaliser::normalise(Pattern& p)
{
	for_each( p.input().begin(), p.input().end(), SubtractAndDivide( theMean.begin(), theStdDev.begin() ) );
	for_each( p.output().begin(), p.output().end(), SubtractAndDivide( theMean.begin()+p.nInput(), theStdDev.begin()+p.nInput() ) );
	return p;
}

DataSet& Normaliser::unnormalise(DataSet& d)
{
	assert(theMean.size()==theStdDev.size());
	assert(theMean.size()==(d.nInput()+d.nOutput()));

	for(uint i=0; i<d.size(); ++i) unnormalise(d.pattern(i));
	return d;
}

Pattern& Normaliser::unnormalise(Pattern& p)
{
	for_each( p.input().begin(), p.input().end(), MultiplyAndAdd( theMean.begin(), theStdDev.begin() ) );
	for_each( p.output().begin(), p.output().end(), MultiplyAndAdd( theMean.begin() + p.nInput(), theStdDev.begin() + p.nInput() ) );
	return p;
}

vector<double>& Normaliser::stdDev(){return theStdDev;}

void Normaliser::stdDev(vector<double>& s){theStdDev=s;}

vector<double>& Normaliser::mean(){return theMean;}

void Normaliser::mean(vector<double>& m){theMean=m;}

//PRIVATE---------------------------------------------------------------------//

void Normaliser::calcMean(DataSet& d)
{
	for(uint i=0; i<d.size(); ++i){
		Pattern& p=d.pattern(i);
		vector<double>& in=p.input();
		vector<double>& out=p.output();
		uint i=0;
		for(uint j=0; j<in.size(); ++j, ++i)
			theMean[i] += in[j];
		for(uint j=0; j<out.size(); ++j, ++i)
			theMean[i] += out[j];
	}
	double n = d.size();
	for(uint i=0; i<theMean.size(); ++i)
		theMean[i] = theMean[i] / n;
}

void Normaliser::calcStdDev(DataSet& d)
{
	for(uint i=0; i<d.size(); ++i){
		Pattern& p=d.pattern(i);
		vector<double>& in=p.input();
		vector<double>& out=p.output();
		uint i=0;
		for(uint j=0; j<in.size(); ++j, ++i)
			theStdDev[i] += pow(in[j]-theMean[i],2);
		for(uint j=0; j<out.size(); ++j, ++i)
			theStdDev[i] += pow(out[j]-theMean[i],2);
	}
	double n = d.size();
	for(uint i=0; i<theStdDev.size(); ++i)
		theStdDev[i] = sqrt(theStdDev[i] / n);
}

void Normaliser::findSkip(DataSet& d)
{
	for(uint i=0; i<d.size(); ++i){
		Pattern& p=d.pattern(i);
		vector<double>& in=p.input();
		vector<double>& out=p.output();
		uint i=0;
		for(uint j=0; j<in.size(); ++j, ++i)
			theSkip[i] = (theSkip[i] && skipBin(in[j])) || 
				(theSkip[i] && skipSig(in[j]));
		for(uint j=0; j<out.size(); ++j, ++i)
			theSkip[i] = (theSkip[i] && skipBin(out[j])) || 
				(theSkip[i] && skipSig(out[j]));
	}
}

bool Normaliser::skipBin(double val) const
{
	double e=0.000001;
	return ( (fabs(val-1) <= e) || (fabs(val) <= e) );
}

bool Normaliser::skipSig(double val) const
{
	double e=0.000001;
	return ( (fabs(val-1) <= e) || (fabs(val+1) <= e) );
}

