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

	for(uint i=0; i<d.size(); ++i){
		Pattern& p=d.pattern(i);
		vector<double>& in=p.input();
		vector<double>& out=p.output();
		uint i=0;
		for(uint j=0; j<in.size(); ++j, ++i)
			in[j]=(in[j]-theMean[i])/theStdDev[i];
		for(uint j=0; j<out.size(); ++j, ++i)
			out[j]=(out[j]-theMean[i])/theStdDev[i];
	}

	return d;
}

DataSet& Normaliser::unnormalise(DataSet& d)
{
	assert(theMean.size()==theStdDev.size());
	assert(theMean.size()==(d.nInput()+d.nOutput()));

	for(uint i=0; i<d.size(); ++i){
		Pattern& p=d.pattern(i);
		vector<double>& in=p.input();
		vector<double>& out=p.output();
		uint i=0;
		for(uint j=0; j<in.size(); ++j, ++i)
			in[j]=in[j]*theStdDev[i]+theMean[i];
		for(uint j=0; j<out.size(); ++j, ++i)
			out[j]=out[j]*theStdDev[i]+theMean[i];
	}

	return d;
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

bool Normaliser::skipBin(double val)
{
	double e=0.000001;
	return ( (fabs(val-1) <= e) || (fabs(val) <= e) );
}

bool Normaliser::skipSig(double val)
{
	double e=0.000001;
	return ( (fabs(val-1) <= e) || (fabs(val+1) <= e) );
}

