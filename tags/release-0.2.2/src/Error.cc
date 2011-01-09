#include "Error.hh"

using namespace NeuralNetHack;

Error::Error(string t):theMlp(0),theDset(0),
					   theType(t),theGradient(0){}

Error::Error(Mlp* mlp, DataSet* dset, string t):theMlp(mlp),theDset(dset),
						theType(t),theGradient(0){}

Error::~Error(){}

Mlp* Error::mlp(){return theMlp;}

void Error::mlp(Mlp* mlp){theMlp = mlp;}

DataSet* Error::dset(){return theDset;}

void Error::dset(DataSet* dset){theDset = dset;}

string Error::type(){return theType;}

//PRIVATE--------------------------------------------------------------------//

Error::Error(const Error& err)
{*this = err;}

Error& Error::operator=(const Error& err)
{
	if(this != &err){
		theMlp = err.theMlp;
		theDset = err.theDset;
		theType = err.theType;
		theGradient = err.theGradient;
	}
	return *this;
}

