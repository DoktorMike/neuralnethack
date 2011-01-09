#include "Error.hh"

using namespace NetHack;

Error::Error(string t):theType(t),theGradient(0){}

Error::~Error(){}

string Error::type(){return theType;}

//PRIVATE--------------------------------------------------------------------//

Error::Error(const Error& err)
{*this = err;}

Error& Error::operator=(const Error& err)
{
	if(this != &err)
	    theType = err.theType;
	return *this;
}

