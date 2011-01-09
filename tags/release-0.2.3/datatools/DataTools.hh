#ifndef __DataTools_hh__
#define __DataTools_hh__

#include <vector>

#include <iostream>
#include <ios>
#include <fstream>
#include <cassert>

/**This namespace encloses a bunch of functions for managing data sets. For the
 * time being this only supports abstraction of a data set into a list of
 * Pattern(s).
 */
namespace DataTools{

	using std::vector;

	//IO stuff
	using std::cout;
	using std::endl;
	using std::cerr;
	using std::ostream;
	using std::ios;
}

#endif
