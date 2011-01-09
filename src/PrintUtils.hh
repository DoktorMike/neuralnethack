#ifndef __PrintUtils_hh__
#define __PrintUtils_hh__

#include "datatools/DataSet.hh"
#include "Committee.hh"

#include <ostream>
#include <string>

namespace NeuralNetHack
{
	namespace PrintUtils
	{
		void printTargetList(std::ostream& os, std::string id, DataTools::DataSet& data);
		void printOutputList(std::ostream& os, std::string id, Committee& c, DataTools::DataSet& data);
	}
}

#endif

