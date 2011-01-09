#include "PrintUtils.hh"

using namespace NeuralNetHack;
using namespace DataTools;
using namespace std;

void PrintUtils::printTargetList(ostream& os, string id, DataSet& data)
{
	os<<id<<endl;

	vector<uint>& indices = data.indices();
	for(uint i=0; i<data.size(); ++i){
		os<<"\t"<<indices[i]<<"\t";
		vector<double>& target = data.pattern(i).output();
		for(uint j=0; j<target.size(); ++j)
			os<<std::scientific<<target[j]<<" ";
		os<<endl;
	}
}

void PrintUtils::printOutputList(ostream& os, string id, Committee& c, DataSet& data)
{
	os<<id<<endl;

	vector<uint>& indices = data.indices();
	for(uint i=0; i<data.size(); ++i){
		os<<"\t"<<indices[i]<<"\t";
		vector<double> output = c.propagate(data.pattern(i).input());
		for(uint j=0; j<output.size(); ++j)
			os<<std::scientific<<output[j]<<" ";
		os<<endl;
	}
}


