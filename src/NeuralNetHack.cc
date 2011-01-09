#include "NeuralNetHack.hh"

//DEBUGGING-------------------------------------------------------------------//

void NeuralNetHack::printWeights(vector<uint>& arch, vector<double>& theWeights)
{
	vector<uint>::iterator itaprev = arch.begin();
	vector<uint>::iterator itacurr = itaprev;
	vector<double>::iterator itw = theWeights.begin();
	uint i=0;

	while(++itacurr != arch.end()){
		cout<<endl<<"Layer "<<++i<<":"<<endl;
		for(uint j=0; j<((*itacurr)*((*itaprev)+1)); j++){
			cout.width(4);
			cout<<*itw<<" ";
			itw++;
		}
		itaprev++;
		cout<<endl;
	}
}

void NeuralNetHack::printVector(vector<uint>& vec)
{
	vector<uint>::iterator it;
	for(it = vec.begin(); it != vec.end(); ++it)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}

void NeuralNetHack::printVector(vector<double>& vec)
{
	vector<double>::iterator it;
	for(it = vec.begin(); it != vec.end(); ++it)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}


