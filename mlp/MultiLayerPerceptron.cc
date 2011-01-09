#include "MultiLayerPerceptron.hh"

//DEBUGGING-------------------------------------------------------------------//

void MultiLayerPerceptron::printWeights(vector<uint>& arch, vector<double>& theWeights)
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
