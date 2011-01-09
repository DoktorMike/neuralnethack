#include "NeuralNetHack.hh"

#include <iostream>

using namespace std;

//DEBUGGING-------------------------------------------------------------------//

template<class T> void NeuralNetHack::printVector(std::vector<T>& vec)
{
	typename std::vector<T>::iterator it;
	for(it = vec.begin(); it != vec.end(); ++it)
		std::cout<<*it<<" ";
	std::cout<<std::endl;
}


