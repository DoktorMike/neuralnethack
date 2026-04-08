#ifndef __NeuralNetHack_hh__
#define __NeuralNetHack_hh__

#include <config.h>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <iostream>

/**This namespace encloses the NeuralNetHack project.
 * It contains all classes and methods needed to create, train, and test a
 * committee of MLPs.
 */
namespace NeuralNetHack
{

	//DEBUGGING------------------------------------------------------------------//

	template<class T> void printVector(std::vector<T>& vec)
	{
		typename std::vector<T>::iterator it;
		for(it = vec.begin(); it != vec.end(); ++it)
			std::cout<<*it<<" ";
		std::cout<<std::endl;
	}

	template<class T> std::vector<T> split(std::string& s)
	{
		std::istringstream is(s);
		std::vector<T> vec; 
		vec.reserve(6);
		std::copy(std::istream_iterator<T>(is), std::istream_iterator<T>(), std::back_inserter(vec));
		return vec;
	}

	template<class T> T getfromstream(std::istream& is)
	{
		T tmp;
		is>>tmp;
		return tmp;
	}

}

#endif
