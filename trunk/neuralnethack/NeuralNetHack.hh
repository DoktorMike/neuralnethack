/*$Id: NeuralNetHack.hh 1622 2007-05-08 08:29:10Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#ifndef __NeuralNetHack_hh__
#define __NeuralNetHack_hh__

#include <config.h>
#include <vector>
#include <string>
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
