#ifndef __NeuralNetHack_hh__
#define __NeuralNetHack_hh__

#include <config.h>
#include <vector>

/**This namespace encloses the NeuralNetHack project.
 * It contains all classes and methods needed to create, train, and test a
 * committee of MLPs.
 */
namespace NeuralNetHack
{

	//DEBUGGING------------------------------------------------------------------//

	template<class T> void printVector(std::vector<T>& vec);

}

#endif
