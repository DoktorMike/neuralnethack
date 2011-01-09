#ifndef __MultiLayerPerceptron_hh__
#define __MultiLayerPerceptron_hh__

#include <config.h>
#include <vector>
#include <stack>
#include <utility>
#include <cassert>
#include <iostream>
#include <fstream>

/**This namespace encloses the MultiLayerPerceptron library.
 * It contains all classes and methods needed to create an MLP.
 */
namespace MultiLayerPerceptron
{
	///A define for the Cross Entropy Error identification string.
	#define CEE "kullback"
	///A define for the Summed Square Error identification string.
	#define SSE "sumsqr"
	///A define for the Quasi Newton minimisation identification string.
	#define QN "qn"
	///A define for the Gradient Descent minimisation identification string.
	#define GD "gd"
	///A define for the Linear activation function identification string.
	#define LINEAR "purelin"
	///A define for the Sigmoid activation function identification string.
	#define SIGMOID "logsig"
	///A define for the TanHyp activation function identification string.
	#define TANHYP "tansig"
	///A define for the maximum training error.
	#define MAX_ERROR 0.00001

	using std::vector;
	using std::pair;
	using std::string;
	using std::cout;
	using std::cerr;
	using std::endl;
	using std::ifstream;
	using std::ofstream;
	using std::ostream;
	using std::ios;

	//DEBUGGING------------------------------------------------------------------//

	/**Print a weight vector.
	 * \param arch the network architecture.
	 * \param theWeights the weight vector.
	 */
	void printWeights(vector<uint>& arch, vector<double>& theWeights);

}

#endif
