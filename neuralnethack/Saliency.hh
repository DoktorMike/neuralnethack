#ifndef __Saliency_hh__
#define __Saliency_hh__

#include "mlp/Mlp.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "Ensemble.hh"

#include <vector>
#include <ostream>

namespace NeuralNetHack
{
	/**This namespace encloses functions for calculating salinecies for a
	 * given datapoint for a given Mlp or Ensemble.
	 */
	namespace Saliency
	{

		/**Propagates an input through an Mlp up to the last layer. This seems
		 * very similar to the propagate function in the mlp, but they differ
		 * in that this propagate does not activate the last layer, but thus
		 * only calculates the induced local field.
		 * \param mlp the mlp to propagate the input through.
		 * \param input the input to propagate.
		 * \return the local induced field for the output neuron.
		 */
		double propagate(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input);

		/**Calculate the magnitude of the saliency for each variable averaged over all
		 * datapoints and committee members. This means that we don't use the
		 * sign of the derivative in the calculation. We only want to measure
		 * importance for the output. This is used in feature selection.
		 * \param committee the committee to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliencyMagnitude(Ensemble& committee, DataTools::DataSet& data, bool inner);

		/**Calculate the saliency for each variable averaged over all
		 * datapoints and committee members.
		 * \param committee the committee to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(Ensemble& committee, DataTools::DataSet& data, bool inner);

		/**Calculate the saliency for each variable in the pattern averaged over all
		 * committee members.
		 * \param committee the committee to use as a predictor
		 * \param pattern the pattern containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(Ensemble& committee, DataTools::Pattern& pattern, bool inner);

		/**Calculate the magnitude of the saliency for each variable averaged over all
		 * datapoints. This means that we don't use the
		 * sign of the derivative in the calculation. We only want to measure
		 * importance for the output. This is used in feature selection.
		 * \param committee the committee to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliencyMagnitude(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& data, bool inner);

		/**Calculate the saliency for each variable averaged over all
		 * datapoints.
		 * \param mlp the mlp to use as a predictor
		 * \param data the data set containing all variables
		 * \return a vector of averaged variable saliences
		 */
		std::vector<double> saliency(MultiLayerPerceptron::Mlp& mlp, DataTools::DataSet& data, bool inner);

		/**Calculate the saliency for each variables in the pattern.
		 * \param mlp the mlp to use as a predictor
		 * \param pattern the pattern containing all variables
		 * \return a vector of variable saliences
		 */
		std::vector<double> saliency(MultiLayerPerceptron::Mlp& mlp, DataTools::Pattern& pattern, bool inner);

		/**Calculate the partial derivative of the output with respect to an
		 * input variable.
		 * \param mlp the mlp to use as a predictor
		 * \param input the input vector containing all variables
		 * \param index the index of the varible to use for the partial derivative
		 * \return the value of the partial derivative
		 */
		double derivative(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input, uint index);

		/**Calculate the partial derivative of the output with respect to an
		 * input variable. This function is restricted to two Layer Mlp:s and
		 * has been used for debugging purposes.
		 * \param mlp the mlp to use as a predictor
		 * \param input the input vector containing all variables
		 * \param index the index of the varible to use for the partial derivative
		 * \return the value of the partial derivative
		 */
		double derivative_debug(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input, uint index);

		/**Calculate the partial derivative of the output with respect to an
		 * input variable. Note that the last activation function is not used
		 * in this derivative. Thus what we use is the linear combination of
		 * the activation of each hidden node plus a bias.
		 * \param mlp the mlp to use as a predictor
		 * \param input the input vector containing all variables
		 * \param index the index of the varible to use for the partial derivative
		 * \return the value of the partial derivative
		 */
		double derivative_inner(MultiLayerPerceptron::Mlp& mlp, std::vector<double>& input, uint index);

		/**Print the result of an Saliency calculation for all Inputs in the
		 * DataSet.
		 * \param os the output stream to write to.
		 * \param saliency the saliencies to write.
		 */
		void print(std::ostream& os, std::vector<double>& saliency);
	}
}

#endif
