#ifndef __Ensemble_hh__
#define __Ensemble_hh__

#include "mlp/Mlp.hh"

#include <memory>
#include <vector>

namespace NeuralNetHack {

/**A class representing an ensemble of Mlp(s). Thus the classification of a
 * pattern is the weighted sum of the classification from each of the Mlp
 * located in this ensemble.
 */
class Ensemble {
  public:
	/**Basic constructor. */
	Ensemble();

	/**Basic constructor. A Ensemble must have at least one MLP.
	 * \param mlp the first mlp in this ensemble.
	 * \param s the scaling of the first MLP.
	 */
	Ensemble(MultiLayerPerceptron::Mlp& mlp, double s);

	/**Copy constructor.
	 * \param c the object to copy from.
	 */
	Ensemble(const Ensemble& c);

	/**Move constructor. */
	Ensemble(Ensemble&&) noexcept = default;

	/**Basic destructor. */
	~Ensemble();

	/**Assignment operator.
	 * \param c the object to assign from.
	 */
	Ensemble& operator=(const Ensemble& c);

	/**Move assignment operator. */
	Ensemble& operator=(Ensemble&&) noexcept = default;

	/**Index operator.
	 * \param i the index of the MLP that is to be returned.
	 * \return the MLP located at index i.
	 */
	MultiLayerPerceptron::Mlp& operator[](const uint i);

	/**Fetch one of the MLPs in the committee.
	 * \param i the index of the MLP that is to be returned.
	 * \return the MLP located at index i.
	 */
	MultiLayerPerceptron::Mlp& mlp(const uint i);

	/**Fetch one of the MLPs in the committee (const).
	 * \param i the index of the MLP that is to be returned.
	 * \return the MLP located at index i.
	 */
	const MultiLayerPerceptron::Mlp& mlp(const uint i) const;

	/**Delete and fetch one of the MLPs in the committee.
	 * \param i the index of the MLP that is to be deleted.
	 */
	void delMlp(const uint i);

	/**Add an MLP to this ensemble. This will copy the mlp.
	 * \param mlp the MLP to add.
	 * \param s the scale for this MLP.
	 */
	void addMlp(MultiLayerPerceptron::Mlp& mlp, double s);

	/**Add an MLP to this ensemble. This will copy the mlp.
	 * The scale for each MLP in the Committe is set to 1/N.
	 * Thus all previous scales are destroyed.
	 * \param mlp the MLP to add.
	 */
	void addMlp(MultiLayerPerceptron::Mlp& mlp);

	/**Add an MLP to this ensemble by taking ownership.
	 * \param mlp the MLP to add (ownership transferred).
	 * \param s the scale for this MLP.
	 */
	void addMlp(std::unique_ptr<MultiLayerPerceptron::Mlp> mlp, double s);

	/**Add an MLP to this ensemble by taking ownership.
	 * The scale for each MLP in the Committe is set to 1/N.
	 * Thus all previous scales are destroyed.
	 * \param mlp the MLP to add (ownership transferred).
	 */
	void addMlp(std::unique_ptr<MultiLayerPerceptron::Mlp> mlp);

	/**Return the scale for the MLP located at index i.
	 * \param i the index of the MLP which scale is to be returned.
	 */
	double scale(const uint i) const;

	/**Set the scale for the MLP located at index i.
	 * \param i the index of the MLP which scale is to be returned.
	 * \param s the scale to set.
	 */
	void scale(const uint i, double s);

	/**Return the number of MLPs residing in this ensemble. */
	uint size() const;

	/**Propagate the input through this ensemble of MLPs.
	 * The input is propagated through each of the MLPs. Their
	 * respective output is then combined linearly with their
	 * respective scaling to produce the output for this Ensemble.
	 * \param input the input to propagate.
	 */
	std::vector<double> propagate(const std::vector<double>& input);

  private:
	/**A committee of MLPs. */
	std::vector<std::unique_ptr<MultiLayerPerceptron::Mlp>> theEnsemble;

	/**The scales(weights) to use when combining the output from each
	 * MLP. By default the standard mean value is computed.
	 * \f[y_c=\sum_i\alpha_i y_i\f]
	 */
	std::vector<double> theScales;
};
} // namespace NeuralNetHack
#endif
