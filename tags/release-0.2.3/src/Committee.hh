#ifndef __Committee_hh__
#define __Committee_hh__

#include "mlp/Mlp.hh"

namespace NeuralNetHack{

	using namespace MultiLayerPerceptron;

	/**A class representing an ensemble of Mlp(s). Thus the classification of a
	 * pattern is the weighted sum of the classification from each of the Mlp
	 * located in this ensemble.
	 */
	class Committee
	{
		public:
			/**Basic constructor. */
			Committee();
			
			/**Basic constructor. A Committee must have at least one MLP.
			 * \param mlp the first mlp in this ensemble.
			 * \param s the scaling of the first MLP.
			 */
			Committee(Mlp& mlp, double s);

			/**Copy constructor.
			 * \param c the object to copy from.
			 */
			Committee(const Committee& c);

			/**Basic destructor. */
			~Committee();

			/**Assignment operator.
			 * \param c the object to assign from.
			 */
			Committee& operator=(const Committee& c);

			/**Index operator.
			 * \param i the index of the MLP that is to be returned.
			 * \return the MLP located at index i.
			 */
			Mlp& operator[](const uint i);

			/**Fetch one of the MLPs in the committee.
			 * \param i the index of the MLP that is to be returned.
			 * \return the MLP located at index i.
			 */
			Mlp& mlp(const uint i);

			/**Delete and fetch one of the MLPs in the committee.
			 * \param i the index of the MLP that is to be deleted.
			 */
			void delMlp(const uint i);

			/**Add an MLP to this ensemble.
			 * \param mlp the MLP to add.
			 * \param s the scale for this MLP.
			 */
			void addMlp(Mlp& mlp, double s);

			/**Add an MLP to this ensemble. The scale for each MLP in the
			 * Committe is set to 1/N. Thus all previous scales are destroyed.
			 * \param mlp the MLP to add.
			 */
			void addMlp(Mlp& mlp);

			/**Return the scale for the MLP located at index i.
			 * \param i the index of the MLP which scale is to be returned.
			 */
			double scale(const uint i);

			/**Set the scale for the MLP located at index i.
			 * \param i the index of the MLP which scale is to be returned.
			 * \param s the scale to set.
			 */
			void scale(const uint i, double s);

			/**Return the number of MLPs residing in this ensemble. */
			uint size();

			/**Propagate the input through this ensemble of MLPs.
			 * The input is propagated through each of the MLPs. Their
			 * respective output is then combined linearly with their
			 * respective scaling to produce the output for this Committee.
			 * \param input the input to propagate.
			 */
			vector<double> propagate(vector<double>& input);

		private:
			/**A committee of MLPs. */
			vector<Mlp*> theCommittee;

			/**The scales(weights) to use when combining the output from each
			 * MLP. By default the standard mean value is computed.
			 * \f[y_c=\sum_i\alpha_i y_i\f]
			 */
			vector<double> theScales;
	};
}
#endif
