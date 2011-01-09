/*$Id: Ensemble.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __Ensemble_hh__
#define __Ensemble_hh__

#include "mlp/Mlp.hh"

#include <vector>

namespace NeuralNetHack{

	/**A class representing an ensemble of Mlp(s). Thus the classification of a
	 * pattern is the weighted sum of the classification from each of the Mlp
	 * located in this ensemble.
	 */
	class Ensemble
	{
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

			/**Basic destructor. */
			~Ensemble();

			/**Assignment operator.
			 * \param c the object to assign from.
			 */
			Ensemble& operator=(const Ensemble& c);

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

			/**Add an MLP to this ensemble. This will use the mlp without
			 * copying it. Thus the mlp pointed to will be destroyed when this
			 * object is destroyed.
			 * \param mlp the MLP to add.
			 * \param s the scale for this MLP.
			 */
			void addMlp(MultiLayerPerceptron::Mlp* mlp, double s);

			/**Add an MLP to this ensemble. This will use the mlp without
			 * copying it. Thus the mlp pointed to will be destroyed when this
			 * object is destroyed.
			 * The scale for each MLP in the Committe is set to 1/N. 
			 * Thus all previous scales are destroyed.
			 * \param mlp the MLP to add.
			 */
			void addMlp(MultiLayerPerceptron::Mlp* mlp);

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
			std::vector<double> propagate(std::vector<double>& input);

		private:
			/**A committee of MLPs. */
			std::vector<MultiLayerPerceptron::Mlp*> theEnsemble;

			/**The scales(weights) to use when combining the output from each
			 * MLP. By default the standard mean value is computed.
			 * \f[y_c=\sum_i\alpha_i y_i\f]
			 */
			std::vector<double> theScales;
	};
}
#endif
