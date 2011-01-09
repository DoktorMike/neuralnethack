/*$Id: QuasiNewton.hh 1626 2007-05-08 12:08:19Z michael $*/

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


#ifndef __QuasiNewton_hh__
#define __QuasiNewton_hh__

#include "Trainer.hh"

#include <ostream>

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Trainer interface.
	 * This learning algorithm is called Quasi Newton and
	 * uses second order gradients of the error function.
	 * The weight update rule:
	 * \f[\omega_{t+1}=\omega_t + \alpha_t G_t g_t\f]
	 * Where \f$ G \f$ is an approximation of the inverse Hessian 
	 * and \f$ g \f$ is the gradient.
	 */
	class QuasiNewton: public Trainer
	{

		public:
			/**Basic constructor.
			 * \param mlp the Mlp to train.
			 * \param data the DataSet to use.
			 * \param error the Error function to use.
			 * \param te the training error at which to stop training.
			 * \param bs the batch size.
			 */
			QuasiNewton(Mlp& mlp, DataTools::DataSet& data, Error& error, double te, uint bs);

			/**Basic destructor. */
			~QuasiNewton();

			/**Method used to train an MLP. This uses the Mlp and the DataSet
			 * in the Trainer.
			 */
			void train(std::ostream& os);

			Trainer* clone() const;

		private:
			/**Copy constructor.
			 * \param qn the object to copy from.
			 */
			QuasiNewton(const QuasiNewton& qn);

			/**Assignment operator.
			 * \param qn the object to assign from.
			 */
			QuasiNewton& operator=(const QuasiNewton& qn);

			/**Reset all the vectors in the QuasiNewton algorithm. */
			void resetVectors();

			/**Updates the estimation of the inverse hessian using the DFP
			 * rule.
			 * \f[G_{t+1}=G_t+\frac{\Delta\omega\Delta\omega^T}{\Delta\omega^T\Delta
			 * g} - \frac{G_t\Delta g\Delta g^TG_t}{\Delta g^TG_t\Delta g}\f]
			 */
			void updateDfp();
			
			/**Updated the estimation of the inverse hessian using the BFGS
			 * rule. This rule is essentially DFP but adds another term in the
			 * end which gives better performance and is not much more
			 * expensive.
			 */
			void updateBfgs();

			/**Finds out the step length we want to use with our quasi newton
			 * direction. This calls mnbrak and brent.
			 * \param alpha the step length we want to use.
			 * \return the error function evaluated at alpha.
			 */
			float findAlpha(float& alpha);

			/**Brackets the minima. When algorithm finishes the wanted value
			 * lies between ax and cx i.e. around bx.
			 * \param ax the leftmost value for the x values.
			 * \param bx the middle value for the x values.
			 * \param cx the rightmost value for the x values.
			 * \param fa the value of the function at ax.
			 * \param fb the value of the function at bx.
			 * \param fc the value of the function at cx.
			 */
			void mnbrak(float *ax, float *bx, float *cx, 
					float *fa, float *fb, float *fc);

			/**An implementation of brents line search. When algorithms
			 * finishes the xmin will hold our desired step length.
			 * \param ax the leftmost value for the x values.
			 * \param bx the middle value for the x values.
			 * \param cx the rightmost value for the x values.
			 * \param tol the tolerance.
			 * \param xmin the x value corresponding to the minimum value of
			 * the error function.
			 * \return the error function evaluated at xmin.
			 */
			float brent(float ax, float bx, float cx, float tol,
					float *xmin);

			/**Calculate the error associated with a certain alfa.
			 * This updated the weights in the Mlp and then calculates the
			 * batch error for the DataSet using the Error function.
			 * \param alfa the quasi newton step length.
			 * \return the error.
			 */
			float err(float alfa);

			/**Checks if the algorithm has converged by inspecting the
			 * gradient. 
			 * \deprecated The Trainer interface implements a hasConverged
			 * which has a better measure.
			 * \return true if converged, false otherwise.
			 */
			bool converged();

			/**The estimation of the inverse Hessian matrix. */
			std::vector< std::vector<double> > G;

			/**The weight vector at t+1. */
			std::vector<double> w;

			/**The weight vector at t. */
			std::vector<double> wPrev;

			/**The gradient vector at t+1. */
			std::vector<double> g;

			/**The gradient vector at t. */
			std::vector<double> gPrev;

			/**The change in the weight vector. */
			std::vector<double> dw;

			/**The change in the gradient vector. */
			std::vector<double> dg;

			/**I will figure out something good to say here. :-)*/
			std::vector<double> u;

			/**Temporary matrix variable. */
			std::vector< std::vector<double> > matrixTemp1;

			/**Temporary matrix variable. */
			std::vector< std::vector<double> > matrixTemp2;

			/**Temporary vector variable. */
			std::vector<double> vectorTemp1;

			/**Temporary vector variable. */
			std::vector<double> vectorTemp2;
	};
}
#endif
