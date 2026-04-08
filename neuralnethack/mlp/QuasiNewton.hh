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
	/**A class implementing the L-BFGS quasi-Newton optimization algorithm.
	 * Uses limited-memory BFGS which stores only the last M (s,y) pairs
	 * instead of the full n*n inverse Hessian approximation.
	 * Memory: O(M*n) instead of O(n^2). Compute: O(M*n) instead of O(n^2).
	 *
	 * The weight update rule:
	 * \f[\omega_{t+1}=\omega_t + \alpha_t H_t g_t\f]
	 * Where \f$ H_t \f$ is the L-BFGS approximation of the inverse Hessian
	 * computed via the two-loop recursion, and \f$ g \f$ is the gradient.
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

			std::unique_ptr<Trainer> clone() const override;

		private:
			/**Copy constructor.
			 * \param qn the object to copy from.
			 */
			QuasiNewton(const QuasiNewton& qn);

			/**Assignment operator.
			 * \param qn the object to assign from.
			 */
			QuasiNewton& operator=(const QuasiNewton& qn);

			/**Reset all the vectors in the L-BFGS algorithm. */
			void resetVectors();

			/**Compute the L-BFGS search direction via the two-loop recursion.
			 * Result is placed in dir (= H_k * grad).
			 * \param grad the current gradient vector.
			 * \param dir the resulting search direction.
			 */
			void lbfgsDirection(const std::vector<double>& grad, std::vector<double>& dir);

			/**Store a new (s,y) pair in the circular history buffer.
			 * s = w - wPrev, y = g - gPrev. Skips if curvature condition not met.
			 */
			void storeHistory();

			/**Finds out the step length we want to use with our quasi newton
			 * direction. This calls mnbrak and brent.
			 * \param alpha the step length we want to use.
			 * \return the error function evaluated at alpha.
			 */
			float findAlpha(float& alpha);

			/**Brackets the minima.
			 * \param ax the leftmost value for the x values.
			 * \param bx the middle value for the x values.
			 * \param cx the rightmost value for the x values.
			 * \param fa the value of the function at ax.
			 * \param fb the value of the function at bx.
			 * \param fc the value of the function at cx.
			 */
			void mnbrak(float *ax, float *bx, float *cx,
					float *fa, float *fb, float *fc);

			/**An implementation of brents line search.
			 * \param ax the leftmost value for the x values.
			 * \param bx the middle value for the x values.
			 * \param cx the rightmost value for the x values.
			 * \param tol the tolerance.
			 * \param xmin the x value corresponding to the minimum.
			 * \return the error function evaluated at xmin.
			 */
			float brent(float ax, float bx, float cx, float tol,
					float *xmin);

			/**Calculate the error associated with a certain alfa.
			 * \param alfa the quasi newton step length.
			 * \return the error.
			 */
			float err(float alfa);

			/**L-BFGS history size (number of (s,y) pairs to store). */
			static constexpr uint LBFGS_M = 20;

			/**Number of weights. */
			uint nWeights;

			/**The weight vector at t+1. */
			std::vector<double> w;

			/**The weight vector at t. */
			std::vector<double> wPrev;

			/**The gradient vector at t+1. */
			std::vector<double> g;

			/**The gradient vector at t. */
			std::vector<double> gPrev;

			/**The change in the weight vector (s_k). */
			std::vector<double> dw;

			/**The change in the gradient vector (y_k). */
			std::vector<double> dg;

			/**Temporary vector for search direction (H*g). */
			std::vector<double> vectorTemp1;

			/**Temporary vector for line search evaluation. */
			std::vector<double> vectorTemp2;

			// L-BFGS history (circular buffer)

			/**History of weight changes s_k = w_{k+1} - w_k. */
			std::vector<std::vector<double>> sHistory;

			/**History of gradient changes y_k = g_{k+1} - g_k. */
			std::vector<std::vector<double>> yHistory;

			/**History of rho_k = 1 / (y_k^T * s_k). */
			std::vector<double> rhoHistory;

			/**Number of (s,y) pairs currently stored. */
			uint historyCount;

			/**Start index in the circular buffer. */
			uint historyStart;
	};
}
#endif

