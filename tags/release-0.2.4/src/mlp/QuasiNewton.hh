#ifndef __QuasiNewton_hh__
#define __QuasiNewton_hh__

#include "Trainer.hh"

namespace MultiLayerPerceptron
{
	/**A class representing the implementation of the Trainer interface.
	 * This learning algorithm is called Quasi Newton and
	 * uses second order gradients of the error function.
	 * The weight update rule:
	 * \f[\omega_{t+1}=\omega_t + \alpha_t G_t g_t\f]
	 * Where \f$ G \f$ is an approximation of the inverse Hessian 
	 * and \f$ g \f$ is the gradient.
	 * \todo Fix so that QuasiNewton uses the internat Mlp and DataSet
	 * pointers instead of passing it around as arguments.
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
			void train();

		private:
			/**Copy constructor.
			 * \param qn the object to copy from.
			 */
			QuasiNewton(const QuasiNewton& qn);

			/**Assignment operator.
			 * \param qn the object to assign from.
			 */
			QuasiNewton& operator=(const QuasiNewton& qn);

			void resetVectors();

			/**Updates the estimation of the inverse hessian using the DFP
			 * rule.
			 * \f[G_{t+1}=G_t+\frac{\Delta\omega\Delta\omega^T}{\Delta\omega^T\Delta
			 * g} - \frac{G_t\Delta g\Delta g^TG_t}{\Delta g^TG_t\Delta g}\f]
			 */
			void updateDfp();
			
			void updateBfgs();

			float findAlpha(float& alpha);

			void mnbrak(float *ax, float *bx, float *cx, 
					float *fa, float *fb, float *fc);

			float brent(float ax, float bx, float cx, float tol,
					float *xmin);

			float err(float alfa);

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
