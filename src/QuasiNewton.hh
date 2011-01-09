#ifndef __QuasiNewton_hh__
#define __QuasiNewton_hh__

#include "Trainer.hh"

namespace NeuralNetHack
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
			 * \param te the training error at which to stop training.
			 * \param bs the batch size.
			 * \param we toggle weight elimination.
			 * \param alpha the importance of the weight elimination term.
			 * \param w0 the scaling of the weight elimination term.
			 */
			QuasiNewton(double te, uint bs, bool we, double alpha, double w0);

			/**Basic destructor. */
			~QuasiNewton();

			/**Method used to train an MLP.
			 * \param mlp the MLP to train.
			 * \param dset the data set to train the MLP on.
			 */
			void train(Mlp& mlp, DataSet& dset);

		private:
			/**Copy constructor.
			 * \param qn the object to copy from.
			 */
			QuasiNewton(const QuasiNewton& qn);

			/**Assignment operator.
			 * \param qn the object to assign from.
			 */
			QuasiNewton& operator=(const QuasiNewton& qn);

			void resetVectors(Mlp& mlp, DataSet& dset);

			/**Updates the estimation of the inverse hessian using the DFP
			 * rule.
			 * \f[G_{t+1}=G_t+\frac{\Delta\omega\Delta\omega^T}{\Delta\omega^T\Delta
			 * g} - \frac{G_t\Delta g\Delta g^TG_t}{\Delta g^TG_t\Delta g}\f]
			 * \param mlp the Mlp used to evaluate the gradient.
			 * \param dset the DataSet used to evaluate the gradient.
			 */
			void updateDfp(Mlp& mlp, DataSet& dset);
			
			void updateBfgs(Mlp& mlp, DataSet& dset);

			float findAlpha(Mlp& mlp, DataSet& dset, float& alpha);

			void mnbrak(float *ax, float *bx, float *cx, 
					float *fa, float *fb, float *fc, Mlp& mlp, DataSet& dset);

			float brent(float ax, float bx, float cx, float tol,
					float *xmin, Mlp& mlp, DataSet& dset);

			float err(Mlp& mlp, DataSet& dset, float alfa);

			bool converged();

			/**The estimation of the inverse Hessian matrix. */
			vector< vector<double> > G;

			/**The weight vector at t+1. */
			vector<double> w;

			/**The weight vector at t. */
			vector<double> wPrev;

			/**The gradient vector at t+1. */
			vector<double> g;

			/**The gradient vector at t. */
			vector<double> gPrev;

			/**The change in the weight vector. */
			vector<double> dw;

			/**The change in the gradient vector. */
			vector<double> dg;

			/**I will figure out something good to say here. :-)*/
			vector<double> u;

			/**Temporary matrix variable. */
			vector< vector<double> > matrixTemp1;

			/**Temporary matrix variable. */
			vector< vector<double> > matrixTemp2;

			/**Temporary vector variable. */
			vector<double> vectorTemp1;

			/**Temporary vector variable. */
			vector<double> vectorTemp2;
	};
}
#endif
