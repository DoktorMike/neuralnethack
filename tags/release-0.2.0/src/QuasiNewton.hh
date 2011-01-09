#ifndef __QuasiNewton_hh__
#define __QuasiNewton_hh__

#include "Trainer.hh"

namespace NetHack
{
    /**An implementation of the Trainer interface.
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
	     * \param e the error function to train the committee with.
	     * \param te the training error at which to stop training.
	     * \param bs the batch size.
	     */
	    QuasiNewton(string e, double te, uint bs);

	    /**Basic destructor. */
	    ~QuasiNewton();

	    /**Method used to train a committee of MLPs.
	     * \param committee the committee of MLPs to train.
	     * \param dset the data set to train the committee on.
	     * \param epochs the maximum number of epochs.
	     */
	    void train(Committee& committee, DataSet& dset, uint epochs);

	    /**Return the batch size for this trainer. */
	    uint batchSize();

	    /**Set the batch size for this trainer.
	     * \param bs the batch size.
	     */
	    void batchSize(uint bs);

	private:
	    /**Copy constructor.
	     * \param qn the object to copy from.
	     */
	    QuasiNewton(const QuasiNewton& qn);

	    /**Assignment operator.
	     * \param qn the object to assign from.
	     */
	    QuasiNewton& operator=(const QuasiNewton& qn);

	    /**Method used to train an MLP.
	     * \param mlp the MLP to train.
	     * \param dset the data set to train the MLP on.
	     * \param epochs the maximum number of epochs.
	     */
	    void train(Mlp& mlp, DataSet& dset, uint epochs);

	    void buildInvHessEstim(Mlp& mlp, DataSet& dset);

	    void buildG(Mlp& mlp, DataSet& dset);

	    void buildP();

	    void buildV();

	    void buildU();

	    float findAlpha(Mlp& mlp, DataSet& dset, float& alpha);
    
	    void mnbrak(float *ax, float *bx, float *cx, 
		    float *fa, float *fb, float *fc, Mlp& mlp, DataSet& dset);

	    float brent(float ax, float bx, float cx, float tol,
		    float *xmin, Mlp& mlp, DataSet& dset);

	    float err(Mlp& mlp, DataSet& dset, float alfa);
    
	    bool converged();

	    /**The number of patterns to use every epoch. */
	    uint theBatchSize;

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
	    vector<double> p;

	    /**The change in the gradient vector. */
	    vector<double> v;

	    /**I will figure out something good to say here. :-)*/
	    vector<double> u;
    };
}
#endif
