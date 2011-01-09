#ifndef __GradientDescent_hh__
#define __GradientDescent_hh__

#include "Trainer.hh"

namespace NetHack
{
    /**An implementation of the Trainer interface.
     * This learning algorithm is called Gradient Descent and
     * uses only first order gradients of the error function.
     * The weight update rule:
     * \f[\omega_{t+1}=\omega_t + \eta\cdot\frac{\partial E}
     * {\partial\omega_t} + \alpha\cdot\Delta\omega_t\f]
     */
    class GradientDescent: public Trainer
    {

	public:
	    /**Basic constructor.
	     * \param e the error function to train the committee with.
	     * \param te the training error at which to stop training.
	     * \param bs the batch size.
	     * \param lr the learning rate.
	     * \param dlr the decrease of learning rate.
	     * \param m the momentum term.
	     */
	    GradientDescent(string e, double te, uint bs, 
		    double lr, double dlr, double m);

	    /**Basic destructor. */
	    ~GradientDescent();

	    /**Method used to train a committee of MLPs.
	     * \param committee the committee of MLPs to train.
	     * \param dset the data set to train the committee on.
	     * \param epochs the maximum number of epochs.
	     */
	    void train(Committee& committee, DataSet& dset, uint epochs);

	    /**Set the batch size for this trainer.
	     * \param bs the batch size.
	     */
	    void batchSize(uint bs);

	    /**Set the learning rate for this trainer.
	     * \param lr the learning rate.
	     */
	    void learningRate(double lr);

	    /**Set the decrease of learning rate for this trainer.
	     * \param dlr the learning rate.
	     */
	    void decLearningRate(double dlr);

	    /**Set the momentum for this trainer.
	     * \param m the momentum term.
	     */
	    void momentum(double m);

	    /**Return the batch size for this trainer. */
	    uint batchSize();

	    /**Return the learning rate this trainer is using. */
	    double learningRate();

	    /**Return the decrease of learning rate this trainer is using. */
	    double decLearningRate();

	    /**Return the momentum term this trainer is using. */
	    double momentum();

	private:
	    /**Copy constructor.
	     * \param gd the object to copy from.
	     */
	    GradientDescent(const GradientDescent& gd);

	    /**Assignment operator.
	     * \param gd the object to assign from.
	     */
	    GradientDescent& operator=(const GradientDescent& gd);

	    /**Method used to train an MLP.
	     * \param mlp the MLP to train.
	     * \param dset the data set to train the MLP on.
	     * \param epochs the maximum number of epochs.
	     */
	    void train(Mlp& mlp, DataSet& dset, uint epochs);

	    /**Method used to train an MLP during one epoch.
	     * \param mlp the MLP to train.
	     * \param dset the data set to train the MLP on.
	     */
	    double train(Mlp& mlp, DataSet& dset);

	    /**Updates the learning rate.
	     * \f[ \eta_{t+1} = \left\{\begin{array}{ll}
	     * \eta_t * \gamma, & E_{t+1}>E_t;\\ 
	     * \eta_t * \left(1+\frac{1-\gamma}{10}\right), & otherwise;
	     * \end{array}\right. \f]
	     */
	    void updateLearningRate(double err, double prevErr);

	    /**The number of patterns to use every epoch. */
	    uint theBatchSize;
	    
	    /**The learning rate. */
	    double theLearningRate;

	    /**The decrease of the learning rate.
	     * This governs how much the learning rate should be decreased
	     * when training error goes up. The value is typically close to
	     * but always less than 1.0.
	     */
	    double theDecLearningRate;

	    /**The momentum term.
	     * This governs how much of the previous weight update should be
	     * included in the current update. This value is typically set
	     * between 0.8 and 0.99. Never more than 1.0.
	     */
	    double theMomentum;

    };
}
#endif
