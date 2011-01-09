#ifndef __Trainer_hh__
#define __Trainer_hh__

#include "datatools/DataSet.hh"
#include "Committee.hh"


namespace NetHack
{
    using DataTools::DataSet;
    using DataTools::Pattern;

    class Error;

    ///Abstract class responsible for training an MLP on one pattern.
    class Trainer
    {
	public:
	    ///Basic destructor.
	    virtual ~Trainer();

	    /**Return the error function type this Trainer is using.
	     * \return the error function type.
	     */
	    string error();

	    ///Set the error function to train this committee with.
	    ///This method deletes eventual previous error function
	    ///and creates a new one as specified by the string e.
	    ///\param e the error function.
	    void error(string e);

	    ///Return the training error for this trainer.
	    double trainingError();

	    ///Set the training error for this trainer.
	    ///\param te the training error.
	    void trainingError(double te);

	    ///Method used to train a committee of MLPs.
	    ///\param committee the committee of MLPs to train.
	    ///\param dset the data set to train the committee on.
	    ///\param epochs the maximum number of epochs.
	    virtual void train(Committee& committee, DataSet& dset, 
		    uint epochs)=0;

	protected:
	    ///Basic constructor.
	    ///\param e the type of error function to create.
	    ///\param te the maximum training error allowed.
	    Trainer(string e, double te);

	    ///Basic constructor.
	    Trainer();

	    ///Method used to train an MLP.
	    ///\param mlp the MLP to train.
	    ///\param dset the data set to train the MLP on.
	    ///\param epochs the maximum number of epochs.
	    virtual void train(Mlp& mlp, DataSet& dset, uint epochs)=0;

	    ///The error function.
	    Error* theError;

	    ///The error required to stop training.
	    double theTrainingError;

	    ///The vector representing the weight update.
	    ///\deprecated This is no longer used in GradientDescent or
	    ///QuasiNewton learning.
	    vector<double> theWeightUpdate;

	private:

	    ///Copy constructor.
	    ///\param trainer the Trainer object to copy.
	    Trainer(const Trainer& trainer);

	    ///Assignment operator.
	    ///\param trainer the Trainer object to copy.
	    Trainer& operator=(const Trainer& trainer);

    };
}
#endif
