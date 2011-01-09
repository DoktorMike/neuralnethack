#ifndef __Error_hh__
#define __Error_hh__

#include "Mlp.hh"
#include "datatools/DataSet.hh"

namespace NetHack
{
    using DataTools::DataSet;

    /**The base class for the error functions.
     * \sa SummedSquare, CrossEntropy.
     */
    class Error
    {
	public:		
	    ///Calculate the gradient for each neuron in the MLP.
	    ///\deprecated Use the block version instead since it more general
	    ///than the online version.
	    ///\param mlp the MLP.
	    ///\param in the input pattern.
	    ///\param out the output from the MLP.
	    ///\param dout the desired output.
	    ///\return the gradient.
	    virtual vector<double>& gradient(Mlp& mlp, vector<double>& in, 
		    vector<double>& out, vector<double>& dout)=0;

	    /**Calculate the gradient vector, and return the error.
	     * This returns the error calculated from a block of patterns.
	     * \param mlp the MLP to calculate the gradient for.
	     * \param dset the data set to use.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double gradient(Mlp& mlp, DataSet& dset, uint bs)=0;

	    /**Calculate the summed square error for this Mlp and DataSet.
	     * \param mlp the output from the MLP.
	     * \param dset the desired output.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double outputError(Mlp& mlp, DataSet& dset, uint bs)=0;

	    ///Return the type of error function.
	    string type();

	    ///Basic destructor.
	    virtual ~Error();

	protected:	
	    ///Basic constructor.
	    ///\param t the type of error.
	    Error(string t);

	    /**Calculate the summed square error for this output.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     * \return the error.
	     */
	    virtual double outputError(vector<double>& out, 
		    vector<double>& dout)=0;
	    
	    ///Type of error.
	    string theType;

	    ///The gradient vector.
	    vector<double> theGradient;

	private:
	    ///Copy constructor.
	    Error(const Error&);

	    ///Assignment operator.
	    Error& operator=(const Error&);

    };
}
#endif
