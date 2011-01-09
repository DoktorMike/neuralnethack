#ifndef __Error_hh__
#define __Error_hh__

#include "mlp/Mlp.hh"
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
	    /**Calculate the gradient for each neuron in the MLP.
	     * \deprecated Use the block version instead since it is more 
	     * general than the online version.
	     * \param mlp the MLP.
	     * \param in the input pattern.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     * \return the gradient.
	     */
	    virtual vector<double>& gradient(Mlp& mlp, vector<double>& in, 
		    vector<double>& out, vector<double>& dout)=0;

	    /**Calculate the gradient vector, and return the error.
	     * This returns the error calculated from a block of patterns.
	     * \deprecated Use gradient(uint bs) instead.
	     * \param mlp the MLP to calculate the gradient for.
	     * \param dset the data set to use.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double gradient(Mlp& mlp, DataSet& dset, uint bs)=0;

	    /**Calculate the gradient vector, and return the error.
	     * This returns the error calculated from a block of patterns.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double gradient(uint bs)=0;

	    /**Calculate the summed square error for this Mlp and DataSet.
	     * \deprecated Use outputError(uint bs) instead.
	     * \param mlp the output from the MLP.
	     * \param dset the desired output.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double outputError(Mlp& mlp, DataSet& dset, uint bs)=0;

	    /**Calculate the summed square error for this Mlp and DataSet.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    virtual double outputError(uint bs)=0;

	    /**Accessor method for the Mlp.
	     * \return the Mlp belonging to the Error.
	     */
	    Mlp* mlp();
	    
	    /**Mutator method for the Mlp.
	     * \param mlp the Mlp to assign to this Error.
	     */
	    void mlp(Mlp* mlp);

	    /**Accessor method for the DataSet.
	     * \return the DataSet belonging to the Error.
	     */
	    DataSet* dset();
	    
	    /**Mutator method for the DataSet.
	     * \param dset the DataSet to assign to this Error.
	     */
	    void dset(DataSet* dset);

	    /**Return the type of error function. 
	     * \return the type of error function used.
	     */
	    string type();

	    /**Basic destructor. */
	    virtual ~Error();

	protected:	
	    /**Basic constructor.
	     * \deprecated use the constructor taking an Mlp and a DataSet
	     * instead.
	     * \param t the type of error.
	     */
	    Error(string t);

	    /**Basic constructor.
	     * \param mlp the mlp to use.
	     * \param dset the dataset to use.
	     * \param t the type of error.
	     */
	    Error(Mlp* mlp, DataSet* dset, string t);

	    /**Calculate the summed square error for this output.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     * \return the error.
	     */
	    virtual double outputError(vector<double>& out, 
		    vector<double>& dout)=0;
	    
	    /**The Mlp associated with an Error.*/
	    Mlp* theMlp;
	    
	    /**The DataSet associated with an Error.*/
	    DataSet* theDset;
	    
	    /**Type of error. */
	    string theType;

	    /**The gradient vector. */
	    vector<double> theGradient;

	private:
	    /**Copy constructor. */
	    Error(const Error&);

	    /**Assignment operator. */
	    Error& operator=(const Error&);

    };
}
#endif
