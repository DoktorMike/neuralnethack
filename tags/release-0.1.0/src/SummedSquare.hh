#ifndef __SummedSquare_hh__
#define __SummedSquare_hh__

#include "Error.hh"

namespace NetHack
{
    /**An implementation of the Error interface.
     * This class represents the Summed Square Error function.
     * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
     */
    class SummedSquare:public Error
    {
	public:
	    /**Basic constructor. */
	    SummedSquare();

	    /**Basic destructor. */
	    ~SummedSquare();

	    /**Calculate the gradient for each neuron in the MLP.
	     * \param mlp the MLP.
	     * \param in the input pattern.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     */
	    vector<double>& gradient(Mlp& mlp, vector<double>& in, 
		    vector<double>& out, vector<double>& dout);

	    /**Return the gradient vector. This returns the gradient vector
	     * calculated from a block of patterns.
	     * \param mlp the MLP to calculate the gradient for.
	     * \param dset the data set to use.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the gradient.
	     */
	    double gradient(Mlp& mlp, DataSet& dset, uint bs);

	    /**Calculate the summed square error for this Mlp and DataSet.
	     * \param mlp the output from the MLP.
	     * \param dset the desired output.
	     * \param bs the batch size i.e. how many patterns to use.
	     * \return the error.
	     */
	    double outputError(Mlp& mlp, DataSet& dset, uint bs);

	private:
	    /**Copy contructor.
	     * \param sse the error to create this object from.
	     */
	    SummedSquare(const SummedSquare& sse);
	    
	    /**Assignment operator.
	     * \param sse the error to copy from.
	     */
	    SummedSquare& operator=(const SummedSquare& sse);
	    
	    /**Calculate the output error for every neuron in the output
	     * layer.
	     * \param ol the output layer.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     */
	    void localGradient(Layer& ol, vector<double>& out, 
		    vector<double>& dout);
	    
	    /**Calculate the local gradient.
	     * \param curr the current layer.
	     * \param next the next layer.
	     */
	    void localGradient(Layer& curr, Layer& next);

	    /**Back-propagate the output error through the network.
	     * \param mlp the MLP to backpropagate the error through.
	     */
	    void backpropagate(Mlp& mlp);

	    /**Calculate gradient for the weights in the first layer.
	     * \param first the first hidden layer with weights.
	     * \param in the input vector.
	     */
	    void gradient(Layer& first, vector<double>& in);

	    /**Calculate gradient for the weights in the first layer. This
	     * does not replace the previous gradients. Instead it adds to new
	     * gradient to the previous one. This is intended for use with the
	     * block version of gradient.
	     * \param first the first hidden layer with weights.
	     * \param in the input vector.
	     */
	    void gradientBatch(Layer& first, vector<double>& in);

	    /**Calculate the gradient between two layers.
	     * \param curr the current layer.
	     * \param prev the previous layer.
	     */
	    void gradient(Layer& curr, Layer& prev); 

	    /**Calculate the gradient between two layers. This
	     * does not replace the previous gradients. Instead it adds to new
	     * gradient to the previous one. This is intended for use with the
	     * block version of gradient.
	     * \param curr the current layer.
	     * \param prev the previous layer.
	     */
	    void gradientBatch(Layer& curr, Layer& prev); 

	    /**Compose one big gradient vector from the MLP.
	     * \param mlp the MLP to compose the vector from.
	     * \deprecated Use the gradient function in the MLP instead.
	     */
	    vector<double>& gradient(Mlp& mlp);
	    
	    /**Calculate the summed square error for this output.
	     * \param out the output from the MLP.
	     * \param dout the desired output.
	     * \return the error.
	     */
	    double outputError(vector<double>& out, vector<double>& dout);
    };
}
#endif
