#ifndef __Neruron_hh__
#define __Neruron_hh__

#include "NetHack.hh"

namespace NetHack
{
    /**An abstract class representing a Neuron. */
    class Neuron
    {
	public:
	    /**A virtual destructor. */
	    virtual ~Neuron();
	    
	    /**Index operator. Return the indeced weight leading into this
	     * neuron.
	     */
	    double& operator[](const uint);
	    
	    /**Calculates the local induced field. */
	    double potential(vector<double>&);
	    
	    /**Return the number of weights leading into this neuron. */
	    uint nWeights();

	    /**Print the weights leading into this neuron. */
	    void printWeights();
	    
	    /**Return the weights leading into this neuron. */
	    vector<double>& weights();
	    
	    /**Return the weight between this neuron and the specified in the
	     * previous layer.
	     */
	    double weights(uint prev);
	    
	    /**Set the weights to those in w.
	     * \param w the weights to be set.
	     */
	    void weights(vector<double>& w);
	    
	    /**Set the weights to those beginning at iterator f.
	     * f is increased until it has reached the end of the 
	     * weight vector.
	     * \param f the iterator pointing to the first element.
	     */
	    vector<double>::iterator weights(vector<double>::iterator f);
	    
	    /**Update the weights by those in w.
	     * \param w the weight updates.
	     */
	    void updateWeights(vector<double>& w);
	    
	    /**Update the weights by those beginning at iterator f.
	     * f is increased until it has reached the end of the 
	     * weight vector.
	     * \param f the iterator pointing to the first element.
	     */
	    vector<double>::iterator updateWeights(vector<double>::iterator f);

	    /**Return the neurons type.
	     */
	    string type();

	    /**Return the local induced field.
	     */
	    double lif();

	    /**Return the activation.
	     */
	    double activation();

	    /**Return the local gradient.
	     */
	    double localGradient();

	    /**Set the local gradient.
	     */
	    void localGradient(double);

	    /**Return the gradient for the weights between this neuron and the
	     * neurons in the previous layer.
	     */
	    vector<double>& gradients();

	    /**Return the gradient for the weight between this neuron and the
	     * specified in the previous layer.
	     */
	    double gradient(uint prev);

	    /**Set the gradients.
	     */
	    void gradients(const vector<double>&);

	    /**Set the gradient for weight at index.
	     * \param index the weight to set gradient for.
	     * \param g the gradient value.
	     */
	    void gradient(uint index, double g);

	    /**Print the gradients for the weights leading into this neuron. */
	    void printGradients();
	    
	    /**Return the previous update for the weight between this neuron and the
	     * specified in the previous layer.
	     */
	    double prevWeightUpd(uint prev);

	    /**Set the previous weight update.
	     */
	    void prevWeightUpd(const vector<double>&);

	    /**Set the previous update for weight at index.
	     * \param index the weight to set gradient for.
	     * \param g the gradient value.
	     */
	    void prevWeightUpd(uint index, double u);

	    /**Print the gradients for the weights leading into this neuron. */
	    void printPrevWeightUpd();
	    
	    /**A virtual activation function. */
	    virtual double fire(vector<double>&)=0;

	    /**A virtual derivative of the activation function. */
	    virtual double firePrime(vector<double>&)=0;

	    /**A virtual derivative of the activation function. */
	    virtual double firePrime()=0;

	protected:
	    /**Basic constructor. */
	    Neuron(uint nprev);
	    
	    /**Copy constructor. */
	    Neuron(const Neuron& n);
	    
	    /**The weights leading into this neuron. This includes the bias.*/
	    vector<double> theWeights;

	    /**Type of neuron. */
	    string theType;

	    /**The local induced field of the neuron. */
	    double theLif;

	    /**The activation level of the neuron. */
	    double theActivation;

	    /**The local gradient associated with this neuron. */
	    double theLocalGradient;

	    /**The gradient associated with this neuron. */
	    vector<double> theGradients;

	    /**The gradient associated with this neuron. */
	    vector<double> thePrevWeightUpd;

	private:

    };
}
#endif
