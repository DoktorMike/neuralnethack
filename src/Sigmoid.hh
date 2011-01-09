#ifndef __Sigmoid_hh__
#define __Sigmoid_hh__

#include "Neuron.hh"

namespace NetHack
{
    /**An implementation of the Neuron interface. 
     * This class implements the sigmoid activation function.
     * \f[\varphi(v)=\frac{1}{1+\exp^{-v}}\f]
     */
    class Sigmoid: public Neuron
    {
	public:
	    /**Basic constructor.
	     * \param nprev the number of neurons in the previous layer.
	     */
	    Sigmoid(uint nprev);
	    
	    /**A copy constructor. 
	     * \param n the object to copy from.
	     */
	    Sigmoid(const Sigmoid& n);
	    
	    /**The destructor. */
	    virtual ~Sigmoid();
	    
	    /**Assignment operator. 
	     * \param n the object to assign from.
	     */
	    Sigmoid& operator=(const Sigmoid& n);
	    
	    /**The activation function.
	     * \param input the vector holding the input to the neuron.
	     */
	    double fire(vector<double>& input);
	    
	    /**A derivative of the activation function.
	     * \param input the vector holding the input to the neuron.
	     */
	    double firePrime(vector<double>& input);
	    
	    /**A derivative of the activation function. */
	    double firePrime();
    };
}
#endif
