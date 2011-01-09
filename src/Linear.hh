#ifndef __Linear_hh__
#define __Linear_hh__

#include "Neuron.hh"

namespace NetHack
{
    /**An implementation of the Neuron interface. 
     * This class implements the sigmoid activation function.
     * \f[\varphi(v)=v\f]
     */
    class Linear: public Neuron
    {
	public:
	    /**Basic constructor.
	     * \param nprev the number of neurons in the previous layer.
	     */
	    Linear(uint nprev);
	    
	    /**A copy constructor. 
	     * \param n the object to copy from.
	     */
	    Linear(const Linear& n);
	    
	    /**The destructor. */
	    virtual ~Linear();
	    
	    /**Assignment operator. 
	     * \param n the object to assign from.
	     */
	    Linear& operator=(const Linear& n);
	    
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
