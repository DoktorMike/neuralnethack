#ifndef __Layer_hh__
#define __Layer_hh__

#include <string>

#include "Neuron.hh"

namespace NetHack
{
    ///Class representing a layer in an MLP.
    ///It has two iterators representing the weight interval for the current
    ///layer. Also it knows the number of neurons contained in itself and its
    ///predecessor.
    class Layer
    {
	public:
	    ///Constructor taking the weight interval for this layer.
	    Layer(int ncurr, int nprev, string type);

	    ///Copy constructur.
	    Layer(const Layer&);

	    ///Destructor.
	    virtual ~Layer();

	    ///Assignment operator.
	    Layer& operator=(const Layer&);

	    ///Index operator.
	    Neuron& operator[](const uint);

	    ///Return the neurons enclosed in this layer.
	    vector<Neuron*>& neurons();
	    
	    ///Fetch the indexed neuron in this layer.
	    Neuron& neuron(uint index);

	    ///Return the number of neurons enclosed in this layer.
	    uint nNeurons();

	    ///Alias for nNeurons.
	    uint size();

	    ///Returns the output from this layer. Not including the bias!
	    vector<double>& output();
	    
	    ///Returns the output from this layer. Including the bias.
	    vector<double>& bOutput();

	    ///Propagates an input pattern through this layer. Note that
	    ///the bias should not be included in the parameter since it is
	    ///explicitly included later.
	    vector<double>& propagate(vector<double>&);

	    ///Return the weights leading into this layer.
	    vector<double> weights();

	    ///Return the gradients leading into this layer.
	    vector<double> gradients();

	    ///Set the weights to those beginning at f.
	    ///\param f the iterator pointing to the first element.
	    ///\return the end position of f.
	    vector<double>::iterator weights(vector<double>::iterator f);

	    ///Update the weights by those beginning at f.
	    ///\param f the iterator pointing to the first element.
	    ///\return the end position of f.
	    vector<double>::iterator updateWeights(vector<double>::iterator f);

	    ///Set the weights to those in w.
	    ///\param w the weights to assign.
	    void weights(vector<double>& w);

	    ///Update the weights by those in w.
	    ///\param w the weights to update by.
	    void updateWeights(vector<double>& w);

	    ///Return the number of weights enclosed in this layer.
	    uint nWeights();

	    ///Prints the weights leading to this layer
	    void printWeights();

	    ///Prints the local gradients for the neurons in this layer.
	    void printLocalGradients();

	    ///Prints the gradients for the neurons in this layer.
	    void printGradients();

	private:

	    ///Number of neurons in this layer.
	    int ncurr;

	    ///Number of neurons in previous layer.
	    int nprev;

	    ///Type of neurons in this layer.
	    string theType;

	    ///The neurons of this layer.
	    vector<Neuron*> theNeurons;

	    ///The output of this layer.
	    vector<double> theOutput;

	    ///The biased output of this layer.
	    vector<double> theBOutput;

	    ///Creates the neurons in this layer.
	    void createNeurons();
    };
}
#endif
