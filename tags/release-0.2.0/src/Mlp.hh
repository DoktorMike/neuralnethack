#ifndef __Mlp_hh__
#define __Mlp_hh__

#include "Layer.hh"

namespace NetHack
{
    ///A Class representing a multilayer perceptron. 
    class Mlp
    {
	public:
	    ///Base constructor.
	    ///\param a the architecture of this MLP.
	    ///\param t the types of neurons that are to be used.
	    ///\param s the softmax output switch.
	    Mlp(vector<uint>& a, vector<string>& t, bool s);

	    ///Copy constructor.
	    Mlp(const Mlp&);

	    ///The default destructor.
	    ~Mlp();

	    ///Assignment operator.
	    Mlp& operator=(const Mlp&);

	    ///Index operator.
	    Layer& operator[](const uint);

	    ///Pushes a pattern through this MLP.
	    vector<double> propagate(vector<double>&);

	    ///Print all the weights in the network.
	    void printWeights();

	    ///Returns the entire weight vector.
	    vector<double> weights();

	    ///Returns the entire gradient vector.
	    vector<double> gradients();

	    ///Add these weights to the weightvector.
	    ///\param dw the weights to add to the weightvector.
	    void updateWeights(vector<double>& dw);

	    ///Set the weightvector to the weights in w.
	    ///\param w the weights to set the weightvector to.
	    void weights(vector<double>& w);

	    ///Return the number of weights contained in this MLP.
	    uint nWeights();

	    ///Return the number of layers contained in this MLP.
	    uint nLayers();

	    ///Alias for nLayers.
	    uint size();

	    ///Return the architechture of this MLP.
	    vector<uint>& arch();

	    ///Returns the indexed layer.
	    Layer& layer(uint index);

	    ///Return the layervector.
	    vector<Layer*>& layers();

	private:
	    ///Create all the layers in this MLP.
	    void createLayers();

	    ///The architecture for this MLP.
	    vector<uint> theArch;

	    ///The type of neurons for each layer.
	    vector<string> theTypes;

	    ///Softmax switch.
	    bool softmax;

	    ///The layers in this MLP.
	    vector<Layer*> theLayers;
    };
}
#endif
