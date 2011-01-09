#ifndef __Config_hh__
#define __Config_hh__

#include "NetHack.hh"
#include <string>

namespace NetHack
{
    ///This class holds all the needed configuration.
    ///Every outside configuration is to be put in this file
    ///via the Parser.
    class Config
    {
	public:
	    ///Basic constructor.
	    Config();
	    ///Basic destructor.
	    virtual ~Config();
	    
	    ///Returns the name of the data file.
	    string filename();
	    ///Return the input columns indices.
	    vector<uint>& inputColumns();
	    ///Return the output columns indices.
	    vector<uint>& outputColumns();
	    ///Return the number of layers to use.
	    uint nLayers();
	    ///Return the number of neurons in each layer.
	    vector<uint>& arch();
	    ///Return the name of th activation functions to use.
	    vector<string>& activationFunctions();
	    ///Return the name of the error function to use.
	    string errorFunction();
	    ///Return the name of the learning algorithm to use.
	    string learningAlgorithm();
	    ///Return the maximum number of epochs to train.
	    uint nEpochs();
	    ///Return the number of samples to use for each epoch.
	    uint batchSize();
	    ///Return the learning rate parameter.
	    double learningRate();
	    ///Return the decrease of learning rate parameter.
	    double decLearningRate();
	    ///Return the momentum term.
	    double momentum();
	    
	    ///Set the filename.
	    ///\param fname the filename.
	    void filename(string fname);
	    ///Set the input column indices vector.
	    ///\param in the input column indices vector.
	    void inputColumns(vector<uint> in);
	    ///Set the output column indices vector.
	    ///\param out the output column indices vector.
	    void outputColumns(vector<uint> out);
	    ///Set the number of layers.
	    ///\param n the number of layers.
	    void nLayers(uint n);
	    ///Set the architecture.
	    ///\param a the architecture to use.
	    void arch(vector<uint> a);
	    ///Set the activation functions to use.
	    ///\param af the activation functions to use.
	    void activationFunctions(vector<string> af);
	    ///Set the error function to use during training.
	    ///\param ef the error function.
	    void errorFunction(string ef);
	    ///Set the learning algorithm to use.
	    ///\param la the learning algorithm.
	    void learningAlgorithm(string la);
	    ///Set the maximum number of epochs to train.
	    ///\param me the maximum number of epochs.
	    void nEpochs(uint me);
	    ///Set the number of samples to use per epoch.
	    ///\param bs the number of samples to use.
	    void batchSize(uint bs);
	    ///Set the learning rate to use during training.
	    ///\param lr the learning rate to use.
	    void learningRate(double lr);
	    ///Set the amount of decrease of learning rate.
	    ///\param dlr the decreasion of the learning rate.
	    void decLearningRate(double dlr);
	    ///Set the momentum term.
	    ///\param m the momentum term.
	    void momentum(double m);

	    ///Print out every bit of information contained in this class.
	    void print();

	private:
	    ///Copy constructor.
	    ///\param c the Config object to copy.
	    Config(const Config& c);
	    ///Assignment operator.
	    ///\param c the Config object to assign from.
	    Config& operator=(const Config& c);

	    ///The file where the data is located.
	    string theFileName;
	    ///The input columns.
	    vector<uint> theInCols;
	    ///The output columns.
	    vector<uint> theOutCols;
	    ///The number of layers.
	    uint theNLayers;
	    ///The architecture.
	    vector<uint> theArch;
	    ///The activation functions.
	    vector<string> theActFcn;
	    ///The error function.
	    string theErrFcn;
	    ///The minimisation algorithm.
	    string theMinMethod;
	    ///The maximum number of epochs to train.
	    uint theMaxEpochs;
	    ///The number of trainingsamples to use per epoch.
	    uint theBatchSize;
	    ///The rate at wich to train the MLP.
	    double theLearningRate;
	    ///The decrease of learning rate.
	    double theDecLearningRate;
	    ///The momentum term used in "poor mans conjugate gradient".
	    double theMomentum;
    };
}
#endif
