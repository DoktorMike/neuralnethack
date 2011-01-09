#include "CrossSplitter.hh"
#include "ErrorMeasures.hh"

#include <vector>
#include <cassert>
#include <fstream>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

//PUBLIC
CrossSplitter::CrossSplitter():EnsembleBuilder(), n(0), k(0){}

CrossSplitter::CrossSplitter(const CrossSplitter& cs):EnsembleBuilder(cs)
{*this = cs;}

CrossSplitter::~CrossSplitter(){}

CrossSplitter& CrossSplitter::operator=(const CrossSplitter& cs)
{
	if(this != &cs){
		this->EnsembleBuilder::operator=(cs);
		n = cs.n;
		k = cs.k;
	}
	return *this;
}

Committee* CrossSplitter::buildEnsemble()
{
	assert(isValid() && n && k && theTrainer->isValid());

	Mlp* mlp = theTrainer->mlp(); //Get the original mlp.
	Committee* committee = new Committee();

	cout<<"Building ensemble using CrossSplitting with N="<<n<<" K="<<k<<endl;
	for(uint i=0; i<n; ++i){
		cout<<"Run (N): "<<i+1<<endl;
		vector<DataSet> dataSets = theDataManager->split(*theData, k);

		for(uint j = 0; j<dataSets.size(); ++j){
			cout<<"Part (K): "<<j+1<<endl;
			DataSet& trnData = dataSets[j];
			Mlp* newMlp = new Mlp(mlp->arch(), mlp->types(), mlp->softmax());
			theTrainer->batchSize(trnData.size());
			theTrainer->train(*newMlp, trnData);
			committee->addMlp(*newMlp); //This copies the mlp.
			delete newMlp;
		}
	}
	theTrainer->mlp(mlp); //Put the original mlp back.

	return committee;
}

uint CrossSplitter::numRuns(){return n;}

void CrossSplitter::numRuns(uint n){this->n = n;}

uint CrossSplitter::numParts(){return k;}

void CrossSplitter::numParts(uint k){this->k = k;}

//PRIVATE

