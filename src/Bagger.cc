#include "Bagger.hh"

#include <cassert>

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace std;

//PUBLIC
Bagger::Bagger():EnsembleBuilder(), n(0){}

Bagger::Bagger(const Bagger& bg):EnsembleBuilder(bg)
{*this = bg;}

Bagger::~Bagger(){}

Bagger& Bagger::operator=(const Bagger& bg)
{
	if(this != &bg){
		this->EnsembleBuilder::operator=(bg);
		n = bg.n;
	}
	return *this;
}

Committee* Bagger::buildEnsemble()
{
	assert(isValid() && n && theTrainer->isValid());

	Mlp* mlp = theTrainer->mlp(); //Get the original mlp.
	Committee* committee = new Committee();

	cout<<"Building ensemble using Bagging with N="<<n<<endl;
	for(uint i=0; i<n; ++i){
		cout<<"Run (N): "<<i+1<<"\n";
		pair<DataSet, DataSet> dataSets = theDataManager->split(*theData);
		DataSet& trnData = dataSets.first;
		Mlp* newMlp = new Mlp(mlp->arch(), mlp->types(), mlp->softmax());
		theTrainer->batchSize(trnData.size());
		theTrainer->train(*newMlp, trnData);
		committee->addMlp(*newMlp); //This copies the mlp.
		delete newMlp;
	}
	theTrainer->mlp(mlp); //Put the original mlp back.

	return committee;
}

uint Bagger::numRuns(){return n;}

void Bagger::numRuns(uint n){this->n = n;}

//PRIVATE

