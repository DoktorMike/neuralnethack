#include "Bootstrapper.hh"

#include <iostream>
#include <cassert>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace std;

//PUBLIC
Bootstrapper::Bootstrapper():ModelEstimator(), n(0){}

Bootstrapper::Bootstrapper(const Bootstrapper& bs) { *this = bs; }

Bootstrapper::~Bootstrapper()
{
}

Bootstrapper& Bootstrapper::operator=(const Bootstrapper& bs)
{
	if(this != &bs){
		ModelEstimator::operator=(bs);
		n = bs.n;
	}
	return *this;
}

pair<double, double>* Bootstrapper::estimateModel()
{
	assert(theEnsembleBuilder != 0 && n != 0);

	cout<<"Estimating model using Bootstrapper with N="<<n<<endl;
	DataManager manager;
	for(uint i=0; i<n; ++i){
		cout<<"Run (N): "<<i+1<<"\n";
		pair<DataSet, DataSet> dataSets = manager.split(*theData);
		DataSet& trnData = dataSets.first;
		DataSet& valData = dataSets.second;
		theEnsembleBuilder->data(&trnData);
		Committee* committee = theEnsembleBuilder->buildEnsemble();
		Estimation e = {committee, new DataSet(trnData), new DataSet(valData)};
		theEstimations.push_back(e);
	}
	return calcMeanTrnValAuc();
}

uint Bootstrapper::numRuns(){return n;}

void Bootstrapper::numRuns(uint n){this->n = n;}

//PRIVATE

