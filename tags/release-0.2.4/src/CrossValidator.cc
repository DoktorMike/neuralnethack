#include "CrossValidator.hh"
#include "ErrorMeasures.hh"
#include "PrintUtils.hh"

#include <ostream>
#include <cassert>
#include <sstream>
#include <fstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace std;

//PUBLIC
CrossValidator::CrossValidator():ModelEstimator(), n(0){}

CrossValidator::CrossValidator(const CrossValidator& cv) { *this = cv; }

CrossValidator::~CrossValidator()
{
}

CrossValidator& CrossValidator::operator=(const CrossValidator& cv)
{
	if(this != &cv){
		ModelEstimator::operator=(cv);
		n = cv.n;
	}
	return *this;
}

pair<double, double>* CrossValidator::estimateModel()
{
	assert(theEnsembleBuilder != 0 && n != 0 && k > 1);

	cout<<"Estimating model using CrossValidation with N="<<n<<" K="<<k<<endl;
	DataManager manager;
	for(uint i=0; i<n; ++i){
		cout<<"Run (N): "<<i+1<<endl;
		vector<DataSet> dataSets = manager.split(*theData, k);  

		for(uint j = 0; j<k; ++j){
			cout<<"Part (K): "<<j+1<<endl;
			DataSet valData = dataSets.front();
			dataSets.erase(dataSets.begin());
			DataSet trnData = manager.join(dataSets);

			theEnsembleBuilder->data(&trnData);
			Committee* committee = theEnsembleBuilder->buildEnsemble();
			Estimation e = {committee, new DataSet(trnData), new DataSet(valData)};
			theEstimations.push_back(e);
			dataSets.push_back(valData);
		}
	}
	return calcMeanTrnValAuc(); //This should be replaced with a more generic call.
}

void CrossValidator::printOutputTargetList(ostream& os)
{
	PrintUtils::printTargetList(os, ">>target trn", *theData);

	vector<Estimation>::iterator it = theEstimations.begin();
	for(uint i=0; i<n; ++i){
		for(uint j=0; j<k; ++j, ++it){
			Committee* committee = it->committee;
			DataSet& trn = *(it->trnData);
			DataSet& val = *(it->valData);

			std::ostringstream s1;
			s1<<">>trn\t"<<i+1<<"\t"<<j+1;
			PrintUtils::printOutputList(os, s1.str(), *committee, trn);
			std::ostringstream s2;
			s2<<">>val\t"<<i+1<<"\t"<<j+1;
			PrintUtils::printOutputList(os, s2.str(), *committee, val);
		}
	}
}

uint CrossValidator::numRuns(){return n;}

void CrossValidator::numRuns(uint n){this->n = n;}

uint CrossValidator::numParts(){return k;}

void CrossValidator::numParts(uint k){this->k = k;}

//PRIVATE

