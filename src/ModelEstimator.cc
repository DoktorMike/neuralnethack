#include "ModelEstimator.hh"
#include "evaltools/Roc.hh"
#include "ErrorMeasures.hh"
#include "PrintUtils.hh"

#include <sstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using namespace std;

//PUBLIC

ModelEstimator::ModelEstimator():theEnsembleBuilder(0), theData(0), theEstimations(0){}

ModelEstimator::ModelEstimator(const ModelEstimator& me) { *this = me; }

ModelEstimator::~ModelEstimator()
{
	for(vector<Estimation>::iterator it = theEstimations.begin(); 
			it!=theEstimations.end(); ++it)
	{
		delete it->committee;
		delete it->trnData;
		delete it->valData;
	}
}

ModelEstimator& ModelEstimator::operator=(const ModelEstimator& me)
{
	if(this != &me){
		theEnsembleBuilder = me.theEnsembleBuilder;
		theData = me.theData;
		theEstimations = me.theEstimations;
	}
	return *this;
}

//ACCESSOR AND MUTATOR

EnsembleBuilder* ModelEstimator::ensembleBuilder(){return theEnsembleBuilder;}
void ModelEstimator::ensembleBuilder(EnsembleBuilder* eb){theEnsembleBuilder = eb;}

DataSet* ModelEstimator::data(){return theData;}
void ModelEstimator::data(DataSet* d){theData = d;}

//VARIOUS

void ModelEstimator::printOutputTargetList(ostream& os)
{
	PrintUtils::printTargetList(os, ">>target trn", *theData);

	uint run = 0;
	for(vector<Estimation>::iterator it = theEstimations.begin(); 
			it != theEstimations.end(); ++it, ++run)
	{
		Committee* committee = it->committee;
		DataSet& trn = *(it->trnData);
		DataSet& val = *(it->valData);

		std::ostringstream s1;
		s1<<">>trn\t"<<run+1;
		PrintUtils::printOutputList(os, s1.str(), *committee, trn);
		std::ostringstream s2;
		s2<<">>val\t"<<run+1;
		PrintUtils::printOutputList(os, s2.str(), *committee, val);
	}
}

//PROTECTED

pair<double, double>* ModelEstimator::calcMeanTrnValAuc()
{
	double trnAuc = 0;
	double valAuc = 0;

	for(vector<Estimation>::iterator it = theEstimations.begin(); 
			it != theEstimations.end(); ++it)
	{
		double tmp = ErrorMeasures::auc(*(it->committee), *(it->trnData));
		//cout<<"----------trnAUC 2: "<<tmp<<endl; //DEBUG
		trnAuc += tmp;
		tmp = ErrorMeasures::auc(*(it->committee), *(it->valData));
		//cout<<"----------valAUC 2: "<<tmp<<endl; //DEBUG
		valAuc += tmp;
	}
	trnAuc /= (double)theEstimations.size();
	valAuc /= (double)theEstimations.size();
	return new pair<double, double>(trnAuc, valAuc);
}

//PRIVATE

