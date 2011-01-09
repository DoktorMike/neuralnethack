#include "CrossValidator.hh"
#include "datatools/DataManager.hh"
#include "evaltools/Roc.hh"
#include "QuasiNewton.hh"
#include "GradientDescent.hh"

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using std::ostream;

CrossValidator::CrossValidator():numRuns(0), numParts(0), 
	valOutput(0), valTarget(0), trnOutput(0), trnTarget(0),
	aucTraining(0), aucValidation(0), theCommittee(0)
{}

CrossValidator::CrossValidator(const CrossValidator& cv)
{*this = cv;}

CrossValidator::~CrossValidator()
{
	if(theCommittee != 0) delete theCommittee;
}

CrossValidator& CrossValidator::operator=(const CrossValidator& cv)
{
	if(this != &cv){
		numRuns = cv.numRuns;
		numParts = cv.numParts;
		valOutput = cv.valOutput;
		valTarget = cv.valTarget;
		trnOutput = cv.trnOutput;
		trnTarget = cv.trnTarget;
		if(cv.theCommittee != 0) 
			theCommittee = new Committee(*(cv.theCommittee));
		else 
			theCommittee = 0;
	}
	return *this;
}

vector<double>& CrossValidator::validateOutput()
{return valOutput;}

void CrossValidator::validateOutput(vector<double>& vo)
{valOutput = vo;}

vector<uint>& CrossValidator::validateTarget()
{return valTarget;}

void CrossValidator::validateTarget(vector<uint>& vt)
{valTarget = vt;}

double CrossValidator::auc(bool validation)
{return (validation == true) ? aucValidation : aucTraining;}

Committee& CrossValidator::committee()
{return *theCommittee;}

void CrossValidator::committee(Committee& c)
{*theCommittee = c;}

void CrossValidator::crossValidate(Trainer& trainer, DataSet& ds, uint n, double ratio)
{
	assert(n>0);
	numRuns = n; 
	numParts = 1;

	valOutput = vector<double>(ds.size(), 0);
	valTarget = vector<uint>(ds.size(), 0);
	trnOutput = vector<double>(ds.size(), 0);
	trnTarget = vector<uint>(ds.size(), 0);
	theCommittee = new Committee();

	DataManager manager;
	trainer.validate();
	Mlp* mlp = trainer.mlp();

	for(uint i=0; i<n; ++i){
		vector<DataSet> dataSets = manager.split(ds, ratio);
		DataSet& trainingSet = dataSets[0];
		DataSet& validationSet = dataSets[1];
		mlp = new Mlp(mlp->arch(), mlp->types(), false);
		trainer.batchSize(trainingSet.size());
		trainer.train(*mlp, trainingSet);
		addToCommitteeOutput(*mlp, trainingSet, false);
		addToCommitteeOutput(*mlp, validationSet, true);
		theCommittee->addMlp(*mlp);
	}
	assert(valOutput.size() == valTarget.size());
	assert(trnOutput.size() == trnTarget.size());
	for(uint i=0; i<valOutput.size(); ++i)
		valOutput[i] = valOutput[i]/(double)numRuns;
	for(uint i=0; i<trnOutput.size(); ++i)
		trnOutput[i] = trnOutput[i]/(double)numRuns;
	Roc roc;
	aucValidation = roc.calcAucWmw(valOutput, valTarget);
	aucTraining = roc.calcAucWmw(trnOutput, trnTarget);
	//aucValidation = roc.calcAucTrapezoidal(valOutput, valTarget);
	//aucTraining = roc.calcAucTrapezoidal(trnOutput, trnTarget);
}

void CrossValidator::crossValidate(Trainer& trainer, DataSet& ds, uint n, uint k)
{
	assert(n > 0 && k > 1);	

	numRuns = n; 
	numParts = k;
	valOutput = vector<double>(ds.size(), 0);
	valTarget = vector<uint>(ds.size(), 0);
	trnOutput = vector<double>(ds.size(), 0);
	trnTarget = vector<uint>(ds.size(), 0);
//	comSizeVal = vector<uint>(ds.size(), 0);
//	comSizeTrn = vector<uint>(ds.size(), 0);
	theCommittee = new Committee();

	for(uint i=0; i<n; ++i){
		cout<<"Crossvalidation run (N): "<<i+1<<"\n";
		crossValidate(trainer, ds, k);
	}
	assert(valOutput.size() == valTarget.size());
	assert(trnOutput.size() == trnTarget.size());
	for(uint i=0; i<valOutput.size(); ++i){
		//cout<<"DEBUG: valoutput["<<i<<"] : "<<valOutput[i];
		valOutput[i] = valOutput[i]/(double)numRuns;
		//cout<<" rescaled to valoutput["<<i<<"] : "<<valOutput[i]<<endl;
	}
	for(uint i=0; i<trnOutput.size(); ++i)
		trnOutput[i] = trnOutput[i]/(double)((numParts-1)*numRuns);
	Roc roc;
	aucValidation = roc.calcAucWmw(valOutput, valTarget);
	aucTraining = roc.calcAucWmw(trnOutput, trnTarget);

//	for(uint i=0; i<comSizeVal.size(); ++i){
//		cout<<"DEBUG: comSizeVal["<<i<<"] : "<<comSizeVal[i];
//		cout<<" comSizeTrn["<<i<<"] : "<<comSizeTrn[i]<<endl;
//	}
}

//PRIVATE-----------------------------------------------------------------------

void CrossValidator::crossValidate(Trainer& trainer, DataSet& ds, uint k)
{
	DataManager manager;
	vector<DataSet> dataSets = manager.split(ds, k);
	trainer.validate();
	Mlp* mlp = trainer.mlp();

	for(uint i = 0; i<dataSets.size(); ++i){
		cout<<"\tPart (K): "<<i+1<<"\n";
		DataSet validationSet = dataSets.front();
		dataSets.erase(dataSets.begin());
		DataSet trainingSet = manager.join(dataSets);

		mlp = new Mlp(mlp->arch(), mlp->types(), false);
		trainer.batchSize(trainingSet.size());
		trainer.train(*mlp, trainingSet);
		theCommittee->addMlp(*mlp);
		//crossResults.push_back(CrossResult(mlp, trainingSet, validationSet));
		addToCommitteeOutput(*mlp, trainingSet, false);
		addToCommitteeOutput(*mlp, validationSet, true);
		dataSets.push_back(validationSet);
	}
}

void CrossValidator::addToCommitteeOutput(Mlp& mlp, DataSet& ds, bool validation)
{
	ds.reset();
	vector<uint> indices = ds.indices();
	vector<uint>::iterator it = indices.begin();
	vector<double>& output = (validation == true) ? valOutput : trnOutput;
	vector<uint>& target = (validation == true) ? valTarget : trnTarget;
	//vector<uint>& size = (validation == true) ? comSizeVal : comSizeTrn;

	while(ds.remaining()){
		Pattern& pat = ds.nextPattern();
		//pat.print(cout);
		vector<double>& tmp = mlp.propagate(pat.input());
		//printVector(tmp);
		output[*it] += tmp.front();
		target[*it] = (uint)pat.output().front(); //This doesn't vary.
		//size[*it]++;
		++it;
	}
}

uint CrossValidator::nkToIndex(uint n, uint k)
{
	//The -1 is because run 1 part 1 is at index 0
	return (n-1)*numParts+(k-1);
}
