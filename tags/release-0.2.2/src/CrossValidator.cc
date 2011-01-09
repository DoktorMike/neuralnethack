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
 valOutput(0), valTarget(0), crossResults(0)
{}

CrossValidator::CrossValidator(const CrossValidator& cv)
{*this = cv;}

CrossValidator::~CrossValidator()
{
	vector<CrossResult>::iterator it;
	for(it = crossResults.begin(); it != crossResults.end(); ++it)
		if(it->mlp != 0) delete it->mlp;
}

CrossValidator& CrossValidator::operator=(const CrossValidator& cv)
{
	if(this != &cv){
		numRuns = cv.numRuns;
		numParts = cv.numParts;
		valOutput = cv.valOutput;
		valTarget = cv.valTarget;
		crossResults = cv.crossResults;
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

void CrossValidator::crossValidate(Trainer& trainer, DataSet& ds, uint n, uint k)
{
	assert(n > 0 && k > 0);	
	
	numRuns = n; 
	numParts = k;
	crossResults = vector<CrossResult>(0);

	for(uint i=0; i<n; ++i){
		crossValidate(trainer, ds, k);
	}
}

void CrossValidator::printValidationStats(ostream& os, uint n, uint k, bool plot)
{printStats(os, n, k, true, plot);}

void CrossValidator::printTrainingStats(ostream& os, uint n, uint k, bool plot)
{printStats(os, n, k, false, plot);}

void CrossValidator::printStats(ostream& os, uint n, uint k, bool validation, bool plot)
{
	assert(k>=0 && n>=0);

	if(!os){
		std::cerr<<"Output stream error.\n"; return;
	}

	Roc roc;
	valOutput = vector<double>(0);
	valTarget = vector<uint>(0);

	if(n == 0){ //print all together
		for(vector<CrossResult>::iterator it=crossResults.begin(); it!=crossResults.end(); ++it)
			validate(*(it->mlp), (validation == false) ? it->trainingSet : it->validationSet);
	}else if(k == 0){ //print all k for spec n
		uint index = (n-1)*numParts;
		for(uint i=index; i<index+numParts; ++i){
			CrossResult& cr = crossResults[i];
			validate(*(cr.mlp), (validation == false) ? cr.trainingSet : cr.validationSet);
		}
	}else{ //print spec k for spec n
		uint index = (n-1)*numParts + (k-1);
		CrossResult& cr = crossResults[index];
		validate(*(cr.mlp), (validation == false) ? cr.trainingSet : cr.validationSet);
	}
	if(plot){
		roc.calcRoc(valOutput, valTarget);
		roc.print(os);
	}else{
		double auc = roc.calcAucWmw(valOutput, valTarget);
		os<<"AUC: "<<auc<<endl;
	}
}

//PRIVATE-----------------------------------------------------------------------

void CrossValidator::crossValidate(Trainer& trainer, DataSet& ds, uint k)
{
	DataManager manager;
	vector<DataSet> dataSets = manager.split(ds, k);
	trainer.validate();
	Mlp* mlp = trainer.mlp();

	for(uint i = 0; i<dataSets.size(); ++i){
		DataSet validationSet = dataSets.front();
		dataSets.erase(dataSets.begin());
		DataSet trainingSet = manager.join(dataSets);

		mlp = new Mlp(mlp->arch(), mlp->types(), false);
		trainer.batchSize(trainingSet.size());
		trainer.train(*mlp, trainingSet);
		crossResults.push_back(CrossResult(mlp, trainingSet, validationSet));
		dataSets.push_back(validationSet);
	}
}

void CrossValidator::validate(Mlp& mlp, DataSet& validationSet)
{
	assert(valOutput.size() == valTarget.size());

	validationSet.reset();
	while(validationSet.remaining()){
		Pattern& pat = validationSet.nextPattern();
		vector<double> tmp = mlp.propagate(pat.input());
		valOutput.push_back(tmp.front());
		valTarget.push_back((uint)(pat.output().front()));
	}
}

