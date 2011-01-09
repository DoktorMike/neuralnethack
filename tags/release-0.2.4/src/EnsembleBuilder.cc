#include "EnsembleBuilder.hh"

using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;

//PUBLIC
EnsembleBuilder::~EnsembleBuilder(){delete theDataManager;}

DataManager* EnsembleBuilder::dataManager(){return theDataManager;}
void EnsembleBuilder::dataManager(DataManager* dm){theDataManager = dm;}

Trainer* EnsembleBuilder::trainer(){return theTrainer;}
void EnsembleBuilder::trainer(Trainer* t){theTrainer = t;}

DataSet* EnsembleBuilder::data(){return theData;}
void EnsembleBuilder::data(DataSet* d){theData = d;}

bool EnsembleBuilder::randomSampling(){return theDataManager->random();}
void EnsembleBuilder::randomSampling(bool rs){theDataManager->random(rs);}

//PROTECTED
EnsembleBuilder::EnsembleBuilder():theDataManager(new DataManager()), 
theTrainer(0), theData(0)//, theRandomSampling(true)
{}

EnsembleBuilder::EnsembleBuilder(const EnsembleBuilder& eb){*this = eb;}

EnsembleBuilder& EnsembleBuilder::operator=(const EnsembleBuilder& eb)
{
	if(this != &eb){
		theDataManager = new DataManager(*eb.theDataManager);
		theTrainer = eb.theTrainer;
		theData = eb.theData;
		//theRandomSampling = eb.theRandomSampling;
	}
	return *this;
}

bool EnsembleBuilder::isValid(){return theDataManager && theTrainer && theData;}

//PRIVATE

