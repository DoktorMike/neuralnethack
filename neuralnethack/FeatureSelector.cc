#include "FeatureSelector.hh"
#include "ModelSelector.hh"
#include "EnsembleBuilder.hh"
#include "Factory.hh"
#include "Ensemble.hh"

#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "datatools/Sampler.hh"

#include "parser/Parser.hh"

#include <utility>
#include <fstream>
#include <iostream>
#include <functional>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <iterator>

using namespace NeuralNetHack;
using NeuralNetHack::Parser;
using DataTools::CoreDataSet;
using DataTools::DataSet;
using DataTools::Sampler;
using DataTools::Normaliser;
using std::pair;
using std::string;
using std::vector;
using std::map;
using std::ifstream;
using std::ofstream;
using std::ostream_iterator;
using std::ios;
using std::endl;
using std::cerr;
using std::cout;
using std::divides;
using std::transform;
using std::set_difference;

FeatureSelector::FeatureSelector(uint minf, uint maxf, uint maxr):minFeatures(minf), maxFeatures(maxf), maxRemove(maxr){}

//template<class T>
Config FeatureSelector::run(Config& config, double (*f)(Ensemble&, DataSet&) )
{
	cout<<"Running FeatureSelector with min: "<<minFeatures<<" max: "<<maxFeatures<<" maxRemove: "<<maxRemove<<endl;
	best = config;
	bool stop = false;
	ModelSelector ms;
	vector< pair<double, vector<uint> > > results;
	string s = "featuresel." + best.suffix() + ".txt";
	ofstream os(s.c_str());
	os<<"AUC\tInputs"<<endl;
	//cerr<<"Before while"<<endl;
	//cerr.flush();

	while(!stop){
		DataSet trnData, tstData;
		//cerr<<"Parse Data"<<endl;
		//cerr.flush();
		parseData(best, trnData, tstData);
		//cerr<<"Create sampler"<<endl;
		//cerr.flush();
		Sampler* sampler = Factory::createSampler(best, trnData);
		//cerr<<"Create effects"<<endl;
		//cerr.flush();
		vector<double> effects(best.inputColumns().size()); //Storing the effect removing and index has on AUC
		double meanAuc = 0;
		//cerr<<"Before 2nd while"<<endl;
		//cerr.flush();
		while(sampler->hasNext()){
			//cerr<<"Finding optimal model"<<endl;
			//cerr.flush();
			// Find optimal parameters for ensemble
			pair<DataSet, DataSet>* sample = sampler->next();
			DataSet trn = sample->first, val = sample->second;
			pair<Config, double> msres = ms.findBestModel(sample->first, best);
			//cerr<<"Building optimal model"<<endl;
			//cerr.flush();
			// Build optimal ensemble
			EnsembleBuilder* eb = Factory::createEnsembleBuilder(msres.first, sample->first);
			Ensemble* e = eb->buildEnsemble();
			// Calculate performance for current sample
			double auc = (*f)(*e, sample->second);
			//cerr<<"AUC: "<<auc<<endl;
			storeClampingEffect(*e, val, auc, effects, f);
			meanAuc += auc;

			// Kill off some memory
			delete sample;
			delete eb;
			delete e;
		}
		//cerr<<"Store and print performance"<<endl;
		//cerr.flush();
		// Store and print the performance for these variables
		meanAuc = meanAuc / sampler->howMany();
		results.push_back(make_pair(meanAuc, best.inputColumns()));
		os<<meanAuc<<"\t";
		copy(best.inputColumns().begin(), best.inputColumns().end(), ostream_iterator<double>(os, " "));
		os<<endl;

		//cerr<<"Remove redundant variables"<<endl;
		//cerr.flush();
		// Remove redundant variables
		transform(effects.begin(), effects.end(), effects.begin(), bind2nd(divides<double>(), (double)effects.size()));
		cerr<<"Effects: ";
		copy(effects.begin(), effects.end(), ostream_iterator<double>(cerr, " "));
		cerr<<endl;
		vector<uint> toRemove = removeFeatures(effects, best.inputColumns());
		vector<uint> toKeep;
		vector<uint> indices = best.inputColumns();
		//cerr<<"Calc set difference"<<endl;
		//cerr.flush();
		set_difference(indices.begin(), indices.end(), toRemove.begin(), toRemove.end(), back_inserter(toKeep));
		//cerr<<"Set variables to keep"<<endl;
		//cerr.flush();
		best.inputColumns(toKeep);
		best.inputColumnsT(toKeep);
		vector<uint> tmp = best.architecture();
		tmp[0] = toKeep.size();
		best.architecture(tmp);

		//cerr<<"Check stop condition"<<endl;
		//cerr.flush();
		//Check stop condition
		//TODO improve this so that we continue removing features until we hit minFeatures.
		if(best.inputColumns().size() < maxFeatures){
			stop = true;
		}
		//cerr<<"Delete sampler"<<endl;
		//cerr.flush();
		delete sampler;
		trnData.killCoreData();
		tstData.killCoreData();
	}
	os.close();
	return best;
}

// PRIVATE

vector<uint> FeatureSelector::removeFeatures(std::vector<double>& effects, const std::vector<uint>& inputs)
{
	vector<uint> indices; // the indices to remove
	map<double, uint> order;

	for(uint i=0; i<effects.size(); ++i) order[effects[i]] = inputs[i];
	for(map<double, uint>::iterator i = order.begin(); i != order.end(); ++i){
		//cerr<<"Value: "<<i->first<<" Index: "<<i->second<<endl;
		uint index = i->second;
		if(indices.size() >= maxRemove) break;
		indices.push_back(index);
	}
	sort(indices.begin(), indices.end());
	//cerr<<"Features to remove: ";
	//copy(indices.begin(), indices.end(), ostream_iterator<uint>(cerr, " "));
	//cerr<<endl;
	return indices;
}

void FeatureSelector::parseData(Config& config, DataSet& trnData, DataSet& tstData)
{
	//cerr<<"Allocating core data"<<endl;
	//cerr.flush();
	ifstream trnStream;
	ifstream tstStream;
	CoreDataSet* trnCoreData = new CoreDataSet();
	CoreDataSet* tstCoreData = new CoreDataSet();

	//cerr<<"Open training stream"<<endl;
	//cerr.flush();
	trnStream.open(config.fileName().c_str(), ios::in);
	//cerr<<"Opening file: "<<config.fileName().c_str()<<endl;
	//cerr.flush();
	if(!trnStream || trnStream.bad() || trnStream.fail() || trnStream.eof()){
		cerr<<"Could not open data file: "<<config.fileName()<<endl;
		abort();
	}
	//cerr<<"Parsing training stream"<<endl;
	//cerr.flush();
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(), 
			config.outputColumns(), config.rowRange(), *trnCoreData);
	trnStream.close();
	trnData.killCoreData();
	trnData.coreDataSet(*trnCoreData);

	//cerr<<"Open testing stream"<<endl;
	//cerr.flush();
	tstStream.open(config.fileNameT().c_str(), ios::in);
	if(!tstStream){
		cerr<<"Could not open data file: "<<config.fileNameT()<<endl;
		abort();
	}
	//cerr<<"Parsing testing stream"<<endl;
	//cerr.flush();
	Parser::readDataFile(tstStream, config.idColumnT(), config.inputColumnsT(), 
			config.outputColumnsT(), config.rowRangeT(), *tstCoreData);
	tstStream.close();
	tstData.killCoreData();
	tstData.coreDataSet(*tstCoreData);

	Normaliser norm;
	cout<<"Normalizing data"<<endl; 
	if(config.normalization() == "Z"){
		norm.calcAndNormalise(trnData, true); 
		norm.normalise(tstData);
	}
}

//template<class T>
void FeatureSelector::storeClampingEffect(Ensemble& e, DataSet& d, double auc, 
		vector<double>& aucImpact, double (*f)(Ensemble&, DataSet&))
{
	for(uint i=0; i<d.nInput(); ++i){
		//Get original value and mean value for index i
		vector<double> origValues; origValues.reserve(d.size());
		for(uint j=0; j<d.size(); ++j) origValues.push_back(d.pattern(j).input()[i]);
		double meanValue = mean(d, i);

		for(uint j=0; j<d.size(); ++j) d.pattern(j).input()[i] = meanValue;
		double newAuc = (*f)(e, d);
		aucImpact[i] += (auc - newAuc);
		//cerr<<"Impact "<<i<<": "<<aucImpact[i]<<endl;
		//cerr.flush();
		for(uint j=0; j<d.size(); ++j) d.pattern(j).input()[i] = origValues[j];
	}
}

double FeatureSelector::mean(DataSet& d, uint index)
{
	double mean = 0;
	for(uint i=0; i<d.size(); ++i) mean += d.pattern(i).input()[index];
	return mean / (double)d.size();
}

