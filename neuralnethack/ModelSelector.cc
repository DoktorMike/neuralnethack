#include "ModelSelector.hh"
#include "ModelEstimator.hh"
#include "Config.hh"
#include "Factory.hh"
#include "datatools/DataSet.hh"
#include "evaltools/EvalTools.hh"

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include <iterator>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <string>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace EvalTools;
using std::pair;
using std::make_pair;
using std::vector;
using std::map;
using std::string;
using std::cerr;
using std::endl;
using std::ofstream;
using std::setw;

// PUBLIC

ModelSelector::ModelSelector()
{}

ModelSelector::~ModelSelector()
{}

pair<Config, double> ModelSelector::findBestModel(DataSet& trnData, Config& config) {
    Config bestConfig = config;
    Config tmpConfig = config;
    double bestAuc = 0;
    map<string, vector<double> >& vary = config.vary();
    vector<double> alphas = sequence(vary["WeightElim"]);
    
    // Create output file and print header
    string fname = "msresult." + config.suffix() + ".txt";
    ofstream of(fname.c_str());
    of<<"#"<<setw(13)<<"Alpha"<<setw(14)<<"TrnAUC"<<setw(14)<<"ValAUC"<<endl;
    
    // Print results for all values of alpha and save the best
    for(vector<double>::iterator it = alphas.begin(); it != alphas.end(); ++it){
        tmpConfig.weightElimAlpha(*it);
        pair<double, double>* auc = trainAndValidateModel(trnData, tmpConfig);
        of<<setw(14)<<tmpConfig.weightElimAlpha()<<setw(14)<<auc->first<<setw(14)<<auc->second<<endl;
        if( auc->second > bestAuc ){
            bestConfig = tmpConfig;
            bestAuc = auc->second;
        }
        delete auc;
    }
    of.close();
    
    return make_pair(bestConfig, bestAuc);
}

// PRIVATE

/* Return the cross-validated error for a specific model */
pair<double, double>* ModelSelector::trainAndValidateModel(DataSet& trnData, const Config& config) {
    
    ModelEstimator* me = 0;
    pair<double, double>* auc = 0;
    
    if(config.msParamN() > 0){
        me = Factory::createModelEstimator(config, trnData);
        auc = me->runAndEstimateModel(&ErrorMeasures::auc);
        //Use 632 rule if bootstrap was used.
        if(config.msParamDataSelection() == "boot"){
            auc->second = Auc632PlusRule(auc->first, auc->second);
        }
    }else{
        cerr<<"Can't do model selection without MSParam set"<<endl;
        abort();
    }
    delete me;
    return auc;
}

vector<double> ModelSelector::sequence(const vector<double>& seq) {
    double nextval = seq[0];
    vector<double> res;
    while(nextval <= seq[1]){
        res.push_back(nextval);
        nextval += seq[2];
    }
    return res;
}

double ModelSelector::Auc632PlusRule(double meanTrn, double meanTst) {
    double r = 0;
    if(meanTrn > meanTst && meanTst > 0.5)
        r = (meanTst - meanTrn) / (0.5 - meanTrn);
    double w = 0.632 / (1.0 - 0.368*r);
    double meanTstPrime = (meanTst > 0.5) ? meanTst : 0.5;
    return (1.0-w)*meanTrn + w*meanTstPrime;
}


