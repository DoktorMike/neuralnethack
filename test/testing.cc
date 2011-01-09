/*$Id: testing.cc 1626 2007-05-08 12:08:19Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#include "matrixtools/MatrixTools.hh"
#include "datatools/Normaliser.hh"
#include "datatools/DataManager.hh"
#include "datatools/CoreDataSet.hh"
//#include "datatools/Bootstrapper.hh"
//#include "datatools/CrossValidator.hh"
#include "evaltools/Roc.hh"
#include "parser/Parser.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"
#include "mlp/CrossEntropy.hh"
#include "Factory.hh"
#include "OddsRatio.hh"
#include "Saliency.hh"

#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace NeuralNetHack;
using namespace std;

void testRoc()
{
	EvalTools::Roc roc;

/*
	double out[] = {0.630477,0.52469,0.630477,0.630477,0.461932,0.630477,
		0.630477,0.630477,0.630477,0.63048,0.630477,0.405618,0.412675,0.524524,
		0.524525,0.461721,0.630477,0.461731};
	uint tar[]   = {1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0};
	vector<double> output(out, &out[17]+1);
	vector<uint> target(tar, &tar[17]+1);
*/
	double out[] = {0, -10, 2, -3, 13, 1, 25, 15};
	uint tar[] = {1,0,1,0,1,0,1,0};
	vector<double> output(out, &out[7]+1);
	vector<uint> target(tar, &tar[7]+1);

	//print(output); print(target);

	cout<<"AUC: ";
	cout<<"Trpz: "<<roc.calcAucTrapezoidal(output, target)<<" ";
	cout<<"Wmw: "<<roc.calcAucWmw(output, target)<<" ";
	cout<<"WmwFast: "<<roc.calcAucWmwFast(output, target)<<" ";
	cout<<endl;
}

void testMlpCopy()
{
	vector<uint> arch;
	arch.push_back(10);
	arch.push_back(5);
	arch.push_back(4);
	vector<string> types;
	types.push_back(TANHYP);
	types.push_back(SIGMOID);
	vector<MultiLayerPerceptron::Mlp*> mlps;

	cout<<"Creating 30000 Mlps.\n";
	for(uint i=0; i<30000; ++i){
		MultiLayerPerceptron::Mlp* mlp = new MultiLayerPerceptron::Mlp(arch, types, false);
		mlps.push_back(mlp);
	}
	cout<<"Deleting 30000 Mlps.\n";
	for(vector<MultiLayerPerceptron::Mlp*>::iterator it = mlps.begin(); it != mlps.end(); ++it)
		delete *it;
	mlps.clear();
	cout<<"Creating 30000 copies of one Mlp.\n";
	MultiLayerPerceptron::Mlp* mlpOrig = new MultiLayerPerceptron::Mlp(arch, types, false);
	for(uint i=0; i<30000; ++i){
		MultiLayerPerceptron::Mlp* mlp = new MultiLayerPerceptron::Mlp(*mlpOrig);
		mlps.push_back(mlp);
	}
	cout<<"Deleting the 30000 copies of one Mlp.\n";
	for(vector<MultiLayerPerceptron::Mlp*>::iterator it = mlps.begin(); it != mlps.end(); ++it)
		delete *it;
	mlps.clear();
}

void testDataManager(DataTools::DataSet& data, Config& config)
{
	DataTools::DataManager manager;
	manager.random(true);

	//ofstream os("crapbefore",ios::out); data.print(os); os.close();
	//vector<DataTools::DataSet> dataSets = manager.split(data, (uint)1);
	//os.open("crapafter",ios::out); dataSets.front().print(os); os.close();

	cout<<"Testing splitting of data using number of training data.\n";
	pair<DataTools::DataSet, DataTools::DataSet>* trnVal = manager.split(data, config.msParamNumTrainingData());
	trnVal->first.print(cout); cout<<endl;
	trnVal->second.print(cout); cout<<endl;
	delete trnVal;

	cout<<"Testing splitting the first of the two previous.\n";
	trnVal = manager.split(trnVal->first, config.msParamNumTrainingData());
	trnVal->first.print(cout); cout<<endl;
	trnVal->second.print(cout); cout<<endl;
	delete trnVal;

	cout<<"Testing splitting of data using bootstrapping.\n";
	trnVal = manager.split(data);
	trnVal->first.print(cout); cout<<endl;
	trnVal->second.print(cout); cout<<endl;
	delete trnVal;

	cout<<"Testing splitting the first of the previous.\n";
	trnVal = manager.split(trnVal->first);
	trnVal->first.print(cout); cout<<endl;
	trnVal->second.print(cout); cout<<endl;
	delete trnVal;

	cout<<"Testing splitting of data using number K.\n";
	vector<DataTools::DataSet>* datasets = manager.split(data, config.msParamK());
	vector<DataTools::DataSet>::iterator it = datasets->begin();
	do{
		it->print(cout);
		cout<<"\n\n";
	}while(++it != datasets->end());
	delete datasets;

	cout<<"Testing splitting the first of the previous.\n";
	datasets = manager.split(datasets->front(), config.msParamK());
	it = datasets->begin();
	do{
		it->print(cout);
		cout<<endl<<endl;
	}while(++it != datasets->end());

	cout<<"Testing joining of data.\n";
	DataTools::DataSet d = manager.join(*datasets);
	d.print(cout);
	delete datasets;
}

void buildOutputTarget(Ensemble& committee, DataTools::DataSet& dset, vector<double>& output, vector<uint>& target)
{
	output.clear();
	target.clear();

	for(uint i=0; i<dset.size(); ++i){
		DataTools::Pattern& pat = dset.pattern(i);
		vector<double> tmp = committee.propagate(pat.input());
		output.push_back(tmp.front());
		target.push_back((uint)pat.output().front());
		//cout<<(uint)pat.output().front()<<"\t";
		//cout<<tmp.front()<<"\n";
	}
}

void evaluateModel(Ensemble& committee, DataTools::DataSet& data)
{
	vector<double> output;
	vector<uint> target;
	EvalTools::Roc roc;
	buildOutputTarget(committee, data, output, target);
	cout<<"Trpz: "<<roc.calcAucTrapezoidal(output, target)<<" ";
	cout<<"Wmw: "<<roc.calcAucWmw(output, target)<<" ";
	cout<<"WmwFast: "<<roc.calcAucWmwFast(output, target)<<"\n";
}

void testCoreDataSetAndDataSet(DataTools::CoreDataSet& data)
{
	cout<<"Creating a DataSet from CoreDataSet."<<endl;
	vector<uint> indices;
	//indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	//indices.push_back(3);
	//indices.push_back(4);
	DataTools::DataSet dset;
	dset.coreDataSet(data);
	dset.indices(indices);
	dset.print(cout);

	cout<<"Creating a copy of the DataSet."<<endl;
	DataTools::DataSet dset2(dset);
	dset2.print(cout);
}

void testMlp(MultiLayerPerceptron::Mlp* mlp, DataTools::DataSet& dset)
{
	for(uint i=0; i<dset.size(); ++i){
		DataTools::Pattern& p = dset.pattern(i);
		vector<double>& in = p.input();
		vector<double>& out = mlp->propagate(in);
		vector<double>& dout = p.output();
		cout<<"In: ";
		for(uint i=0; i<in.size(); ++i)
			cout<<in[i]<<" ";
		cout<<"\tOut: ";
		for(uint i=0; i<out.size(); ++i)
			cout<<out[i]<<" ";
		cout<<"\tTarget: ";
		for(uint i=0; i<dout.size(); ++i)
			cout<<dout[i]<<" ";
		cout<<"\n";
	}
}

void testTrainer(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData)
{
	MultiLayerPerceptron::Trainer* trainer = Factory::createTrainer(config, trnData);
	trainer->train(cout);
	trainer->mlp()->printWeights(cout);

	//testMlp(trainer->mlp(), trnData);
	Ensemble c(*(trainer->mlp()), 1);
	double trnErrCE = EvalTools::ErrorMeasures::crossEntropy(c, trnData);
	double trnErrAUC = EvalTools::ErrorMeasures::auc(c, trnData);
	double tstErrCE = EvalTools::ErrorMeasures::crossEntropy(c, tstData);
	double tstErrAUC = EvalTools::ErrorMeasures::auc(c, tstData);
	cout<<"TrnAUC: "<<trnErrAUC<<endl;
	cout<<"TrnCE: "<<trnErrCE<<endl;
	cout<<"TstAUC: "<<tstErrAUC<<endl;
	cout<<"TstCE: "<<tstErrCE<<endl;

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
}

void testEnsemble(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData)
{
	MultiLayerPerceptron::Trainer* trainer = Factory::createTrainer(config, trnData);
	trainer->train(cout);
	Ensemble c(*(trainer->mlp()), 1);
	MultiLayerPerceptron::Mlp* cMlp = new MultiLayerPerceptron::Mlp(*(trainer->mlp()));
	c.addMlp(*cMlp);

	Ensemble* cCopy = new Ensemble(c);

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
	delete cCopy;
	delete cMlp;
}

void testOddsRatio(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData)
{
	MultiLayerPerceptron::Trainer* trainer = Factory::createTrainer(config, trnData);
	trainer->train(cout);

	cout<<"Testing OddRatios."<<endl;
	Ensemble c(*(trainer->mlp()), 1.0);
	vector<double> oddsrat = OddsRatio::oddsRatio(c, trnData);
	for(vector<double>::iterator it=oddsrat.begin(); it!=oddsrat.end(); ++it)
		cout<<"Odds Ratio for input "<<it - oddsrat.begin()<<": "<<*it<<endl;

	cout<<"Testing Saliency."<<endl;
	oddsrat = Saliency::saliency(*(trainer->mlp()), trnData);
	for(vector<double>::iterator it=oddsrat.begin(); it!=oddsrat.end(); ++it)
		cout<<"Saliency for input "<<it - oddsrat.begin()<<": "<<*it<<endl;
	//trainer->mlp()->printWeights(cout);

	delete trainer->mlp();
	delete trainer->error();
	delete trainer;
}

void parseConfAndData(string fname, Config& config, DataTools::CoreDataSet& trnData, DataTools::CoreDataSet& tstData)
{
	ifstream confStream;
	ifstream trnStream;
	ifstream tstStream;

	confStream.open(fname.c_str(), ios::in);
	assert(confStream);
	Parser::readConfigurationFile(confStream, config);
	confStream.close();

	trnStream.open(config.fileName().c_str(), ios::in);
	assert(trnStream);
	Parser::readDataFile(trnStream, config.idColumn(), config.inputColumns(), 
			config.outputColumns(), config.rowRange(), trnData);
	trnStream.close();

	tstStream.open(config.fileNameT().c_str(), ios::in);
	assert(tstStream);
	Parser::readDataFile(tstStream, config.idColumnT(), config.inputColumnsT(), 
			config.outputColumnsT(), config.rowRangeT(), tstData);
	tstStream.close();
}

void errFcnVsPType(Config& config)
{
	if(config.problemType() == true && config.errFcn() == CEE){
		cerr<<"Regression should not be performed with kullback leibler error function.\n";
		abort();
	}
}

int main(int argc, char* argv[])
{
	srand(1);
	string fname;
	if(argc>1) fname=string(argv[1]); else fname="./config.txt";

	//testMlpCopy();
	Config config;
	DataTools::CoreDataSet trnCoreData;
	DataTools::CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	//trnCoreData.print(cout);
	//tstCoreData.print(cout);
	//testCoreDataSetAndDataSet(trnCoreData);
	//testDataManager(trnData, config);
	DataTools::Normaliser norm;
	norm.normalise(trnData, true);
	norm.normalise(tstData, true);
	//testEnsemble(config, trnData, tstData);
	//testTrainer(config, trnData, tstData);
	//testOddsRatio(config, trnData, tstData);
	//testCrossValidator(trnData, config);
	//testCrossSplitter(trnData, tstData, config);
	//testBagger(trnData, tstData, config);
	//testBootstrapper(trnData, tstData, config);
	//testCrossValidator(trnData, tstData, config);
	//testRoc();

	return 0;
}
