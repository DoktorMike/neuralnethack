/*$Id: PrintUtils.cc 3344 2009-03-13 00:04:02Z michael $*/

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


#include "PrintUtils.hh"
#include "Saliency.hh"

#include <sstream>
#include <fstream>
#include <ostream>
#include <iterator>
#include <functional>
#include <algorithm>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace MultiLayerPerceptron;
using std::cout;
using std::endl;
using std::ostream;
using std::ostream_iterator;
using std::ostringstream;
using std::string;
using std::vector;
using std::bind2nd;
using std::plus;
using std::divides;

void PrintUtils::printTargetList(ostream& os, string id, DataSet& data)
{
	os<<id<<endl;
	for(uint i=0; i<data.size(); ++i){
		os<<"\t"<<data.pattern(i).idstring()<<"\t";
		vector<double>& target = data.pattern(i).output();
		for(uint j=0; j<target.size(); ++j)
			os<<std::scientific<<target[j]<<" ";
		os<<endl;
	}
}

void PrintUtils::printOutputList(ostream& os, string id, Mlp& mlp, DataSet& data)
{
	os<<id<<endl;
	for(uint i=0; i<data.size(); ++i){
		os<<"\t"<<data.pattern(i).idstring()<<"\t";
		vector<double> output = mlp.propagate(data.pattern(i).input());
		for(uint j=0; j<output.size(); ++j)
			os<<std::scientific<<output[j]<<" ";
		os<<endl;
	}
}

void PrintUtils::printOutputList(ostream& os, string id, Ensemble& c, DataSet& data)
{
	os<<id<<endl;
	for(uint i=0; i<data.size(); ++i){
		os<<"\t"<<data.pattern(i).idstring()<<"\t";
		vector<double> output = c.propagate(data.pattern(i).input());
		for(uint j=0; j<output.size(); ++j)
			os<<std::scientific<<output[j]<<" ";
		os<<endl;
	}
}

void PrintUtils::printTstEnslist(ostream& os, vector<Session>& sessions, 
		DataSet& trnData, DataSet& tstData, const Config& config)
{
	//Header
	os<<"#! ptype\t"<<!config.problemType()<<endl;
	os<<"#! targets\t"<<"1"<<endl;
	os<<"#! nout \t"<<trnData.nOutput()<<endl;

	//Target for training and testing data.
	PrintUtils::printTargetList(os, ">>target trn", trnData);
	PrintUtils::printTargetList(os, ">>target tst", tstData);

	for(uint i = 0; i<sessions.size(); ++i){
		Session& session = sessions.at(i);
		Ensemble& ensemble = *(session.ensemble);
		DataSet& trn = *(session.trnData);
		DataSet& tst = tstData;

		//Find the N and K
		std::ostringstream s;
		if(config.ensParamDataSelection() == "cs")
			s<<i/config.ensParamK()+1<<" "<<i%config.ensParamK()+1;
		else if(config.ensParamDataSelection() == "bagg")
			s<<i+1;

		//Print the training part.
		std::ostringstream s1;
		s1<<">>trn\t"<<s.str();
		PrintUtils::printOutputList(os, s1.str(), ensemble, trn);

		//print the testing part.
		std::ostringstream s2;
		s2<<">>tst\t"<<s.str();
		PrintUtils::printOutputList(os, s2.str(), ensemble, tst);
	}
}

void PrintUtils::printValEnslist(ostream& os, vector<Session>& sessions, 
		DataSet& trnData, DataSet& tstData, const Config& config)
{
	//Header
	os<<"#! ptype\t"<<!config.problemType()<<endl;
	os<<"#! targets\t"<<"1"<<endl;
	os<<"#! nout \t"<<trnData.nOutput()<<endl;

	//Target for training and testing data.
	PrintUtils::printTargetList(os, ">>target trn", trnData);
	PrintUtils::printTargetList(os, ">>target tst", tstData);

	//Outputlist for modelselection.
	for(uint i=0; i<sessions.size(); ++i){
		Session& session = sessions.at(i);
		Ensemble& ensemble = *(session.ensemble);
		DataSet& trn = *(session.trnData);
		DataSet& val = *(session.valData);

		//Find the N and K
		std::ostringstream s;
		if(config.msParamDataSelection() == "cv")
			s<<i/config.msParamK()+1<<" "<<i%config.msParamK()+1;
		else if(config.msParamDataSelection() == "boot")
			s<<i+1;

		//Print the training part.
		std::ostringstream s1;
		s1<<">>trn\t"<<s.str();
		PrintUtils::printOutputList(os, s1.str(), ensemble, trn);

		//print the validation part.
		std::ostringstream s2;
		s2<<">>val\t"<<s.str();
		PrintUtils::printOutputList(os, s2.str(), ensemble, val);
	}
}

void PrintUtils::printSaliencies(std::ostream& os, std::vector<Session>& sessions, const Config& config)
{
	vector<double> sals(config.architecture()[0], 0);
	for(vector<Session>::iterator it = sessions.begin(); it != sessions.end(); ++it){
		Ensemble* ensemble = it->ensemble;
		DataSet* trn = it->trnData;
		vector<double> tmp = Saliency::saliency(*ensemble, *trn, false);
		transform(sals.begin(), sals.end(), tmp.begin(), sals.begin(), plus<double>());
	}
	transform(sals.begin(), sals.end(), sals.begin(), bind2nd(divides<double>(), sessions.size()));
	Saliency::print(os, sals);
}

template<class T>
string argValueGen(const string arg, const T value)
{
	ostringstream s;
	s<<" "<<arg<<"=\""<<value<<"\"";
	//return s.str();
	return "";
}

template<class T>
void printXMLvector(ostream& os, const string indent, const string name, 
		const string args, const vector<T>& vec)
{
	os<<indent<<"<"<<name<<args<<">"<<endl<<indent<<"\t";
	copy(vec.begin(), vec.end(), ostream_iterator<T>(os, " "));
	os<<endl<<indent<<"</"<<name<<">"<<endl;
}

void printXMLnormaliser(ostream& os, const string indent, Normaliser& norm)
{
	os<<indent<<"<normalisation>"<<endl;
	printXMLvector(os, indent+"\t", "mean", "", norm.mean());
	printXMLvector(os, indent+"\t", "stddev", "", norm.stdDev());
	printXMLvector(os, indent+"\t", "skip", "", norm.skip());
	os<<indent<<"</normalisation>"<<endl;
}

void printXMLptype(ostream& os, const string indent, const string problemType)
{	os<<indent<<"<ptype>"<<endl<<indent<<"\t"<<problemType<<endl<<indent<<"</ptype>"<<endl; }

void printXMLlayer(ostream& os, const string indent, const string args, Layer& layer)
{
	os<<indent<<"<layer"<<args<<">"<<endl;
	printXMLvector(os, indent+"\t", "weights", "", layer.weights());
	os<<indent<<"</layer>"<<endl;
}

void printXMLmlp(ostream& os, const string indent, const string args, Mlp& mlp)
{
	os<<indent<<"<mlp"<<args<<">"<<endl;
	printXMLvector(os, indent+"\t", "arch", "", mlp.arch());
	printXMLvector(os, indent+"\t", "activation", "", mlp.types());
	vector<double> weights = mlp.weights();
	printXMLvector(os, indent+"\t", "weights", "", weights);
	os<<indent<<"</mlp>"<<endl;
}

void printXMLensemble(ostream& os, const string indent, const string args, Ensemble& ensemble)
{
	os<<indent<<"<ensemble"<<args<<">"<<endl;
	os<<indent<<"\t<weights> ";
	for(uint i=0; i<ensemble.size(); ++i) os<<ensemble.scale(i)<<" ";
	os<<indent<<"</weights>"<<endl;
	for(uint i=0; i<ensemble.size(); ++i)
		printXMLmlp(os, indent+"\t", argValueGen("nbr",i), ensemble.mlp(i));
	os<<indent<<"</ensemble>"<<endl;
}

void PrintUtils::printXML(ostream& os, vector<Session>& sessions, Normaliser& norm, const Config& config)
{
	os<<"<?xml version=\"1.0\" standalone=\"yes\"?>"<<endl;
	os<<"<networks>"<<endl;
	printXMLptype(os, "", config.problemType() == false ? "classification" : "regression");
	printXMLnormaliser(os, "", norm);
/*	if(sessions.size() > 1){ //All ensembles in the sessions contain only 1 Mlp
		Ensemble ensemble;
		for(vector<Session>::iterator it = sessions.begin(); it != sessions.end(); ++it){
			ensemble.addMlp(it->ensemble->mlp(0));
		}
		printXMLensemble(os, "", argValueGen("nbr",1), ensemble);
	}else{*/
	for(uint i=0; i<sessions.size(); ++i){
		Ensemble& ensemble = *(sessions.at(i).ensemble);
		printXMLensemble(os, "", argValueGen("nbr",i), ensemble);
	}
	//}
	os<<"</networks>"<<endl;
}

void PrintUtils::printXML(ostream& os, Ensemble& ensemble, Normaliser& norm, const Config& config)
{
	os<<"<?xml version=\"1.0\" standalone=\"yes\"?>"<<endl;
	os<<"<networks>"<<endl;
	printXMLptype(os, "", config.problemType() == false ? "classification" : "regression");
	printXMLnormaliser(os, "", norm);
	printXMLensemble(os, "", argValueGen("nbr",1), ensemble);
	os<<"</networks>"<<endl;
}
