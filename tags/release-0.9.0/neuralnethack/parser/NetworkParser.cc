/*$Id: NetworkParser.cc 1605 2007-01-24 20:47:36Z michael $*/

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

#include "../mlp/Mlp.hh"
#include "../datatools/DataSet.hh"
#include "../datatools/Normaliser.hh"
#include "Ensemble.hh"
#include "NetworkParser.hh"

#include <istream>
#include <string>
#include <algorithm>
#include <ext/algorithm>
#include <functional>

using namespace std;
using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;


// PUBLIC

NetworkParser::NetworkParser(){}

pair<vector<Ensemble*>, Normaliser*> NetworkParser::parseXML(istream& is)
{
	string token;
	vector<Ensemble*> committees;
	Normaliser* normalisation = 0;

	while(is>>token && token != "</networks>")
		if(token == "<committee>") committees.push_back(parseXMLcommittee(is));
		else if(token == "<normalisation>") normalisation = parseXMLnormalisation(is);

	return pair<vector<Ensemble*>, Normaliser*>(committees, normalisation);
}

Ensemble* NetworkParser::buildEnsemble(vector<Ensemble*>& committees)
{
	//cout<<"Found "<<committees.size()<<" ensembles"<<endl;
	Ensemble* committee = new Ensemble();
	for(vector<Ensemble*>::iterator it=committees.begin(); it!=committees.end(); ++it)
		for(uint i=0; i<(*it)->size(); ++i)
			committee->addMlp((*it)->mlp(i));
	//cout<<"Created a new Ensemble with "<<committee->size()<<" mlps"<<endl;
	return committee;
}

void NetworkParser::killAll(vector<Ensemble*>& committees, Ensemble* committee, Normaliser* normalisation)
{
	for(vector<Ensemble*>::iterator it=committees.begin(); it!=committees.end(); ++it) delete *it;
	delete committee;
	delete normalisation;
}


// PRIVATE 
void NetworkParser::parseXMLvector(istream& is, vector<uint>& vec, string stop)
{string token; while(is>>token && token != stop) vec.push_back(atoi(token.c_str()));}

void NetworkParser::parseXMLvector(istream& is, vector<double>& vec, string stop)
{string token; while(is>>token && token != stop) vec.push_back(atof(token.c_str()));}

void NetworkParser::parseXMLvector(istream& is, vector<string>& vec, string stop)
{string token; while(is>>token && token != stop) vec.push_back(token);}

Mlp* NetworkParser::parseXMLmlp(istream& is)
{
	string token = "";
	vector<uint> arch;
	vector<double> weights;
	vector<string> activations;

	while(is>>token && token != "</mlp>"){
		if(token == "<weights>"){
			parseXMLvector(is, weights, "</weights>");
		}else if(token == "<activation>"){
			parseXMLvector(is, activations, "</activation>");
		}else if(token == "<arch>"){
			parseXMLvector(is, arch, "</arch>");
		}
	}
	Mlp* mlp = new Mlp(arch, activations, false);
	mlp->weights(weights);
	return mlp;
}

Ensemble* NetworkParser::parseXMLcommittee(istream& is)
{
	Ensemble* committee = new Ensemble();
	string token = "";
	while(is>>token && token != "</committee>"){
		if(token == "<mlp>"){
			Mlp* mlp = parseXMLmlp(is);
			committee->addMlp(*mlp);
			delete mlp;
		}
	}
	return committee;
}

Normaliser* NetworkParser::parseXMLnormalisation(istream& is)
{
	string token = "";
	vector<double> mean, stddev;
	while(is>>token && token != "</normalisation>")
		if(token == "<mean>") parseXMLvector(is, mean, "</mean>");
		else if(token == "<stddev>") parseXMLvector(is, stddev, "</stddev>");
	Normaliser* normalisation = new Normaliser();
	normalisation->stdDev(stddev);
	normalisation->mean(mean);
	return normalisation;
}


