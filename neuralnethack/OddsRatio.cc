/*$Id: OddsRatio.cc 1594 2007-01-12 16:20:01Z michael $*/

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


#include "OddsRatio.hh"
#include "matrixtools/MatrixTools.hh"

#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>

using namespace NeuralNetHack;
using MultiLayerPerceptron::Mlp;
using MultiLayerPerceptron::Layer;
using DataTools::DataSet;
using DataTools::Pattern;

using std::vector;
using std::plus;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;

//Effective odds ratio calculations

vector<double> OddsRatio::oddsRatio(Ensemble& committee, DataSet& data)
{
	vector<double> oddsrat(data.nInput(),0);

	for(uint i=0; i<data.size(); ++i){
		vector<double> tmp = oddsRatio(committee, data.pattern(i));
		transform(oddsrat.begin(), oddsrat.end(), tmp.begin(), 
				oddsrat.begin(), plus<double>());
	}
	for(vector<double>::iterator it=oddsrat.begin(); it!=oddsrat.end(); ++it)
		(*it) /= (double)data.size();
	return oddsrat;
}

vector<double> OddsRatio::oddsRatio(Ensemble& committee, Pattern& pattern)
{
	vector<double> input = pattern.input();
	vector<double> oddsrat(input.size(), 0);

	for(uint i=0; i<input.size(); ++i){
		double tmp = input[i];
		input[i] = 0;
		double absent = committee.propagate(input).front();
		input[i] = 1;
		double present = committee.propagate(input).front();
		input[i] = tmp;
		oddsrat[i] = (present/(1.0-present))/(absent/(1.0-absent));
	}
	return oddsrat;
}

//Utilities like printing and stuff

void OddsRatio::print(ostream& os, vector<double>& oddsrat)
{
	if(!os){
		cerr<<"OddsRatio::print: Problem with output stream."<<endl;
		return;
	}

	os<<"Input\tOR"<<endl;
	for(uint i=0; i<oddsrat.size(); ++i) os<<i<<"\t"<<oddsrat[i]<<endl;
}
