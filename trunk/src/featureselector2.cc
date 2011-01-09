/*$Id: featureselector2.cc 1705 2008-02-04 21:07:12Z michael $*/

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

#include "neuralnethack/Config.hh"
#include "neuralnethack/parser/Parser.hh"
#include "neuralnethack/evaltools/Roc.hh"
#include "neuralnethack/NeuralNetHack.hh"
#include "neuralnethack/FeatureSelector.hh"
#include "neuralnethack/evaltools/EvalTools.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <functional>
#include <cmath>

using NeuralNetHack::Config;
using NeuralNetHack::Parser;
using NeuralNetHack::FeatureSelector;
using namespace EvalTools;

using std::vector;
using std::pair;
using std::map;
using std::string;
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::ostringstream;
using std::ostream_iterator;
using std::make_pair;
using std::unary_function;
using std::abs;
using std::advance;

template<class T> struct mapValueIndex : public unary_function<T, void>
{
	mapValueIndex():index(0){}
	void operator() (T& x){	sals.insert(make_pair(abs(x), ++index)); }
	map<T, uint> sals;
	uint index;
};

void parseConf(string fname, Config& config)
{
	ifstream confStream;

	cout<<"Parsing and storing Configuration."<<endl<<endl;
	confStream.open(fname.c_str(), ios::in);
	if(!confStream){ 
		cerr<<"Could not open configuration file: "<<fname<<endl;
		abort();
	}
	Parser::readConfigurationFile(confStream, config);
	confStream.close();
}

int main(int argc, char* argv[])
{
	Config config;
	uint minF=1, maxF=10, maxR=1;
	if(argc == 5){
		minF = atoi(argv[1]);
		maxF = atoi(argv[2]);
		maxR = atoi(argv[3]);
		parseConf(argv[4], config);
	}else{
		cerr<<"Usage: "<<endl<<argv[0]<<" MinFeatures MaxFeatures MaxRemove configfile"<<endl;
		exit(0);
	}

	srand48(config.seed() == 0 ? time(0) : config.seed()); //This is the ONLY place one may set the seed!

	FeatureSelector fs(minF, maxF, maxR);
	fs.run(config, ErrorMeasures::auc);

	return 0;
}
