/*$Id: testParser.cc 1619 2007-05-07 13:39:55Z michael $*/

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


#include "datatools/DataSet.hh"
#include "datatools/CoreDataSet.hh"
#include "parser/Parser.hh"

#include <iostream>
#include <fstream>
#include <cassert>

using namespace NeuralNetHack;
using namespace std;

int testParser(Config& config, DataTools::DataSet& trnData, DataTools::DataSet& tstData)
{
	ofstream ofs("tmp.txt");
	config.print(ofs);
	ofs.close();
	int ret = system("diff -q parserconfigure.txt tmp.txt");
	system("rm tmp.txt");
	return (ret > 0) ? -1 : 0;
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

int main(int argc, char* argv[])
{
	srand(1);

	string fname;
	//if(argc>1) fname=string(argv[1]); else fname="./apa.txt";
	if(argc>1) fname=string(argv[1]); else fname="./parserconfigure.txt";

	Config config;
	DataTools::CoreDataSet trnCoreData;
	DataTools::CoreDataSet tstCoreData;
	parseConfAndData(fname, config, trnCoreData, tstCoreData);
	DataTools::DataSet trnData;
	DataTools::DataSet tstData;
	trnData.coreDataSet(trnCoreData);
	tstData.coreDataSet(tstCoreData);
	return testParser(config, trnData, tstData);
}
