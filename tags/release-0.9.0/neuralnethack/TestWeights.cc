/*$Id: TestWeights.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "Supervisor.hh"
#include "Parser.hh"

using namespace NeuralNetHack;
void indexTest(Mlp& mlp, vector<uint>& arch)
{
    cout<<endl<<"Simple test of the index operators:"<<endl;
    for(uint i=0; i<mlp.size(); ++i){
	cout<<endl<<"Layer "<<i<<":"<<endl;
	for(uint j=0; j<mlp[i].size(); ++j)
	    for(uint k=0; k<arch[i]+1; ++k)
		cout<<mlp[i][j][k]<<"\t";
    }
    cout<<endl;
}

void testMlp(Mlp& mlp, DataSet& data)
{
    while(data.remaining()){
	Pattern pattern = data.nextPattern();
	vector<double> input = pattern.input();
	vector<double> output = pattern.output();

	cout<<endl<<"Writing pattern "<<data.remaining()+1<<endl;
	for(uint j=0; j<input.size(); j++)
	    cout<<input[j]<<" ";
	cout<<"\t";
	for(uint j=0; j<output.size(); j++)
	    cout<<output[j]<<" ";
	cout<<"\t";

	cout<<"\nMlp output:\n";
	vector<double> outp = mlp.propagate(input);
	for(uint j=0; j<outp.size(); j++)
	    cout<<outp[j]<<" ";
    }
    cout<<endl;
    data.reset();

}

void printIt(vector<double>::iterator f, vector<double>::iterator l)
{
    vector<double>::iterator it;
	for(it=f; it<=l; ++it)
	    cout<<*it<<" ";
	cout<<endl;
}

void testWeights(Mlp& mlp)
{
    cout<<endl<<"Simple test of the weights:"<<endl;
    cout<<endl;
    cout<<"Testing weights in layer 0"<<endl;
    mlp[0].printWeights();
    
    cout<<"Testing weights in layer 1"<<endl;
    mlp[1].printWeights();
    cout<<endl;
}

void testMlps()
{
    vector<uint> arch(3);
    arch[0] = 2;
    arch[1] = 2;
    arch[2] = 1;

    vector<string> types(arch.size()-1);
    types[0] = SIGMOID;
    types[1] = SIGMOID;

    Mlp mlp1(arch, types, false);
    Mlp mlp2(mlp1);

    mlp1[0][0][0] = 1000;

    indexTest(mlp1, arch);
    indexTest(mlp2, arch);

    testWeights(mlp1);

    Parser parser;
    DataSet data;

    ifstream in;
    in.open("../datafiles/xor.dat");
    parser.readDataFile(in, arch.front(), arch.back(), data);
    in.close();

    testMlp(mlp1, data);
    testMlp(mlp2, data);
}

void testParser()
{
    Parser parser;
    Config config;

    ifstream in;
    in.open("./config.txt");
    parser.readConfigurationFile(in, config);
    in.close();
    config.print();

}

int main(void)
{
    //testMlps();
    //testParser();
    
    Parser parser;
    Config config;
    ifstream in;
    ifstream apa;
    
    cout<<endl<<"Parsing and storing Configuration."<<endl;
    in.open("./config.txt", ios::in);
    parser.readConfigurationFile(in, config);
    in.close();
    config.print();
    
    DataSet data;
    cout<<endl<<"Parsing and adding data to the DataSet."<<endl;
    apa.open(config.filename().c_str(), ios::in);
    assert(apa);
    parser.readDataFile(apa, config.inputColumns(), 
	    config.outputColumns(), data);
    apa.close();
    data.print();
    cout<<endl;

    Supervisor supervisor(config, data);
    supervisor.train();

    return 1;
}
