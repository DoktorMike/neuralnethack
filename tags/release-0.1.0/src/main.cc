#include "Supervisor.hh"
#include "Parser.hh"
#include "datatools/Normaliser.hh"

using namespace NetHack;
using namespace DataTools;

int main(int argc, char* argv[])
{
    string fname;
    if(argc>1)
	fname=string(argv[1]);
    else
	fname="./config.txt";
    Config config;
    ifstream in;
    ifstream apa;
    
    cout<<endl<<"Parsing and storing Configuration."<<endl;
    in.open(fname.c_str(), ios::in);
    Parser::readConfigurationFile(in, config);
    in.close();
    //config.print();
    
    DataSet data;
    cout<<endl<<"Parsing and adding data to the DataSet."<<endl;
    apa.open(config.filename().c_str(), ios::in);
    assert(apa);
    Parser::readDataFile(apa, config.inputColumns(), 
	    config.outputColumns(), data);
    apa.close();


    /*ofstream dataOrig("DataOrig.txt");
    ofstream dataNorm("DataNorm.txt");
    ofstream dataUnnorm("DataUnnorm.txt");
   
    data.print(dataOrig);
    cout<<"Normalising data...\n";
    
    Normaliser norm;
    norm.normalise(data, true);
    data.print(dataNorm);
    
    cout<<"Unnormalising data...\n";
    norm.unnormalise(data);
    data.print(dataUnnorm);
    
    dataOrig.close();
    dataNorm.close();
    dataUnnorm.close();*/

    cout<<"\nNormalising data.\n";
    Normaliser norm;
    norm.normalise(data, true);
    
    cout<<"\nTraining committee.\n";
    Supervisor supervisor(config, data);
    supervisor.train();
    //supervisor.test();

    return 1;
}
