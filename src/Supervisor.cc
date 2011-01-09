#include "Supervisor.hh"
#include "GradientDescent.hh"
#include "QuasiNewton.hh"
#include "evaltools/Roc.hh"

using namespace NetHack;

Supervisor::Supervisor(Config& config, DataSet& dset)
{
    this->config = &config;
    
    Mlp mlp(config.arch(),config.activationFunctions(),false);
    theCommittee = new Committee(mlp, 1);
    //mlp=Mlp(config.arch(),config.activationFunctions(),false);
    //theCommittee->addMlp(mlp);
    
    dataSet = &dset;

    if(config.learningAlgorithm() == GD)
	trainer = new GradientDescent(config.errorFunction(), 
		MAX_ERROR, config.batchSize(), config.learningRate(), 
		config.decLearningRate(), config.momentum());
    else if(config.learningAlgorithm() == QN)
	trainer = new QuasiNewton(config.errorFunction(), MAX_ERROR, 
		config.batchSize());
}

Supervisor::~Supervisor()
{
    delete trainer;
    delete theCommittee;
}

void Supervisor::train()
{
    trainer->train(*theCommittee, *dataSet, config->nEpochs());
}

void Supervisor::test()
{
    using EvalTools::Roc;
    using namespace DataTools;
    
    dataSet->reset();
    vector<double> out;
    vector<uint> dout;
    while(dataSet->remaining()){
	Pattern& p=dataSet->nextPattern();
	vector<double> output=theCommittee->propagate(p.input());
	out.push_back(output.front());
	dout.push_back((uint)round(p.output().front()));
    }
    Roc roc;
    double bf = roc.calcAucBf(out, dout);
    double wmw = roc.calcAucWmw(out, dout);

    cout<<"Trapezoidal rule AUC: "<<bf<<endl;
    cout<<"WMW AUC: "<<wmw<<endl;
    
    ofstream os("roc.dat");
    if(!os){
	std::cerr<<"Couldn't open file shit.plot for writing.";
	abort();
    }
    roc.print(os);
    os.close();
    //(theCommittee->mlp(0)).printWeights();
}

//PRIVATE--------------------------------------------------------------------//

Supervisor::Supervisor(const Supervisor& sup){*this=sup;}

Supervisor& Supervisor::operator=(const Supervisor& sup){return *this;}


