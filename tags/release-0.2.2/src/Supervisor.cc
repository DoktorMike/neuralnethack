#include "Supervisor.hh"
#include "GradientDescent.hh"
#include "QuasiNewton.hh"
#include "SummedSquare.hh"
#include "evaltools/Roc.hh"

#include <cmath>

using namespace NeuralNetHack;

Supervisor::Supervisor(Config& config, DataSet& dset)
{
	this->config = &config;

	mlp = new Mlp(config.arch,config.actFcn,false);
	dataSet = &dset;

	if(config.minMethod == GD)
		trainer = new GradientDescent(
				MAX_ERROR, 
				config.batchSize, 
				config.learningRate, 
				config.decLearningRate, 
				config.momentum,
				config.weightElimOn,
				config.weightElimAlpha, 
				config.weightElimW0);
	else if(config.minMethod == QN)
		trainer = new QuasiNewton(
				MAX_ERROR, 
				config.batchSize,
				config.weightElimOn,
				config.weightElimAlpha, 
				config.weightElimW0);
	
	if(config.errFcn == SSE)
		error = new SummedSquare();
	trainer->error(error);
}

Supervisor::~Supervisor()
{
	delete trainer;
	delete mlp;
	delete error;
}

void Supervisor::train()
{
	trainer->numEpochs(config->maxEpochs);
	trainer->train(*mlp, *dataSet);
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
		vector<double> output=mlp->propagate(p.input());
		out.push_back(output.front());
		dout.push_back((uint)round(p.output().front()));
	}
	Roc roc;
	double bf = roc.calcAucTrapezoidal(out, dout);
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


