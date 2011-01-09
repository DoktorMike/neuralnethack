#include "SummedSquare.hh"

using namespace NetHack;
using namespace DataTools;

SummedSquare::SummedSquare():Error(SSE){}

SummedSquare::~SummedSquare(){}

vector<double>& SummedSquare::gradient(Mlp& mlp, vector<double>& in, 
	vector<double>& out, vector<double>& dout)
{
    Layer& last = mlp[mlp.nLayers()-1];

    localGradient(last, out, dout);
    backpropagate(mlp);
    gradient(mlp[0], in);

    for(uint i=1; i<mlp.nLayers(); ++i)
	gradient(mlp[i],mlp[i-1]);

    return theGradient=mlp.gradients();//gradient(mlp);
}

double SummedSquare::gradient(Mlp& mlp, DataSet& dset, uint bs)
{
    double err=0;

    //Set all gradients to zero.
    for(uint i=0; i<mlp.nLayers(); ++i)
	for(uint j=0; j<mlp[i].nNeurons(); ++j)
	    for(uint k=0; k<mlp[i][j].nWeights(); ++k)
		mlp[i][j].gradient(k,0);
		
    for(uint i=0; i<bs; ++i){
	Pattern& p = dset.nextPattern();
	vector<double> out = mlp.propagate(p.input());
	Layer& last = mlp[mlp.nLayers()-1];
	localGradient(last, out, p.output());
	backpropagate(mlp);
	gradientBatch(mlp[0], p.input());

	for(uint i=1; i<mlp.nLayers(); ++i)
	    gradientBatch(mlp[i],mlp[i-1]);
	err += outputError(out, p.output());
    }
    
    //Divide all gradients by batch size.
    for(uint i=0; i<mlp.nLayers(); ++i)
	for(uint j=0; j<mlp[i].nNeurons(); ++j)
	    for(uint k=0; k<mlp[i][j].nWeights(); ++k){
		double tmp=mlp[i][j].gradient(k);
		//Note the minus. Done here and not in the gradient
		//calculations. thus the gradient outputs should be noticed
		//negated!!!
		mlp[i][j].gradient(k,-tmp/(double)bs);
	    }
    return 0.5*err/bs;
}

double SummedSquare::outputError(Mlp& mlp, DataSet& dset, uint bs)
{
    double err=0;
    for(uint i=0; i<bs; ++i){
	Pattern& p=dset.nextPattern();
	vector<double> output=mlp.propagate(p.input());
	err+=outputError(output, p.output());
    }
    return 0.5*err/bs;
}

//PRIVATE--------------------------------------------------------------------//

SummedSquare::SummedSquare(const SummedSquare& sse):Error(sse.theType){*this = sse;}

SummedSquare& SummedSquare::operator=(const SummedSquare& sse){return *this;}

void SummedSquare::localGradient(Layer& ol, vector<double>& out, 
	vector<double>& dout)
{
    assert(out.size() == ol.size() && dout.size() == out.size());

    vector<double>::iterator ito = out.begin();
    vector<double>::iterator itdo = dout.begin();

    //cout<<"Output layer local gradient: ";
    for(uint i=0; i<ol.size(); ++i, ++ito, ++itdo){
	double tmp = ol[i].firePrime();
	ol[i].localGradient((*itdo - *ito) * tmp);
	//cout<<"lg = ("<<*itdo<<"-"<<*ito<<")*"<<tmp<<" = ";
	//cout<<ol[i].localGradient()<<"; ";
    }
    //cout<<endl;
}

void SummedSquare::backpropagate(Mlp& mlp)
{
    for(int i=mlp.size()-1; i>0; --i)
	localGradient(mlp[i-1], mlp[i]);
}

void SummedSquare::localGradient(Layer& curr, Layer& next)
{
    //cout<<"Hidden layer local gradients: ";
    for(uint j=0; j<curr.size(); ++j){
	double err = 0;
	for(uint i=0; i<next.size(); ++i)
	    err += next[i].localGradient()*next[i][j];
	err = err*curr[j].firePrime();
	curr[j].localGradient(err);
	//cout<<curr[j].localGradient()<<" ";
    }
    //cout<<endl;
}

void SummedSquare::gradient(Layer& first, vector<double>& in)
{
    for(uint i=0; i<first.size(); ++i){
	for(uint j=0; j<in.size(); ++j)
	    first[i].gradient(j, first[i].localGradient() * in[j]);
	first[i].gradient(in.size(), first[i].localGradient()); //bias
    }
}

void SummedSquare::gradientBatch(Layer& first, vector<double>& in)
{
    //cout<<"Hidden gradient: ";
    for(uint i=0; i<first.size(); ++i){
	for(uint j=0; j<in.size(); ++j){
	    double tmp=first[i].localGradient() * in[j];
	    //cout<<tmp<<" ";
	    tmp+=first[i].gradient(j);
	    first[i].gradient(j, tmp);
	}
	double tmp=first[i].localGradient();
	//cout<<tmp<<" ";
	tmp+=first[i].gradient(in.size());
	first[i].gradient(in.size(), tmp); //bias
    }
    //cout<<endl;
}

void SummedSquare::gradient(Layer& curr, Layer& prev)
{
    for(uint i=0; i<curr.size(); ++i){
	for(uint j=0; j<prev.size(); ++j)
	    curr[i].gradient(j, curr[i].localGradient() * prev[j].activation());
	curr[i].gradient(prev.size(), curr[i].localGradient()); //bias
    }
}

void SummedSquare::gradientBatch(Layer& curr, Layer& prev)
{
    //cout<<"Output gradient: ";
    for(uint i=0; i<curr.size(); ++i){
	for(uint j=0; j<prev.size(); ++j){
	    double tmp=curr[i].localGradient() * prev[j].activation();
	    //cout<<tmp<<" ";
	    tmp+=curr[i].gradient(j);
	    curr[i].gradient(j, tmp);
	}
	double tmp=curr[i].localGradient(); //bias
	//cout<<tmp<<" ";
	tmp+=curr[i].gradient(prev.size());
	curr[i].gradient(prev.size(), tmp); //bias
    }
    //cout<<endl;
}

vector<double>& SummedSquare::gradient(Mlp& mlp)
{
    theGradient = vector<double>(0);
    theGradient.reserve(mlp.nWeights()); //Is this needed?

    for(uint i=0; i<mlp.nLayers(); ++i)
	for(uint j=0; j<mlp[i].nNeurons(); ++j){
	    const vector<double>& g = mlp[i][j].gradients();
	    theGradient.insert(theGradient.end(), g.begin(), g.end());
	}

    assert(theGradient.size() == mlp.nWeights());

    //cout<<"Gradients: \n";
    //printWeights(mlp.arch(), theGradient);

    return theGradient;
}

double SummedSquare::outputError(vector<double>& out, vector<double>& dout)
{
    assert(out.size()==dout.size());

    vector<double>::iterator ito = out.begin();
    vector<double>::iterator itd = dout.begin();
    double e = 0;
    for(; ito!=out.end(); ++ito, ++itd)
	e += pow(*itd - *ito,2);
    return e;
}

