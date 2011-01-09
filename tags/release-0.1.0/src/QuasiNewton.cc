#include "QuasiNewton.hh"
#include "Error.hh"
#include "MatrixTools.hh"

using namespace NetHack;
using namespace MatrixTools;

QuasiNewton::QuasiNewton(string e, double te, uint bs)
    :Trainer(e, te), theBatchSize(bs), G(0,vector<double>(0)), w(0), wPrev(0), 
    g(0), gPrev(0), p(0), v(0), u(0) {}

QuasiNewton::~QuasiNewton(){}

void QuasiNewton::train(Committee& committee, DataSet& dset, uint epochs)
{
    for(uint i=0; i<committee.size(); ++i){
	Mlp& mlp=committee[i];
	uint n=mlp.nWeights();
	theWeightUpdate = vector<double>(n,0);
	G     = vector< vector<double> >(n,vector<double>(n,0));
	w     = vector<double>(n,0);
	wPrev = vector<double>(n,0);
	p     = vector<double>(n,0);
	g     = vector<double>(n,0);
	gPrev = vector<double>(n,0);
	v     = vector<double>(n,0);
	u     = vector<double>(n,0);
	train(mlp, dset, epochs);
    }
}

uint QuasiNewton::batchSize(){return theBatchSize;}

void QuasiNewton::batchSize(uint bs){theBatchSize=bs;}

//PRIVATE--------------------------------------------------------------------//

QuasiNewton::QuasiNewton(const QuasiNewton& qn)
	:Trainer(qn.theError->type(), qn.theTrainingError)
{*this=qn;}

QuasiNewton& QuasiNewton::operator=(const QuasiNewton& qn)
{
    if(this!=&qn){
    }
    return *this;
}

void QuasiNewton::train(Mlp& mlp, DataSet& dset, uint epochs)
{
    double err = INT_MAX;
    uint cntr = epochs; 
    w = mlp.weights();
    G = identity(w.size());

    while(cntr-- && err > theTrainingError){
	err = train(mlp, dset);
	mlp.weights(w);
	//if(cntr % 10 == 0)
	  cout<<"ERROR: "<<err<<" IN EPOCH "<<epochs-cntr<<endl;
    }
    //cout<<"The error in epoch "<<epochs-cntr-1<<" is "<<err<<endl;
}

double QuasiNewton::train(Mlp& mlp, DataSet& dset)
{
    double err=buildInvHessEstim(mlp, dset);
    wPrev=w;
    double alfa=0.1;
    vector< vector<double> > aG=mul(G,alfa);
    vector<double> term2=mul(aG,g);
    add(w,term2,w);

    return 0.5*(err/theBatchSize);
}

double QuasiNewton::buildInvHessEstim(Mlp& mlp, DataSet& dset)
{
    cout<<"w(t+1):\n";
    print(w);
    cout<<"w(t):\n";
    print(wPrev);
    double err=buildG(mlp, dset);
    cout<<"g(t+1):\n";
    print(g);
    cout<<"g(t):\n";
    print(gPrev);
    buildP();
    cout<<"p:\n";
    print(p);
    buildV();
    cout<<"v:\n";
    print(v);
    buildU();
    cout<<"u:\n";
    print(u);
    
    vector< vector<double> > term1=G; //G
    
    vector< vector<double> > term2=G; //pp/pv
    outerProduct(p,p,term2);
    div(term2,innerProduct(p,v), term2);
    
    vector< vector<double> > term3=G; //((Gv)vG)/vGv
    vector<double> Gv=v;
    vector<double> vG=v;
    mul(G,v,Gv);
    mul(v,G,vG);
    outerProduct(Gv,vG,term3);
    double vGv=innerProduct(vG,v);
    div(term3,vGv,term3);
    
    vector< vector<double> > term4=G; //(vGv)uu
    outerProduct(u,u,term4);
    mul(term4,vGv,term4);

    add(term1,term2,G);
    sub(G,term3,G);
    add(G,term4,G);

    cout<<"G:\n";
    print(G);
    
    return err;
}

double QuasiNewton::buildG(Mlp& mlp, DataSet& dset)
{
    gPrev=g;
    double err = theError->gradient(mlp, dset, theBatchSize);
    g=mlp.gradients();
    return err;
}

void QuasiNewton::buildP()
{
    sub(w,wPrev,p);
}

void QuasiNewton::buildV()
{
    sub(g,gPrev,v);
}

void QuasiNewton::buildU()
{
    vector<double> term1=p; 
    div(term1,innerProduct(p,v));

    vector<double> term2=v;
    mul(G,v,term2);
    vector<double> tmp=v;
    mul(v,G,tmp);
    div(term2,innerProduct(v,tmp));

    sub(term1,term2,u);
}
