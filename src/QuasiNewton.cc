#include "QuasiNewton.hh"
#include "Error.hh"
#include "MatrixTools.hh"


#define ITMAX			100
#define CGOLD			0.3819660
#define ZEPS			1.0e-10
#define EPS			1.0e-8
#define GOLD			1.618034
#define GLIMIT			100.0
#define TINY			1.0e-20
#define WEIGHT_DIFF_TOL		1.0e-8
#define GRADIENT_DIFF_TOL	1.0e-8
#define SIGN(a,b)		((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d)		(a)=(b);(b)=(c);(c)=(d);
#define FMAX(a,b)		(maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
				(maxarg1) : (maxarg2))

using namespace NetHack;
using namespace MatrixTools;

static float maxarg1,maxarg2;

QuasiNewton::QuasiNewton(string e, double te, uint bs)
    :Trainer(e, te), theBatchSize(bs), G(0,vector<double>(0)), w(0), wPrev(0), 
    g(0), gPrev(0), p(0), v(0), u(0) {}

QuasiNewton::~QuasiNewton(){}

void QuasiNewton::train(Committee& committee, DataSet& dset, uint epochs)
{
    for(uint i=0; i<committee.size(); ++i){
	Mlp& mlp=committee[i];
	uint n=mlp.nWeights();
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
    float alpha=0;
    uint cntr = epochs; 
    w = mlp.weights();
    G = identity(w.size());

    while(cntr-- && err > theTrainingError){
	buildInvHessEstim(mlp, dset);
	wPrev=w;
	err=findAlpha(mlp, dset, alpha);
	//cout<<"Found alpha="<<alpha<<" at error="<<err<<endl;
	vector< vector<double> > aG(G);
	mul(aG,alpha);
	vector<double> term2=mul(aG,g);
	add(w,term2,w);

	mlp.weights(w);
	if(converged()){
	    cout<<"Convergence criterion reached."<<endl;
	    break;
	}
	//if(cntr % 10 == 0)
	cout<<"ERROR: "<<err<<" IN EPOCH "<<epochs-cntr<<endl;
    }
    //cout<<"The error in epoch "<<epochs-cntr-1<<" is "<<err<<endl;
}

void QuasiNewton::buildInvHessEstim(Mlp& mlp, DataSet& dset)
{
    //cout<<"w(t+1):\n";
    //print(w);
    //cout<<"w(t):\n";
    //print(wPrev);
    buildG(mlp, dset);
    //cout<<"g(t+1):\n";
    //print(g);
    //cout<<"g(t):\n";
    //print(gPrev);
    buildP();
    //cout<<"p:\n";
    //print(p);
    buildV();
    //cout<<"v:\n";
    //print(v);
    buildU();
    //cout<<"u:\n";
    //print(u);

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

    vector< vector<double> > term4(G); //(vGv)uu
    outerProduct(u,u,term4);
    mul(term4,vGv,term4);

    add(term1,term2,G);
    sub(G,term3,G);
    add(G,term4,G);

    //cout<<"G:\n";
    //print(G);
}

void QuasiNewton::buildG(Mlp& mlp, DataSet& dset)
{
    gPrev=g;
    theError->gradient(mlp, dset, theBatchSize);
    g=mlp.gradients();
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

float QuasiNewton::findAlpha(Mlp& mlp, DataSet& dset, float& alpha)
{
    float ax,bx,cx,fa,fb,fc,tol;
    ax=0.0;
    bx=-0.001;//alpha < -0.1 ? alpha : -0.1;
    tol=EPS;

    mnbrak(&ax,&bx,&cx,&fa,&fb,&fc,mlp,dset);
    //cout<<"Found brackets ax: "<<ax<<" bx: "<<bx<<" cx: "<<cx<<endl;
    //cout<<"with Values fa: "<<fa<<" fb: "<<fb<<" fc: "<<fc<<endl;
    return brent(ax,bx,cx,tol,&alpha,mlp,dset);
}

void QuasiNewton::mnbrak(float *ax, float *bx, float *cx, 
	float *fa, float *fb, float *fc, Mlp& mlp, DataSet& dset)
{
    float ulim,u,r,q,fu,dum;
    *fa=err(mlp,dset,*ax);                     
    *fb=err(mlp,dset,*bx); 
    if (*fb > *fa) {
	SHFT(dum,*ax,*bx,dum);
	SHFT(dum,*fb,*fa,dum);
    }
    *cx=(*bx)+GOLD*(*bx-*ax); 
    *fc=err(mlp,dset,*cx);
    while (*fb > *fc) { 
	r=(*bx-*ax)*(*fb-*fc);
	q=(*bx-*cx)*(*fb-*fa);
	u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/(2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
	ulim=(*bx)+GLIMIT*(*cx-*bx);
	if ((*bx-u)*(u-*cx) > 0.0) { 
	    fu=err(mlp,dset,u);
	    if (fu < *fc) {
		*ax=(*bx);
		*bx=u;
		*fa=(*fb);
		*fb=fu;
		return;
	    } else if (fu > *fb) {
		*cx=u;
		*fc=fu;
		return;
	    }
	    u=(*cx)+GOLD*(*cx-*bx);
	    fu=err(mlp,dset,u); 
	} else if ((*cx-u)*(u-ulim) > 0.0) { 
	    fu=err(mlp,dset,u);
	    if (fu < *fc) { 
		SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx));
		SHFT(*fb,*fc,fu,err(mlp,dset,u));
	    }       
	} else if ((u-ulim)*(ulim-*cx) >= 0.0) {
	    u=ulim;
	    fu=err(mlp,dset,u);
	} else {
	    u=(*cx)+GOLD*(*cx-*bx);
	    fu=err(mlp,dset,u);
	}
	SHFT(*ax,*bx,*cx,u); 
	SHFT(*fa,*fb,*fc,fu);
    }           
}       

float QuasiNewton::brent(float ax, float bx, float cx, float tol,
	float *xmin, Mlp& mlp, DataSet& dset)

{           
    int iter;
    float a,b,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
    float e=0.0;
    float d=0.0;
    a=(ax < cx ? ax : cx);
    b=(ax > cx ? ax : cx);
    x=w=v=bx; 
    fw=fv=fx=err(mlp,dset,x);
    for (iter=1;iter<=ITMAX;iter++) {
	xm=0.5*(a+b);
	tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
	if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
	    *xmin=x; 
	    return fx;
	}       
	if (fabs(e) > tol1) {
	    r=(x-w)*(fx-fv);
	    q=(x-v)*(fx-fw);
	    p=(x-v)*q-(x-w)*r;
	    q=2.0*(q-r);
	    if (q > 0.0) p = -p;
	    q=fabs(q);
	    etemp=e;
	    e=d;
	    if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
		d=CGOLD*(e=(x >= xm ? a-x : b-x));
	    else {
		d=p/q;
		    u=x+d;
		if (u-a < tol2 || b-u < tol2)
		    d=SIGN(tol1,xm-x);
	    }
	} else {
	    d=CGOLD*(e=(x >= xm ? a-x : b-x));
	} 
	u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
	fu=err(mlp,dset,u);
	if (fu <= fx) {
	    if (u >= x) a=x; else b=x;
	    SHFT(v,w,x,u) 
		SHFT(fv,fw,fx,fu)
	} else {
	    if (u < x) a=u; else b=u;
	    if (fu <= fw || w == x) {
		v=w;
		w=u;
		fv=fw;
		fw=fu;
	    } else if (fu <= fv || v == x || v == w) {
		v=u;
		fv=fu;
	    }
	}
    }
    cerr<<"Too many iterations in brent."<<endl;
    *xmin=x;
    return fx;
}                                          

float QuasiNewton::err(Mlp& mlp, DataSet& dset, float alfa)
{
    vector<double> term1(w);
    vector< vector<double> > term2(G);
    mul(term2,alfa);
    mul(term2,g,term1); //term1=alfa*G*g;
    add(term1,w); //term1=w+alfa*G*g;
    
    mlp.weights(term1); //set weights.
    float err=theError->outputError(mlp, dset, dset.size());
    mlp.weights(w); //reset weights.

    return err;
}

bool QuasiNewton::converged()
{
   vector<double>::iterator it;
   
   double test=0.0;
   for(it=p.begin(); it!=p.end(); ++it){
       double tmp=fabs(*it);
       if(tmp>test)
	   test=tmp;
   }
   if(test<WEIGHT_DIFF_TOL){
       cout<<"Weight difference less than tolerance."<<endl;
       return true;
   }

   test=0.0;
   for(it=v.begin(); it!=v.end(); ++it){
       double tmp=fabs(*it);
       if(tmp>test)
	   test=tmp;
   }
   if(test<GRADIENT_DIFF_TOL){
       cout<<"Gradient difference less than tolerance."<<endl;
       return true;
   }
   return false;
}

