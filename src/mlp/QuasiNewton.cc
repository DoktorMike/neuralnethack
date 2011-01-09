#include "QuasiNewton.hh"
#include "Error.hh"
#include "matrixtools/MatrixTools.hh"

#include <cmath>
#include <ostream>

#define ITMAX				10
#define CGOLD				0.3819660
#define ZEPS				1.0e-10
#define EPS					1.0e-8
#define GOLD				1.618034
#define GLIMIT				100.0
#define TINY				1.0e-20
#define SIGN(a,b)			((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d)		(a)=(b);(b)=(c);(c)=(d);
#define FMAX(a,b)			(maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
							(maxarg1) : (maxarg2))

using namespace MultiLayerPerceptron;
using namespace DataTools;
using namespace MatrixTools;
using namespace std;

static float maxarg1,maxarg2;

QuasiNewton::QuasiNewton(Mlp& mlp, DataSet& data, Error& error, double te, uint bs):Trainer(mlp, data, error, te, bs)
{}

QuasiNewton::~QuasiNewton(){}

void QuasiNewton::train()
{
	theError->mlp(theMlp);
	theError->dset(theData);
	
	double err = 10;
	double prevErr = 100;
	float alpha=0;
	uint cntr = theNumEpochs; 
	resetVectors();

	while(cntr-- && !hasConverged(err, prevErr)){//err > theTrainingError){
		mul(G, g, vectorTemp1); //Gg used in findAlpha!
		prevErr = err;
		err=findAlpha(alpha); //Determine steplength
		//cout<<"Found alpha="<<alpha<<" at error="<<err<<" previous error "<<prevErr<<endl;
		mul(vectorTemp1,alpha);

		wPrev = w;
		add(w,vectorTemp1,w);
		theMlp->weights(w); //Update the weights.
		gPrev = g;
		theError->gradient(*theMlp, *theData);
		g=theMlp->gradients(); //Update the gradients.

		if(cntr % 20 == 0)
			cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr<<endl;
		updateBfgs(); //Build G(t+1)
	}
	cout<<"ERROR: "<<err<<" IN EPOCH "<<theNumEpochs-cntr<<endl;
}

//PRIVATE--------------------------------------------------------------------//

QuasiNewton::QuasiNewton(const QuasiNewton& qn):Trainer(qn){*this=qn;}

QuasiNewton& QuasiNewton::operator=(const QuasiNewton& qn)
{
	if(this!=&qn){
		Trainer::operator=(qn);
		G = qn.G;
		w = qn.w;
		wPrev = qn.wPrev;
		g = qn.g;
		gPrev = qn.gPrev;
		dw = qn.dw;
		dg = qn.dg;
		u = qn.u;
		matrixTemp1 = qn.matrixTemp1;
		matrixTemp2 = qn.matrixTemp2;
		vectorTemp1 = qn.vectorTemp1;
		vectorTemp2 = qn.vectorTemp2;
	}
	return *this;
}

void QuasiNewton::resetVectors()
{
	uint n	= theMlp->nWeights();
	G		= identity(n);
	w		= theMlp->weights();
	wPrev	= vector<double>(n,0);
	dw		= vector<double>(n,0);
	theError->gradient(*theMlp, *theData);
	g		= theMlp->gradients();
	gPrev	= vector<double>(n,0);
	dg		= vector<double>(n,0);
	u		= vector<double>(n,0);

	//Temporary variables that I don't want to reallocate all the time.
	matrixTemp1 = identity(n);
	matrixTemp2 = identity(n);
	vectorTemp1 = vector<double>(n,0);
	vectorTemp2 = vector<double>(n,0);
}

void QuasiNewton::updateBfgs()
{
	sub(w,wPrev,dw);
	sub(g,gPrev,dg);

	/*
	cout<<"w(t):   "; print(w); cout<<"w(t-1): "; print(wPrev);
	cout<<"g(t):   "; print(g); cout<<"g(t-1): "; print(gPrev);
	*/

	double dwdg = innerProduct(dw, dg);		//dwdg
	vector<double> Gdg=dg;					//Gdg
	mul(G,dg,Gdg);       
	double dgGdg = innerProduct(dg, Gdg);	//dgGdg

	//Term 1
	outerProduct(dw,dw,matrixTemp1);
	div(matrixTemp1, dwdg);

	//Term 2
	outerProduct(Gdg, Gdg, matrixTemp2);
	div(matrixTemp2, dgGdg);

	sub(matrixTemp1, matrixTemp2, matrixTemp1); //matrixTemp1 holds the result.

	//Building u
	div(dw,dwdg,vectorTemp1); 
	div(Gdg,dgGdg,vectorTemp2);
	sub(vectorTemp1, vectorTemp2, vectorTemp1);

	//Term 3
	outerProduct(vectorTemp1, vectorTemp1, matrixTemp2);
	mul(matrixTemp2, dgGdg);

	sub(matrixTemp1, matrixTemp2, matrixTemp1); //matrixTemp1 holds the result.

	add(G, matrixTemp1, G);
}

void QuasiNewton::updateDfp()
{
	sub(w,wPrev,dw);
	sub(g,gPrev,dg);

	/*
	cout<<"w(t):   "; print(w); cout<<"w(t-1): "; print(wPrev);
	cout<<"g(t):   "; print(g); cout<<"g(t-1): "; print(gPrev);
	*/
	
	double dwdg = innerProduct(dw, dg);		//dwdg
	vector<double> Gdg=dg;					//Gdg
	mul(G,dg,Gdg);       
	double dgGdg = innerProduct(dg, Gdg);	//dgGdg

	//matrixTemp1 = G;
	outerProduct(dw,dw,matrixTemp1);
	div(matrixTemp1, dwdg);

	outerProduct(Gdg, Gdg, matrixTemp2);
	div(matrixTemp2, dgGdg);

	add(G,matrixTemp1,G);
	sub(G,matrixTemp2,G);
}

float QuasiNewton::findAlpha(float& alpha)
{
	float ax,bx,cx,fa,fb,fc,tol;
	ax=0.0; bx=-0.001;//alpha < -0.1 ? alpha : -0.1;
	tol=EPS;

	mnbrak(&ax,&bx,&cx,&fa,&fb,&fc);
	/*
	cout<<"Found brackets ax: "<<ax<<" bx: "<<bx<<" cx: "<<cx<<endl;
	cout<<"with Values fa: "<<fa<<" fb: "<<fb<<" fc: "<<fc<<endl;
	alpha = bx;
	return fb;//brent(ax,bx,cx,tol,&alpha,theMlp,dset);
	*/
	return brent(ax,bx,cx,tol,&alpha);
}

void QuasiNewton::mnbrak(float *ax, float *bx, float *cx, 
		float *fa, float *fb, float *fc)
{
	float ulim,u,r,q,fu,dum;
	*fa=err(*ax);					 
	*fb=err(*bx); 
	if (*fb > *fa) {
		SHFT(dum,*ax,*bx,dum);
		SHFT(dum,*fb,*fa,dum);
	}
	*cx=(*bx)+GOLD*(*bx-*ax); 
	*fc=err(*cx);
	while (*fb > *fc) { 
		r=(*bx-*ax)*(*fb-*fc);
		q=(*bx-*cx)*(*fb-*fa);
		u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/(2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
		ulim=(*bx)+GLIMIT*(*cx-*bx);
		if ((*bx-u)*(u-*cx) > 0.0) { 
			fu=err(u);
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
			fu=err(u); 
		} else if ((*cx-u)*(u-ulim) > 0.0) { 
			fu=err(u);
			if (fu < *fc) { 
				SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx));
				SHFT(*fb,*fc,fu,err(u));
			}	   
		} else if ((u-ulim)*(ulim-*cx) >= 0.0) {
			u=ulim;
			fu=err(u);
		} else {
			u=(*cx)+GOLD*(*cx-*bx);
			fu=err(u);
		}
		SHFT(*ax,*bx,*cx,u); 
		SHFT(*fa,*fb,*fc,fu);
	}		   
}	   

float QuasiNewton::brent(float ax, float bx, float cx, float tol,
		float *xmin)

{		   
	int iter;
	float a,b,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
	float e=0.0;
	float d=0.0;
	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx; 
	fw=fv=fx=err(x);
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
		fu=err(u);
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
	//cerr<<"Too many iterations in brent."<<endl;
	*xmin=x;
	return fx;
}										  

float QuasiNewton::err(float alfa)
{
	//mul(G,g,vectorTemp1); //vectorTemp1=G*g;
	mul(vectorTemp1, alfa, vectorTemp2);
	add(vectorTemp2,w,vectorTemp2); //vectorTemp1=w+alfa*G*g;

	theMlp->weights(vectorTemp2); //set weights.
	float err=theError->outputError(*theMlp, *theData);
	theMlp->weights(w); //reset weights.

	return err;
}

bool QuasiNewton::converged()
{
	vector<double>::iterator it;

	double test=0.0;
	for(it=dw.begin(); it!=dw.end(); ++it){
		double tmp=fabs(*it);
		if(tmp>test)
			test=tmp;
	}
	if(test<EPS){
		//cout<<"Weight difference less than tolerance."<<endl;
		return true;
	}

	test=0.0;
	for(it=dg.begin(); it!=dg.end(); ++it){
		double tmp=fabs(*it);
		if(tmp>test)
			test=tmp;
	}
	if(test<EPS){
		//cout<<"Gradient difference less than tolerance."<<endl;
		return true;
	}
	return false;
}
