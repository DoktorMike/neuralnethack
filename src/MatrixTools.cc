#include "MatrixTools.hh"
#include <cassert>
#include <iostream>

using namespace MatrixTools;
using namespace std;

//VECTOR FUNCTIONS------------------------------------------------------------//

void MatrixTools::add(vector<double>& v1, vector<double>& v2)
{
    assert(v1.size()==v2.size());
    vector<double>::iterator itv1 = v1.begin();
    vector<double>::iterator itv2 = v2.begin();
    for(; itv1!=v1.end(); ++itv1, ++itv2)
	*itv1 = *itv1 + *itv2;
}

void MatrixTools::add(vector<double>& v1, vector<double>& v2, 
	vector<double>& res)
{
    assert(v1.size()==v2.size() && v1.size()==res.size());
    vector<double>::iterator itv1 = v1.begin();
    vector<double>::iterator itv2 = v2.begin();
    vector<double>::iterator itr = res.begin();
    for(; itr!=res.end(); ++itr, ++itv1, ++itv2)
	*itr = *itv1 + *itv2;
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2)
{
    assert(v1.size()==v2.size());
    vector<double>::iterator itv1 = v1.begin();
    vector<double>::iterator itv2 = v2.begin();
    for(; itv1!=v1.end(); ++itv1, ++itv2)
	*itv1 = *itv1 - *itv2;
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2, 
	vector<double>& res)
{
    assert(v1.size()==v2.size() && v1.size()==res.size());
    vector<double>::iterator itv1 = v1.begin();
    vector<double>::iterator itv2 = v2.begin();
    vector<double>::iterator itr = res.begin();
    for(; itr!=res.end(); ++itr, ++itv1, ++itv2)
	*itr = *itv1 - *itv2;
}

void MatrixTools::mul(vector<double>& vec, double scalar)
{
    vector<double>::iterator itv = vec.begin();
    for(; itv!=vec.end(); ++itv)
	*itv = (*itv) * scalar;
}

void MatrixTools::div(vector<double>& vec, double scalar)
{mul(vec,1.0/scalar);}

double MatrixTools::innerProduct(vector<double>& v1, vector<double>& v2)
{
    assert(v1.size()==v2.size());
    double res=0;
    for(uint i=0; i<v1.size(); ++i)
	res+=v1[i]*v2[i];
    return res;
}

void MatrixTools::outerProduct(vector<double>& v1, vector<double>& v2, 
	vector< vector<double> >& res)
{
    assert(v1.size()==res.size());
    assert(v2.size()==res[0].size());
    for(uint i=0; i<v1.size(); ++i)
	for(uint j=0; j<v2.size(); ++j)
	    res[i][j]=v1[i]*v2[j];
}

vector< vector<double> > MatrixTools::outerProduct(vector<double>& v1,
	vector<double>& v2)
{
    vector< vector<double> > res=matrix(v1.size(),v2.size());
    for(uint i=0; i<v1.size(); ++i)
	for(uint j=0; j<v2.size(); ++j)
	    res[i][j]=v1[i]*v2[j];
    return res;
}

void MatrixTools::print(vector<double>& v)
{
    for(uint i=0; i<v.size(); ++i)
	    cout<<v[i]<<" ";
    cout<<endl;
}

//MATRIX FUNCTIONS------------------------------------------------------------//

vector< vector<double> > MatrixTools::matrix(uint row, uint col)
{return vector< vector<double> >(row,vector<double>(col,0));}

vector< vector<double> > MatrixTools::identity(uint n)
{
    vector< vector<double> > res(n,vector<double>(n,0));
    for(uint i=0; i<n; ++i)
	res[i][i]=1;
    return res;
}

void MatrixTools::add(vector< vector<double> >& m1, 
	vector< vector<double> >& m2)
{
    assert(m1.size()==m2.size());
    assert(m1[0].size()==m2[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    m1[i][j]=m1[i][j]+m2[i][j];
}

void MatrixTools::add(vector< vector<double> >& m1, 
	vector< vector<double> >& m2,
	vector< vector<double> >& res)
{
    assert(m1.size()==res.size());
    assert(m1[0].size()==res[0].size());
    assert(m2.size()==res.size());
    assert(m2[0].size()==res[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    res[i][j]=m1[i][j]+m2[i][j];
}

/*vector< vector<double> > MatrixTools::add(vector< vector<double> >& m1, 
	vector< vector<double> >& m2)
{
    assert(m1[0].size()==m2[0].size());
    assert(m1.size()==m2.size());
    vector< vector<double> > res=matrix(m1.size(),m1[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    res[i][j]=m1[i][j]+m2[i][j];
    return res;
}*/

void MatrixTools::sub(vector< vector<double> >& m1, 
	vector< vector<double> >& m2)
{
    assert(m1.size()==m2.size());
    assert(m1[0].size()==m2[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    m1[i][j]=m1[i][j]+m2[i][j];
}

void MatrixTools::sub(vector< vector<double> >& m1, 
	vector< vector<double> >& m2,
	vector< vector<double> >& res)
{
    assert(m1.size()==res.size());
    assert(m1[0].size()==res[0].size());
    assert(m2.size()==res.size());
    assert(m2[0].size()==res[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    res[i][j]=m1[i][j]-m2[i][j];
}

/*vector< vector<double> > MatrixTools::sub(vector< vector<double> >& m1, 
	vector< vector<double> >& m2)
{
    assert(m1[0].size()==m2[0].size());
    assert(m1.size()==m2.size());
    vector< vector<double> > res=matrix(m1.size(),m1[0].size());
    for(uint i=0; i<m1.size(); ++i)
	for(uint j=0; j<m1[i].size(); ++j)
	    res[i][j]=m1[i][j]-m2[i][j];
    return res;
}*/

void MatrixTools::mul(vector< vector<double> >& m, double scale)
{
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    m[i][j]=m[i][j]*scale;
}

void MatrixTools::mul(vector< vector<double> >& m, double scale,
	vector< vector<double> >& res)
{
    assert(m.size()==res.size());
    assert(m[0].size()==res[0].size());
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i][j]=m[i][j]*scale;
}

/*vector< vector<double> > MatrixTools::mul(vector< vector<double> >& m, 
	double scale)
{
    vector< vector<double> > res=matrix(m.size(), m[0].size());
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i][j]=m[i][j]*scale;
    return res;
}*/

void MatrixTools::mul(vector< vector<double> >& m, vector<double>& v,
	vector<double>& res)
{
    assert(v.size()==res.size());
    assert(m[0].size()==v.size());
    res.assign(res.size(),0);
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i]+=m[i][j]*v[j];
}

vector<double> MatrixTools::mul(vector< vector<double> >& m, vector<double>& v)
{
    assert(m[0].size()==v.size());
    vector<double> res(v.size(),0);
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i]+=m[i][j]*v[j];
    return res;
}

void MatrixTools::mul(vector<double>& v, vector< vector<double> >& m,
	vector<double>& res)
{
    assert(v.size()==res.size());
    assert(m.size()==v.size());
    res.assign(res.size(),0);
    for(uint i=0; i<m[0].size(); ++i)
	for(uint j=0; j<m.size(); ++j)
	    res[i]+=m[j][i]*v[j];
}

vector<double> MatrixTools::mul(vector<double>& v, vector< vector<double> >& m)
{
    assert(m.size()==v.size());
    vector<double> res(v.size(),0);
    for(uint i=0; i<m[0].size(); ++i)
	for(uint j=0; j<m.size(); ++j)
	    res[i]+=m[j][i]*v[j];
    return res;
}

void MatrixTools::div(vector< vector<double> >& m, double scale)
{
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    m[i][j]=m[i][j]/scale;
}

void MatrixTools::div(vector< vector<double> >& m, double scale,
	vector< vector<double> >& res)
{
    assert(m.size()==res.size());
    assert(m[0].size()==res[0].size());
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i][j]=m[i][j]/scale;
}

/*vector< vector<double> > MatrixTools::div(vector< vector<double> >& m,
	double scale)
{
    vector< vector<double> > res=matrix(m.size(), m[0].size());
    for(uint i=0; i<m.size(); ++i)
	for(uint j=0; j<m[i].size(); ++j)
	    res[i][j]=m[i][j]/scale;
    return res;
}*/

void MatrixTools::print(vector< vector<double> >& m)
{
    for(uint i=0; i<m.size(); ++i){
	for(uint j=0; j<m[0].size(); ++j)
	    cout<<m[i][j]<<" ";
	cout<<endl;
    }
}


