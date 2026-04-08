#include "MatrixTools.hh"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

#include <cassert>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iterator>

using namespace MatrixTools;
using namespace std;

typedef vector<double>::iterator ditor;

//VECTOR FUNCTIONS------------------------------------------------------------//

void MatrixTools::add(vector<double>& v1, vector<double>& v2)
{
	assert(v1.size()==v2.size());
#ifdef USE_BLAS
	cblas_daxpy(v1.size(), 1.0, v2.data(), 1, v1.data(), 1);
#else
	transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), plus<double>());
#endif
}

void MatrixTools::add(vector<double>& v1, vector<double>& v2,
		vector<double>& res)
{
	assert(v1.size()==v2.size() && v1.size()==res.size());
#ifdef USE_BLAS
	cblas_dcopy(v1.size(), v1.data(), 1, res.data(), 1);
	cblas_daxpy(res.size(), 1.0, v2.data(), 1, res.data(), 1);
#else
	transform(v1.begin(), v1.end(), v2.begin(), res.begin(), plus<double>());
#endif
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2)
{
	assert(v1.size()==v2.size());
#ifdef USE_BLAS
	cblas_daxpy(v1.size(), -1.0, v2.data(), 1, v1.data(), 1);
#else
	transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), minus<double>());
#endif
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2,
		vector<double>& res)
{
	assert(v1.size()==v2.size() && v1.size()==res.size());
#ifdef USE_BLAS
	cblas_dcopy(v1.size(), v1.data(), 1, res.data(), 1);
	cblas_daxpy(res.size(), -1.0, v2.data(), 1, res.data(), 1);
#else
	transform(v1.begin(), v1.end(), v2.begin(), res.begin(), minus<double>());
#endif
}

void MatrixTools::mul(vector<double>& vec, double scalar)
{
#ifdef USE_BLAS
	cblas_dscal(vec.size(), scalar, vec.data(), 1);
#else
	const uint n = vec.size();
	double* __restrict__ v = vec.data();
	for(uint i = 0; i < n; ++i)
		v[i] *= scalar;
#endif
}

void MatrixTools::mul(vector<double>& vec, double scalar, vector<double>& res)
{
	assert(vec.size() == res.size());
#ifdef USE_BLAS
	cblas_dcopy(vec.size(), vec.data(), 1, res.data(), 1);
	cblas_dscal(res.size(), scalar, res.data(), 1);
#else
	const uint n = vec.size();
	const double* __restrict__ v = vec.data();
	double* __restrict__ r = res.data();
	for(uint i = 0; i < n; ++i)
		r[i] = v[i] * scalar;
#endif
}

void MatrixTools::div(vector<double>& vec, double scalar)
{mul(vec,1.0/scalar);}

void MatrixTools::div(vector<double>& vec, double scalar, vector<double>& res)
{mul(vec,1.0/scalar,res);}

double MatrixTools::innerProduct(vector<double>& v1, vector<double>& v2)
{
#ifdef USE_BLAS
	return cblas_ddot(v1.size(), v1.data(), 1, v2.data(), 1);
#else
	return inner_product(v1.begin(), v1.end(), v2.begin(), (double)0);
#endif
}

void MatrixTools::outerProduct(vector<double>& v1, vector<double>& v2,
		vector< vector<double> >& res)
{
	assert( (v1.size()==res.size()) && (v2.size()==res[0].size()) );
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
{ copy(v.begin(), v.end(), ostream_iterator<double>(cout, " ")); }

void MatrixTools::print(vector<uint>& v)
{ copy(v.begin(), v.end(), ostream_iterator<uint>(cout, " ")); }

//MATRIX FUNCTIONS------------------------------------------------------------//

vector< vector<double> > MatrixTools::matrix(uint row, uint col)
{return vector< vector<double> >(row,vector<double>(col,0));}

vector< vector<double> > MatrixTools::identity(uint n)
{
	vector< vector<double> > res(n,vector<double>(n,0));
	for(uint i=0; i<n; ++i) res[i][i]=1;
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

void MatrixTools::sub(vector< vector<double> >& m1,
		vector< vector<double> >& m2)
{
	assert(m1.size()==m2.size());
	assert(m1[0].size()==m2[0].size());
	for(uint i=0; i<m1.size(); ++i)
		for(uint j=0; j<m1[i].size(); ++j)
			m1[i][j]=m1[i][j]-m2[i][j];
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

void MatrixTools::mul(vector< vector<double> >& m, vector<double>& v,
		vector<double>& res)
{
	assert(v.size()==res.size());
	assert(m[0].size()==v.size());
	res.assign(res.size(),0);
#ifdef USE_BLAS
	// Row-by-row dot product (matrix rows are not contiguous across rows)
	for(uint i=0; i<m.size(); ++i)
		res[i] = cblas_ddot(m[i].size(), m[i].data(), 1, v.data(), 1);
#else
	for(uint i=0; i<m.size(); ++i)
		for(uint j=0; j<m[i].size(); ++j)
			res[i]+=m[i][j]*v[j];
#endif
}

vector<double> MatrixTools::mul(vector< vector<double> >& m, vector<double>& v)
{
	assert(m[0].size()==v.size());
	vector<double> res(v.size(),0);
#ifdef USE_BLAS
	for(uint i=0; i<m.size(); ++i)
		res[i] = cblas_ddot(m[i].size(), m[i].data(), 1, v.data(), 1);
#else
	for(uint i=0; i<m.size(); ++i)
		for(uint j=0; j<m[i].size(); ++j)
			res[i]+=m[i][j]*v[j];
#endif
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

void MatrixTools::print(vector< vector<double> >& m)
{
	for(uint i=0; i<m.size(); ++i){
		copy(m[i].begin(), m[i].end(), ostream_iterator<double>(cout, " "));
		cout<<endl;
	}
}

