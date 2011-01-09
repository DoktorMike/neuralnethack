/*$Id: MatrixTools.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "MatrixTools.hh"
#include <cassert>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace MatrixTools;
using namespace std;

typedef vector<double>::iterator ditor;

//VECTOR FUNCTIONS------------------------------------------------------------//

void MatrixTools::add(vector<double>& v1, vector<double>& v2)
{
	assert(v1.size()==v2.size());
	transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), plus<double>());
}

void MatrixTools::add(vector<double>& v1, vector<double>& v2, 
		vector<double>& res)
{
	assert(v1.size()==v2.size() && v1.size()==res.size());
	transform(v1.begin(), v1.end(), v2.begin(), res.begin(), plus<double>());
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2)
{
	assert(v1.size()==v2.size());
	transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), minus<double>());
}

void MatrixTools::sub(vector<double>& v1, vector<double>& v2, 
		vector<double>& res)
{
	assert(v1.size()==v2.size() && v1.size()==res.size());
	transform(v1.begin(), v1.end(), v2.begin(), res.begin(), minus<double>());
}

template<class T> struct scalarMul : public unary_function<T, void>
{
	scalarMul(T s) : scalar(s) {}
	T operator() (T x) { return x * scalar; }
	T scalar;
};

void MatrixTools::mul(vector<double>& vec, double scalar)
{ transform(vec.begin(), vec.end(), vec.begin(), scalarMul<double>(scalar)); }

void MatrixTools::mul(vector<double>& vec, double scalar, vector<double>& res)
{
	assert(vec.size() == res.size());
	transform(vec.begin(), vec.end(), res.begin(), scalarMul<double>(scalar));
}

void MatrixTools::div(vector<double>& vec, double scalar)
{mul(vec,1.0/scalar);}

void MatrixTools::div(vector<double>& vec, double scalar, vector<double>& res)
{mul(vec,1.0/scalar,res);}

double MatrixTools::innerProduct(vector<double>& v1, vector<double>& v2)
{ return inner_product(v1.begin(), v1.end(), v2.begin(), (double)0); }

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
		copy(m[i].begin(), m[i].end(), ostream_iterator<double>(cout, " "));
		cout<<endl;
	}
}


