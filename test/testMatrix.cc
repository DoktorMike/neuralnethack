/*$Id: testMatrix.cc 1546 2006-04-18 08:38:01Z michael $*/

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

#include "matrixtools/MatrixTools.hh"
#include <numeric>
#include <ext/numeric>
#include <functional>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>

using namespace MatrixTools;
using namespace std;

int testMatrix()
{
	bool ok = true;
	vector<double> v1(4);
	vector<double> v2(4);

	iota(v1.begin(), v1.end(), 7);
	iota(v2.begin(), v2.end(), 1);
	add(v1, v2);
	if(v1[0] != 8 || v1[1] != 10 || v1[2] != 12 || v1[3] != 14) ok = false;
	//copy(v1.begin(), v1.end(), ostream_iterator<double>(cout, " "));
	
	mul(v1, 0.5);
	//copy(v1.begin(), v1.end(), ostream_iterator<double>(cout, " "));
	if(v1[0] != 4 || v1[1] != 5 || v1[2] != 6 || v1[3] != 7) ok = false;
	mul(v1, 2, v2);
	if(v2[0] != 8 || v2[1] != 10 || v2[2] != 12 || v2[3] != 14) ok = false;
	if(innerProduct(v1, v2) != 252) ok = false;

	v1.clear(); v2.clear();
	double a[] = {0.034, 0.23, 3.45};
	v1.assign(a, a+3);
	mul(v1, 0.25);
	if(v1[0] != 0.0085 || v1[1] != 0.0575 || v1[2] != 0.8625) ok = false;

	return (ok == true) ? 0 : 1;
}

int main(int argc, char* argv[])
{
	srand(time(0));

	return testMatrix();
}
