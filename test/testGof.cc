/*$Id: testGof.cc 1551 2006-05-02 11:56:08Z michael $*/

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

#include <evaltools/Gof.hh>

#include <vector>
#include <cmath>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <ctime>

void buildDeterministic(std::vector<double>& output, std::vector<uint>& target)
{
	using namespace EvalTools;
	using namespace std;

	output.clear();
	target.clear();

	for(uint i=0; i<200; ++i){
		if(i % 2 == 0){
			output.push_back(0.001);
			target.push_back((uint)0);
		}else{
			output.push_back(0.99);
			target.push_back((uint)1);
		}
	}
}

void buildRandom(std::vector<double>& output, std::vector<uint>& target)
{
	using namespace EvalTools;
	using namespace std;

	output.clear();
	target.clear();

	for(uint i=0; i<200; ++i){
		double x = (double)rand()/RAND_MAX;
		output.push_back(x);
		target.push_back((uint)round(x));
	}
}

int testGof()
{
	using namespace EvalTools;
	using namespace std;

	vector<double> output;
	vector<uint> target;

	buildDeterministic(output, target);

	/*
	cout<<"OUTPUT: ";
	copy(output.begin(), output.end(), ostream_iterator<double>(cout, " "));
	cout<<endl<<"TARGET: ";
	copy(target.begin(), target.end(), ostream_iterator<uint>(cout, " "));
	*/
	Gof* gof = new Gof(10);
	double res = gof->goodnessOfFit(output, target);
	delete gof;
	//cout<<endl<<"Chi2HL: "<<res<<endl;
	return 0;
}

int main(int argc, char* argv[])
{
	srand48(time(0));

	return testGof();
}
