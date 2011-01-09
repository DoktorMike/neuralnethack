/*$Id: auc.cc 1641 2007-05-26 11:37:10Z michael $*/

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


#include <neuralnethack/evaltools/Roc.hh>

#include <istream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <iterator>
#include <algorithm>

using std::cout;
using std::cin;
using std::endl;
using std::back_inserter;
using std::istream_iterator;
using std::vector;
using std::string;
using std::istringstream;
using EvalTools::Roc;

int main(int argc, char* argv[])
{
	vector<double> output;
	vector<uint> target;
	string line;
	while(!getline(cin, line, '\n').eof()){
		istringstream ss(line);
		vector<double> input;
		copy(istream_iterator<double>(ss), istream_iterator<double>(), back_inserter(input));
		output.push_back(input[0]);
		target.push_back((uint)input[1]);
	}
	Roc roc;
	double auc1 = roc.calcAucWmw(output, target);
	double auc2 = roc.calcAucTrapezoidal(output, target);
	cout<<"Wilcoxon AUC: "<<auc1<<endl;
	cout<<"Trapezoidal AUC: "<<auc2<<endl;

	return EXIT_SUCCESS;
}
