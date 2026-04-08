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
