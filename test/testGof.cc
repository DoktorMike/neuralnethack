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
