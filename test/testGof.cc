#include <Random.hh>
#include <evaltools/Gof.hh>

#include <vector>
#include <iostream>
#include <ctime>

using namespace EvalTools;
using std::vector;

// Perfect predictions: output ~= target. HL chi-squared should be tiny.
static void buildGoodFit(vector<double>& output, vector<uint>& target) {
	output.clear();
	target.clear();
	for (uint i = 0; i < 200; ++i) {
		if (i % 2 == 0) {
			output.push_back(0.001);
			target.push_back(0);
		} else {
			output.push_back(0.99);
			target.push_back(1);
		}
	}
}

// Inverted predictions: target is the opposite of what the output suggests.
// HL chi-squared should be huge.
static void buildBadFit(vector<double>& output, vector<uint>& target) {
	output.clear();
	target.clear();
	for (uint i = 0; i < 200; ++i) {
		if (i % 2 == 0) {
			output.push_back(0.001);
			target.push_back(1);
		} else {
			output.push_back(0.99);
			target.push_back(0);
		}
	}
}

int main() {
	nnh::rand::seed(time(0));

	vector<double> output;
	vector<uint> target;

	Gof gof(10);

	buildGoodFit(output, target);
	double chi2_good = gof.goodnessOfFit(output, target);

	buildBadFit(output, target);
	double chi2_bad = gof.goodnessOfFit(output, target);

	std::cout << "Chi2HL good fit: " << chi2_good << std::endl;
	std::cout << "Chi2HL bad fit:  " << chi2_bad << std::endl;

	bool pass = true;
	// A perfect calibrated model lands well under any reasonable threshold.
	if (!(chi2_good < 5.0)) {
		std::cerr << "FAIL: expected chi2_good < 5, got " << chi2_good << std::endl;
		pass = false;
	}
	// An anti-calibrated model blows up far above the good-fit value.
	if (!(chi2_bad > 100.0)) {
		std::cerr << "FAIL: expected chi2_bad > 100, got " << chi2_bad << std::endl;
		pass = false;
	}
	if (!(chi2_bad > 10.0 * chi2_good)) {
		std::cerr << "FAIL: bad fit should dwarf good fit, got " << chi2_bad << " vs " << chi2_good
		          << std::endl;
		pass = false;
	}

	return pass ? 0 : 1;
}
