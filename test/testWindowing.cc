#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "datatools/Windowing.hh"
#include "parser/Parser.hh"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace DataTools;
using namespace NeuralNetHack;

namespace {

// windowLagged: counts, channel-major lag ordering (lag 0 = current
// period), covariates from the current period, targets from the current
// period, warmup rows consumed.
bool windowing() {
	std::cout << "windowLagged layout: ";
	const uint C = 2, L = 3, P = 1, T = 5;
	auto core = std::make_shared<CoreDataSet>();
	// channel 0: 10,20,30,40,50; channel 1: 1,2,3,4,5; covariate: t/10
	for (uint t = 0; t < T; ++t) {
		std::vector<double> in = {10.0 * (t + 1), 1.0 * (t + 1), t / 10.0};
		std::vector<double> out = {100.0 + t};
		core->addPattern(Pattern(std::to_string(t), in, out));
	}
	DataSet raw;
	raw.coreDataSet(core);

	auto win = windowLagged(raw, C, L, P);
	if (win->size() != T - L + 1) {
		std::cerr << "FAIL (rows " << win->size() << ", expected " << T - L + 1 << ")" << std::endl;
		return false;
	}
	// First windowed row is period t=2: c0 lags [30,20,10], c1 [3,2,1],
	// covariate 0.2, target 102.
	const std::vector<double> expect = {30, 20, 10, 3, 2, 1, 0.2};
	Pattern& p = win->pattern(0);
	for (uint j = 0; j < expect.size(); ++j)
		if (std::fabs(p.input()[j] - expect[j]) > 1e-12) {
			std::cerr << "FAIL (input " << j << " = " << p.input()[j] << ", expected " << expect[j]
			          << ")" << std::endl;
			return false;
		}
	if (p.output()[0] != 102.0) {
		std::cerr << "FAIL (target " << p.output()[0] << ")" << std::endl;
		return false;
	}
	// Last row is period T-1: target 104
	if (win->pattern(win->size() - 1).output()[0] != 104.0) {
		std::cerr << "FAIL (last target)" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

// Parser: comma-separated values and a header line must both work.
bool csvParsing() {
	std::cout << "csv + header parsing: ";
	std::istringstream in("week,spend,sales\n1,100.5,7\n2,200,8\n");
	CoreDataSet core;
	Parser::readDataFile(in, 1, {2}, {3}, {0}, core);
	if (core.size() != 2 || std::fabs(core.pattern(0).input()[0] - 100.5) > 1e-12 ||
	    core.pattern(1).output()[0] != 8.0) {
		std::cerr << "FAIL (rows " << core.size() << ")" << std::endl;
		return false;
	}
	// Semicolons too
	std::istringstream in2("1;3.5;9\n");
	CoreDataSet core2;
	Parser::readDataFile(in2, 0, {2}, {3}, {0}, core2);
	if (core2.size() != 1 || core2.pattern(0).input()[0] != 3.5) {
		std::cerr << "FAIL (semicolon)" << std::endl;
		return false;
	}
	std::cout << "PASS" << std::endl;
	return true;
}

} // namespace

int main() {
	bool allPass = true;
	std::cout << "=== Windowing / CSV Test Suite ===" << std::endl << std::endl;
	allPass &= windowing();
	allPass &= csvParsing();
	std::cout << std::endl << (allPass ? "ALL PASS" : "SOME FAILED") << std::endl;
	return allPass ? 0 : 1;
}
