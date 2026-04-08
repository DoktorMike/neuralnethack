#include "OddsRatio.hh"
#include "matrixtools/MatrixTools.hh"

#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>

using namespace NeuralNetHack;
using DataTools::DataSet;
using DataTools::Pattern;
using MultiLayerPerceptron::Layer;
using MultiLayerPerceptron::Mlp;

using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
using std::plus;
using std::vector;

// Effective odds ratio calculations

vector<double> OddsRatio::oddsRatio(Ensemble& committee, DataSet& data) {
	vector<double> oddsrat(data.nInput(), 0);

	for (uint i = 0; i < data.size(); ++i) {
		vector<double> tmp = oddsRatio(committee, data.pattern(i));
		transform(oddsrat.begin(), oddsrat.end(), tmp.begin(), oddsrat.begin(), plus<double>());
	}
	for (vector<double>::iterator it = oddsrat.begin(); it != oddsrat.end(); ++it)
		(*it) /= (double)data.size();
	return oddsrat;
}

template <class T> struct or_calc {
	or_calc(Ensemble& e, std::vector<T>& i) : ens(e), input(i), oddsrat(0) {
		oddsrat.reserve(input.size());
	}
	void operator()(T& x) {
		T inc = 0.1;
		T p0 = ens.propagate(input).front();
		x += inc;
		T p1 = ens.propagate(input).front();
		x -= inc;
		oddsrat.push_back((p1 * (1 - p0)) / (p0 * (1 - p1)));
	}
	Ensemble& ens;
	std::vector<T>& input;
	std::vector<T> oddsrat;
};

vector<double> OddsRatio::oddsRatio(Ensemble& committee, Pattern& pattern) {
	vector<double>& input = pattern.input();
	struct or_calc<double> ret =
	    for_each(input.begin(), input.end(), or_calc<double>(committee, input));
	return ret.oddsrat;
}

// Utilities like printing and stuff

void OddsRatio::print(ostream& os, vector<double>& oddsrat) {
	if (!os) {
		cerr << "OddsRatio::print: Problem with output stream." << endl;
		return;
	}

	os << "Input\tOR" << endl;
	for (uint i = 0; i < oddsrat.size(); ++i)
		os << i << "\t" << oddsrat[i] << endl;
}
