#include "Gof.hh"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>

using namespace EvalTools;
using std::pair;
using std::vector;

// PUBLIC

Gof::Gof(uint nb) : numBins(nb) {}

double Gof::goodnessOfFit(const vector<double>& output, const vector<uint>& target) {
	assert(output.size() == target.size());
	uint numInBin = output.size() / numBins;
	uint leftOvers = output.size() % numBins;
	double chi2 = 0;
	vector<pair<double, uint>> sortedVec = doSort(output, target);
	vector<uint> n;

	for (uint i = 0; i < numBins; ++i)
		n.push_back(numInBin);
	for (uint i = 0; i < leftOvers; ++i)
		n[i]++;
	uint id = 0;
	for (uint i = 0; i < numBins; ++i) {
		double sum1 = 0, sum2 = 0;
		for (uint j = 0; j < n[i]; ++j) {
			// std::cerr<<std::endl<<id<<"\t"<<sortedVec[id].second<<"\t"<<sortedVec[id].first;
			sum1 += sortedVec[id].second;
			sum2 += sortedVec[id++].first;
		}
		sum2 /= (double)n[i];

		double tmp = (sum1 - n[i] * sum2) * (sum1 - n[i] * sum2) / (n[i] * sum2 * (1.0 - sum2));
		chi2 += tmp;

		/*
		   std::cout<<std::endl<<tmp<<" = "<<"("<<sum1<<" - "<<n[i]<<" * "<<sum2<<") * ("<<
		   sum1<<" - "<<n[i]<<" * "<<sum2<<") / ("<<n[i]<<" * "<<sum2<<" *(1.0 - "<<sum2<<"))";
		*/
	}

	return chi2;
}

// PRIVATE

vector<pair<double, uint>> Gof::doSort(const vector<double>& output, const vector<uint>& target) {
	assert(output.size() == target.size());
	vector<pair<double, uint>> sortedVec;

	for (uint i = 0; i < output.size(); ++i) {
		sortedVec.push_back(std::make_pair(output[i], target[i]));
	}
	std::sort(sortedVec.begin(), sortedVec.end());
	return sortedVec;
}
