/*$Id: Gof.cc 1546 2006-04-18 08:38:01Z michael $*/

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

#include "Gof.hh"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>

using namespace EvalTools;
using std::vector;
using std::pair;

//PUBLIC

Gof::Gof(uint nb):numBins(nb){}

double Gof::goodnessOfFit(const vector<double>& output, const vector<uint>& target)
{
	assert(output.size() == target.size()); 
	uint numInBin = output.size() / numBins;
	uint leftOvers = output.size() % numBins;
	double chi2 = 0;
	vector< pair<double, uint> > sortedVec = doSort(output, target);
	vector<uint> n;

	for(uint i=0; i<numBins; ++i) n.push_back(numInBin);
	for(uint i=0; i<leftOvers; ++i) n[i]++;
	uint id = 0;
	for(uint i=0; i<numBins; ++i){
		double sum1=0, sum2=0;
		for(uint j=0; j<n[i]; ++j){
			//std::cerr<<std::endl<<id<<"\t"<<sortedVec[id].second<<"\t"<<sortedVec[id].first;
			sum1 += sortedVec[id].second;
			sum2 += sortedVec[id++].first;
		}
		sum2 /= (double)n[i];

		double tmp = (sum1 - n[i] * sum2) * (sum1 - n[i] * sum2) / 
			(n[i] * sum2 *(1.0 - sum2));
		chi2 += tmp;

		/*
		   std::cout<<std::endl<<tmp<<" = "<<"("<<sum1<<" - "<<n[i]<<" * "<<sum2<<") * ("<<
		   sum1<<" - "<<n[i]<<" * "<<sum2<<") / ("<<n[i]<<" * "<<sum2<<" *(1.0 - "<<sum2<<"))";
		*/
	}

	return chi2;
}

//PRIVATE

vector< pair<double, uint> > Gof::doSort(const vector<double>& output, const vector<uint>& target)
{
	assert(output.size() == target.size());
	vector< pair<double, uint> > sortedVec;

	for(uint i=0; i<output.size(); ++i){
		sortedVec.push_back(std::make_pair(output[i], target[i]));
	}
	std::sort(sortedVec.begin(), sortedVec.end());
	return sortedVec;
}

