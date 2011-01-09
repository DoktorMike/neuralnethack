#ifndef __ErrorMeasures_hh__
#define __ErrorMeasures_hh__

#include "datatools/DataSet.hh"
#include "Committee.hh"

#include <vector>

namespace NeuralNetHack
{
namespace ErrorMeasures
{
	double crossEntropy(Committee& committee, DataTools::DataSet& data);
	double crossEntropy(std::vector<double>& output, std::vector<double>& target);
	double summedSquare(Committee& committee, DataTools::DataSet& data);
	double summedSquare(std::vector<double>& output, std::vector<double>& target);
	double auc(Committee& committee, DataTools::DataSet& data);
	void buildOutputTargetVectors(Committee& committee, 
			DataTools::DataSet& data, std::vector<double>& output, 
			std::vector<uint>& target);
	void buildOutputTargetVectors(Committee& committee, 
			DataTools::DataSet& data, std::vector< std::vector<double> >& output, 
			std::vector< std::vector<double> >& target);
}
}
#endif
