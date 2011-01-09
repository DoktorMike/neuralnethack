/*$Id: EvalTools.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __EvalTools_hh__
#define __EvalTools_hh__

#include "datatools/DataSet.hh"
#include "Ensemble.hh"

#include <vector>

/**This namespace encloses a bunch of functions needed to 
 * evaluate the performance of a classifier.
 */
namespace EvalTools
{
	namespace ErrorMeasures
	{
		/**Measures the CrossEntropyError for an entire DataSet. 
		 * This is also known as the kullback leibler error measure. 
		 * This measure is intended to be used with classification 
		 * problems for two class single output problems. In the
		 * special case of one output and two classes the following will be used:
		 * \f[E=-\frac{1}{N}\sum_{n}\left(d_n\ln y_n + (1-d_n)\ln (1-y_n)\right)\f]
		 * Otherwise we use:
		 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
		 * \param committee the ensemble of Mlp to estimate the error for.
		 * \param data the DataSet to use for error measure.
		 * \return the cross entropy error.
		 */
		double crossEntropy(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

		/**Measures the CrossEntropyError for one data point.
		 * This is also known as the kullback leibler error measure. 
		 * This measure is intended to be used with classification 
		 * problems for two class single output problems. In the
		 * special case of one output and two classes the following will be used:
		 * \f[E=-\frac{1}{N}\sum_{n}\left(d_n\ln y_n + (1-d_n)\ln (1-y_n)\right)\f]
		 * Otherwise we use:
		 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
		 * \param output the output vector of the classifier.
		 * \param target the target.
		 * \return the cross entropy error.
		 */
		double crossEntropy(const std::vector<double>& output, const std::vector<double>& target);

		/**Measures the SummedSquareError for an entire DataSet. 
		 * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
		 * \param committee the ensemble of Mlp to estimate the error for.
		 * \param data the DataSet to use for error measure.
		 * \return the summed square error.
		 */
		double summedSquare(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

		/**Measures the SummedSquareError for an entire DataSet. 
		 * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
		 * \param output the output vector of the classifier.
		 * \param target the target.
		 * \return the summed square error.
		 */
		double summedSquare(const std::vector<double>& output, const std::vector<double>& target);

		/**Measures the AUC(Area Under Curve) for the ROC(Receicer Operating
		 * Characteristics).
		 * \param committee the ensemble of Mlp to estimate the error for.
		 * \param data the DataSet to use for error measure.
		 * \return the area under curve.
		 */
		double auc(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

		/**Measures the Hosmer Lemeshow goodness of fit statistics.
		 * \f[\chi ^2 = \sum_{j=1}^{G}\frac{(o_j - n_j \bar{\pi}_j)^2}{ n_j \bar{\pi}_j (1 - \bar{\pi}_j) }\f]
		 * Where \f$o_j\f$ is the number of observed positives in bin j, and 
		 * \f$\bar{\pi}_j\f$ is the mean average predicted value in bin j. G is
		 * the number of bins meanwhile \f$n_j\f$ is the number of samples in the
		 * bin.This test statistics follow the chi square statistics with a (G-2)
		 * deegree of freedom.
		 * \param committee the ensemble of Mlp to estimate the error for.
		 * \param data the DataSet to use for error measure.
		 * \return the hosmer lemeshow statistics.
		 */
		double gof(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

		void buildOutputTargetVectors(NeuralNetHack::Ensemble& committee, 
				DataTools::DataSet& data, std::vector<double>& output, 
				std::vector<uint>& target);

		void buildOutputTargetVectors(NeuralNetHack::Ensemble& committee, 
				DataTools::DataSet& data, std::vector< std::vector<double> >& output, 
				std::vector< std::vector<double> >& target);
	}
}

#endif
