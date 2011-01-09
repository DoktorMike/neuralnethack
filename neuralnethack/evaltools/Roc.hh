/*$Id: Roc.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __Roc_hh__
#define __Roc_hh__

#include "EvalTools.hh"

#include <vector>
#include <utility>
#include <ostream>

namespace EvalTools{

	class Evaluator;

	/**A class representing the creation and evaluation of an ROC.
	 * The ROC consists of FPF(1-specificity) plotted against TPF(sensitivity) 
	 * This class only accepts data sets that have a single output featuring 
	 * two classes. It can calculate the AUC(Area Under Curve) from the ROC in 
	 * two ways. (i) Using Trapezoidal rule on the generated FPF,TPF data points.
	 * (ii) Using Wilcoxon-Mann-Whitney rank statistics.
	 * \todo Both wmw methods are sensitive to ties i.e. when an output from
	 * the negative class is equal to an output from the positive class. The
	 * problem is how to rank them since they give different rank if permuted.
	 * The normal wmw tries to compensate a bit using 0.5 when an output from
	 * negative class has the same value as one from the positive class. 
	 */
	class Roc{

		public:
			/**Basic constructor. */
			Roc();

			/**Copy constructor.
			 * \param roc the object to copy from.
			 */
			Roc(const Roc& roc);

			/**Basic destructor. */
			virtual ~Roc();

			/**Assignment operator.
			 * \param roc the object to assign from.
			 * \return the object assigned to.
			 */
			Roc& operator=(const Roc& roc);

			/**Fetch the ROC in pairs of FPF and TPF.
			 * \return the ROC.
			 */
			std::vector< std::pair<double,double> >& roc();

			/**Return the area under the ROC curve.
			 * \return the AUC for the ROC curve.
			 */
			double auc();

			/**Estimate the AUC using the Wilcoxon-Mann-Whitney rank
			 * statistics. This is exactely the AUC when no ties are present.
			 * This compares all pairs in the positive output list and the
			 * negative output list.
			 * \f[AUC=\frac{\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}I(x_i, y_j)}{m\cdot n}\f]
			 * \f[I(x_i, y_j)=\left\{\begin{array}{cc}
			 * 1, & x_i>y_j\\
			 * 0, & otherwise
			 * \end{array}\right.\f]
			 * \todo Make this less sensitive to ties.
			 * \param out the data.
			 * \param dout the target data.
			 * \return the AUC.
			 */
			double calcAucWmw(std::vector<double>& out, std::vector<uint>& dout);

			/**Estimate the AUC using the Wilcoxon-Mann-Whitney rank
			 * statistics. This is exactely the AUC when no ties are present.
			 * This is another way of calculating the WMW statistics using the
			 * wilcoxon rank sum.
			 * \f[AUC=\frac{1}{m\cdot
			 * n}\left(\sum_{i=0}^{m-1}r_i-\frac{m(m-1)}{2}\right)\f]
			 * where \f$m\f$ is the number of positive samples and
			 * \f$n\f$ is the number of negative samples in the data set. \f$r_i\f$ is
			 * the rank for the outputs generated from the positive sample set.
			 * \todo Make this less sensitive to ties.
			 * \param out the data.
			 * \param dout the target data.
			 * \return the AUC.
			 */
			double calcAucWmwFast(std::vector<double>& out, std::vector<uint>& dout);

			/**Estimate the AUC using the trapezoidal rule. This
			 * is a brute force way of integrating the area under the ROC plot.
			 * \f[AUC=\sum_{i=1}^{n-1}(x_i-x_{i-1})\cdot 0.5\cdot(y_i+y_{i-1})\f] where
			 * \f$n\f$ is the number of data points.
			 * \param out the data.
			 * \param dout the target data.
			 * \return the AUC.
			 */
			double calcAucTrapezoidal(std::vector<double>& out, std::vector<uint>& dout);

			/**Create a FPF,TPF pair for each value in out. 
			 * This basically generate a (1-specificity), sensitivity 
			 * pair for each output.
			 * \param out the output to evaluate, which also serves as the
			 * cuts.
			 * \param dout the target output.
			 */
			void calcRoc(std::vector<double>& out, std::vector<uint>& dout);

			/**Print the ROC data to stdout. 
			 * \param os the stream to output to.
			 */
			void print(std::ostream& os);

		private:
			/**The FPF and TPF for each cut. The first double is the FPF
			 * and the second is the TPF.
			 */
			std::vector< std::pair<double,double> > theRoc;

			void printVector(std::vector<uint>& vec);

			void printVector(std::vector<double>& vec);

			/**The area under the ROC curve. */
			double theAuc;

			/**The evaluator. */
			Evaluator* theEval;
	};

	//INLINES

	inline double Roc::auc() { return theAuc; }
	inline std::vector< std::pair<double,double> >& Roc::roc() { return theRoc; }

}
#endif
