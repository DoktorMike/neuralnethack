#ifndef __Roc_hh__
#define __Roc_hh__

#include "EvalTools.hh"

namespace EvalTools{

	class Evaluator;

	/**A class representing the creation and evaluation of an ROC.
	 * The ROC consists of FPF(1-specificity) plotted against TPF(sensitivity) 
	 * This class only accepts data sets that have a single output featuring 
	 * two classes. It can calculate the AUC(Area Under Curve) from the ROC in 
	 * two ways. (i) Using Trapezoidal rule on the generated FPF,TPF data points.
	 * \f[AUC=\sum_{i=1}^{n-1}(x_i-x_{i-1})\cdot 0.5\cdot(y_i+y_{i-1})\f] where
	 * \f$n\f$ is the number of data points.
	 * (ii) Using Wilcoxon-Mann-Whitney rank statistics. 
	 * \f[AUC=\frac{1}{m\cdot
	 * n}\left(\sum_{i=0}^{m-1}r_i-\frac{m(m-1)}{2}\right)\f]
	 * where \f$m\f$ is the number of positive samples and
	 * \f$n\f$ is the number of negative samples in the data set. \f$r_i\f$ is
	 * the rank for the outputs generated from the positive sample set.
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
			 */
			Roc& operator=(const Roc& roc);

			/**Fetch the ROC in pairs of FPF and TPF.
			 * \return the ROC.
			 */
			vector< pair<double,double> >& roc();

			/**Return the area under the ROC curve. */
			double auc();

			/**Estimate the AUC using the Wilcoxon-Mann-Whitney rank
			 * statistics. This is a reasonably good estimation of the AUC.
			 * \param out the data.
			 * \param dout the target data.
			 * \return the AUC.
			 */
			double calcAucWmw(vector<double>& out, vector<uint>& dout);

			/**Estimate the AUC using the trapezoidal rule. This
			 * is a brute force way of summing up the area under the ROC plot.
			 * \param out the data.
			 * \param dout the target data.
			 * \return the AUC.
			 */
			double calcAucTrapezoidal(vector<double>& out, vector<uint>& dout);

			/**Create a FPF,TPF pair for each value in out. 
			 * This basically generate a (1-specificity), sensitivity 
			 * pair for each output.
			 * \param out the output to evaluate, which also serves as the
			 * cuts.
			 * \param dout the target output.
			 */
			void calcRoc(vector<double>& out, vector<uint>& dout);

			/**Print the ROC data to stdout. 
			 * \param os the stream to output to.
			 */
			void print(ostream& os);

			void printVector(vector<uint>& vec);
			
			void printVector(vector<double>& vec);
			
		private:
			/**The FPF and TPF for each cut. The first double is the FPF
			 * and the second is the TPF.
			 */
			vector< pair<double,double> > theRoc;

			/**The area under the ROC curve. */
			double theAuc;

			/**The evaluator. */
			Evaluator* theEval;
	};
}
#endif
