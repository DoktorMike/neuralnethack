#ifndef __Evaluator_hh__
#define __Evaluator_hh__

#include "EvalTools.hh"

#define NEG	0
#define POS	1

namespace EvalTools{
	
	/**A class used to evaluate output from a classifier.
	 * The classifier is evaluated using sensitivity(TPF) and specificity(TNF). 
	 * Currently it is limited to only two classes using local encoding.
	 * Distributed encoding is not supported in this class.
	 * \f[TPF=\frac{Number~of~true~positive~decisions}
	 * {Number~of~actually~positive~cases}\f]
	 * \f[TNF=\frac{Number~of~true~negative~decisions}
	 * {Number~of~actually~negative~cases}\f]
	 * Also we have the following equivalences:
	 * \f[TPF+FNF=1\f] and \f[TNF+FPF=1\f]
	 */
	class Evaluator
	{
		public:
			/**Basic constructor. */
			Evaluator();

			/**Copy constructor.
			 * \param eval the object to copy.
			 */
			Evaluator(const Evaluator& eval);

			/**Basic destructor.*/
			virtual ~Evaluator();

			/**Assigment operator.
			 * \param eval the object to copy.
			 */
			Evaluator& operator=(const Evaluator& eval);

			/**Return the True Positive Fraction for the evaluation.
			 * i.e. the sensitivity.
			 */
			double tpf();

			/**Return the False Negative Fraction for the evaluation. */
			double fnf();

			/**Return the True Negative Fraction for the evaluation.
			 * i.e. the specificity.
			 */
			double tnf();

			/**Return the False Positive Fraction for the evaluation.*/
			double fpf();

			/**Return the cut used for the evaluation.*/
			double cut();

			/**Set the cut used for the evaluation.
			 * \param c the cut to set.
			 */
			void cut(double c);

			/**Calculate the number of TP and TN for a classifier.
			 * \param out the output from the classifier.
			 * \param dout the target output for the data points.
			 */
			void evaluate(vector<double>& out, vector<uint>& dout);

			/**Print the sensitivity and specificity.
			 * \param os the stream to output to.
			 */
			void print(ostream& os);

		private:

			/**Put all counters to zero.
			 * In effect all attributes are set to 0.
			 */
			void reset();

			/**Classify each output as POS or NEG depending on the cut.
			 * This uses theCut variable to decide whether the output should
			 * be regarded as POS or NEG.
			 * \param out the output to be classified.
			 * \return the classified outputs.
			 */
			vector<uint> cutOutput(vector<double>& out);

			/**Convert a vector of doubles to a vector of uints.
			 * \param vec the double vector.
			 * \return the converted vector.
			 */
			vector<uint> vectorDoubleToUint(vector<double>& vec);

			/**Calculate the Sensitivity and Specificity for the classifier.
			 * This uses the TruePos and TrueNeg variables to do this.
			 */
			void calcRates();

			/**The True Positive Fraction for the current evaluation.
			 * In other words the specificity.
			 */
			double theTnf;

			/**The True Positive Fraction for the current evaluation.
			 * In other words the sensitivity.
			 */
			double theTpf;

			/**The cut assigning the classifier output to 1 or 0.
			 * This cut decides wheather the classifier output should be classified
			 * as POS or NEG.
			 */
			double theCut;

			/**The number of true positives.*/
			uint nTp;

			/**The number of true negatives.*/
			uint nTn;

			/**The number of positive samples in the data. */
			uint nP;

			/**The number of negative samples in the data. */
			uint nN;
	};
}
#endif
