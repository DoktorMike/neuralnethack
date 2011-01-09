#ifndef __CrossValidator_hh__
#define __CrossValidator_hh__

#include "ModelEstimator.hh"

#include <utility>
#include <ostream>

namespace NeuralNetHack
{
	class CrossValidator:public ModelEstimator
	{
		public:
			CrossValidator();
			CrossValidator(const CrossValidator& me);
			virtual ~CrossValidator();
			CrossValidator& operator=(const CrossValidator& me);

			std::pair<double, double>* estimateModel();

			/**Print the output and target for each data point in the DataSet.
			 * \param os the output stream to write to.
			 * \todo The function shouldn't assume single output.
			 */
			void printOutputTargetList(std::ostream& os);

			uint numRuns();
			void numRuns(uint n);
			uint numParts();
			void numParts(uint k);

		private:
			uint n;
			uint k;
	};
}
#endif
