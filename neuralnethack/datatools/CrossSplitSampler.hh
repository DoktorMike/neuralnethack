#ifndef __CrossSplitSampler_hh__
#define __CrossSplitSampler_hh__

#include "Sampler.hh"

#include <utility>
#include <ostream>

namespace DataTools
{
	/**Represents the Cross Validation sampling method. The data is split
	 * evenly into K bins. Eeach bin represents a validation set. The training
	 * set is created by combining the K-1 other bins. This is then done
	 * during N independent runs.
	 * \sa Sampler, Bootstrapper.
	 */
	class CrossSplitSampler:public Sampler
	{
		public:
			CrossSplitSampler(DataSet& data, const uint numSplits, const uint numParts);
			CrossSplitSampler(const CrossSplitSampler& cv);
			virtual ~CrossSplitSampler();
			using Sampler::operator=;
			CrossSplitSampler& operator=(const CrossSplitSampler& cv);

			/**\todo Fix the k=1 issue. Abort or empty validation set?*/
			std::pair<DataSet, DataSet>* next();
			bool hasNext() const;
			uint howMany() const;
			void reset();

			uint numRuns() const;
			void numRuns(const uint n);
			uint numParts() const;
			void numParts(const uint k);

		private:
			/**The number of independent K-fold cross validation splits. */
			uint n;

			/**The number parts to split the data into. */
			uint k;

			/**The index to which split we are currently at. */
			uint index;

			/**The index of which independent run we are at. */
			uint runCntr;
	};
}

inline uint DataTools::CrossSplitSampler::numRuns() const {return n;}

inline void DataTools::CrossSplitSampler::numRuns(const uint n){this->n = n;}

inline uint DataTools::CrossSplitSampler::numParts() const {return k;}

inline void DataTools::CrossSplitSampler::numParts(const uint k){this->k = k;}

#endif
