#ifndef __HoldOutSampler_hh__
#define __HoldOutSampler_hh__

#include "Sampler.hh"

#include <utility>

namespace DataTools
{
	class HoldOutSampler:public Sampler
	{
		public:
			HoldOutSampler(DataSet& data, const double rat, const uint numSplits);
			HoldOutSampler(const HoldOutSampler& ho);
			virtual ~HoldOutSampler();
			using Sampler::operator=;
			HoldOutSampler& operator=(const HoldOutSampler& ho);

			std::pair<DataSet, DataSet>* next();
			bool hasNext() const;
			uint howMany() const;

			void reset();

			uint numRuns() const;
			void numRuns(uint n);

		private:
			/**The ratio of the dataset to be used as training. */
			double ratio;

			/**The number of independent hold outs. */
			uint n;

			/**The index to which split we are currently at. */
			uint index;
	};
}
#endif
