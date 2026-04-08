#ifndef __DummySampler_hh__
#define __DummySampler_hh__

#include "Sampler.hh"

#include <utility>

namespace DataTools
{
	class DummySampler:public Sampler
	{
		public:
			DummySampler(DataSet& data, const uint numSplits);
			DummySampler(const DummySampler& me);
			virtual ~DummySampler();
			using Sampler::operator=;
			DummySampler& operator=(const DummySampler& me);

			std::pair<DataSet, DataSet>* next();
			bool hasNext() const;
			uint howMany() const;
			void reset();

			uint numRuns() const;
			void numRuns(uint n);

		private:
			/**The number of independent bootstraps. */
			uint n;

			/**The index to which split we are currently at. */
			uint index;
	};
}
#endif
