#ifndef __BootstrapSampler_hh__
#define __BootstrapSampler_hh__

#include "Sampler.hh"

#include <utility>

namespace DataTools
{
	class BootstrapSampler:public Sampler
	{
		public:
			BootstrapSampler(DataSet& data, const uint numSplits);
			BootstrapSampler(const BootstrapSampler& me);
			virtual ~BootstrapSampler();
			using Sampler::operator=;
			BootstrapSampler& operator=(const BootstrapSampler& me);

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
