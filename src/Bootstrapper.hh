#ifndef __Bootstrapper_hh__
#define __Bootstrapper_hh__

#include "ModelEstimator.hh"

#include <utility>

namespace NeuralNetHack
{
	class Bootstrapper:public ModelEstimator
	{
		public:
			Bootstrapper();
			Bootstrapper(const Bootstrapper& me);
			virtual ~Bootstrapper();
			Bootstrapper& operator=(const Bootstrapper& me);

			std::pair<double, double>* estimateModel();

			uint numRuns();
			void numRuns(uint n);

		private:
			uint n;
	};
}
#endif
