#ifndef __CrossSplitter_hh__
#define __CrossSplitter_hh__

#include "EnsembleBuilder.hh"
#include "Committee.hh"

namespace NeuralNetHack
{
	class CrossSplitter: public EnsembleBuilder
	{
		public:
			CrossSplitter();
			CrossSplitter(const CrossSplitter& cs);
			~CrossSplitter();
			CrossSplitter& operator=(const CrossSplitter& cs);

			Committee* buildEnsemble();

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
