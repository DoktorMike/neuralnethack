#ifndef __Bagger_hh__
#define __Bagger_hh__

#include "EnsembleBuilder.hh"
#include "Committee.hh"

namespace NeuralNetHack
{
	class Bagger: public EnsembleBuilder
	{
		public:
			Bagger();
			Bagger(const Bagger& bg);
			~Bagger();
			Bagger& operator=(const Bagger& bg);

			Committee* buildEnsemble();

			uint numRuns();
			void numRuns(uint n);

		private:
			uint n;
	};
}

#endif
