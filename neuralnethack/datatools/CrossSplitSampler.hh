/*$Id: CrossSplitSampler.hh 1678 2007-10-01 14:42:23Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


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
