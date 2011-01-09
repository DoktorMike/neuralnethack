/*$Id: DummySampler.hh 1678 2007-10-01 14:42:23Z michael $*/

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
