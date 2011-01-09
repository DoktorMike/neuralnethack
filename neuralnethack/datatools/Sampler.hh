/*$Id: Sampler.hh 1678 2007-10-01 14:42:23Z michael $*/

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


#ifndef __Sampler_hh__
#define __Sampler_hh__

#include "DataSet.hh"
#include "DataManager.hh"

namespace DataTools
{
	class Sampler
	{
		public:

			/**Basic destructor. */
			virtual ~Sampler();

			/**Return the next trainig and validation sample.
			 * Note that it's the users responsability to destroy the pointer
			 * returned.
			 * \return a pair of training and validation DataSet.
			 */
			virtual std::pair<DataSet, DataSet>* next() = 0;

			/**Check if there is another sample left.
			 * \return true when there is at least one more sample in the que, false otherwise.
			 */
			virtual bool hasNext() const = 0;

			/**Check how many samples we get in total.
			 * \return the total number of samples we can draw from this Sampler.
			 */
			virtual uint howMany() const = 0;

			/**Accessor for the random sampling attribute.
			 * \return true if random sampling is on, false otherwise.
			 */
			bool randomSampling() const;

			/**Mutator for the random sampling attribute.
			 * \param rs set to true if random sampling is on, false otherwise.
			 */
			void randomSampling(bool rs);

			/**Accessor for the DataSet pointer.
			 * \return the pointer to the DataSet.
			 */
			DataTools::DataSet* data() const;

			/**Mutator for the Data pointer.
			 * \param d the pointer to the Data to set.
			 */
			void data(DataTools::DataSet* d);

			/**Reset the Sampler. This should be called whenever the
			 * attributes has been changed.
			 */
			virtual void reset() = 0;

		protected:

			/**Basic constructor.
			 * \param data the DataSet to sample from.
			 */
			Sampler(DataSet& data);

			/**Copy constructor. 
			 * \param eb the Sampler to copy from.
			 */
			Sampler(const Sampler& eb);

			/**Assignment operator.
			 * \param eb the Sampler to assign from.
			 * \return the Sampler assigned to.
			 */
			virtual Sampler& operator=(const Sampler& eb);

			/**Pointer to the DataManager. */
			DataManager* theDataManager;

			/**Pointer to the Data. */
			DataSet* theData;

			/**A vector of the splits. */
			std::vector<DataSet>* theSplits;

		private:

	};
}

#endif
