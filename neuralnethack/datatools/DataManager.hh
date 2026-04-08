/*$Id: DataManager.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __DataManager_hh__
#define __DataManager_hh__

#include "DataSet.hh"

#include <vector>
#include <ostream>
#include <utility>

namespace DataTools
{
	/**A class representing splitting one large DataSet into K DataSet(s).
	 * Both bootstrap and cross-splitting is possible.
	 * This can be done sequentially or randomised.
	 * \sa DataSet
	 */
	class DataManager
	{
		public:
			/**Basic constructor. */
			DataManager();

			/**Copy constructor.
			 * \param ds the DataManager to copy from.
			 */
			DataManager(const DataManager& ds);

			/**Basic destructor. */
			~DataManager();

			/**Assignment operator.
			 * \param ds the DataManager to assign from.
			 * \return the new DataManager.
			 */
			DataManager& operator=(const DataManager& ds);
			
			/**Get state of the data selection mode.
			 * \return true if randomised data split is on, false otherwise.
			 */
			bool random() const;

			/**Set state of the data selection mode.
			 * \param rnd controlling randomised split on/off.
			 */
			void random(bool rnd);
			
			/**Split the DataSet into two disjuctive parts.
			 * The split is often referred to as training and validation. 
			 * \param ds the DataSet to split.
			 * \param ratio the ratio between training and testing.
			 * \return a pair of DataSet where first is training and second is validation.
			 */
			std::pair<DataSet, DataSet>* split(DataSet& ds, double ratio);

			/**Split the DataSet into k disjunctive parts. The reason for returning a
			 * real vector instead of a reference is that DataSet is a rather
			 * lightweight structure.
			 * \param ds the DataSet to split.
			 * \param k the number of splits.
			 * \return a vector of the k created DataSet(s).
			 */
			std::vector<DataSet>* split(DataSet& ds, uint k);

			/**Split the DataSet into two disjunctive parts by bootstrapping it. 
			 * The splitting is done by bootstrapping the DataSet which means
			 * that the DataSet will be sampled with replacement n times where
			 * n is the size of the DataSet. The elements that were never
			 * sampled is left for the validation set.
			 * \param ds the DataSet to split.
			 * \return a pair of DataSet where first is training and second is validation.
			 */
			std::pair<DataSet, DataSet>* split(DataSet& ds);

			/**Concatenate all DataSet(s) in the vector to one DataSet.
			 * \param splits the DataSet(s) to concatenate.
			 * \return one DataSet consisting of the concatenated DataSet(s).
			 */
			DataSet join(std::vector<DataSet>& splits);
			
		private:
			/**Build the index vector. This puts all the indices of the DataSet
			 * into a vector in random order.
			 * \param n the number of indices to generate.
			 */
			void buildIndices(uint n);

			/**Build the index vector. This puts all the indices of the DataSet
			 * into a vector in random order. This doesn't check whether we
			 * have already added the generated index. Thus we sample with
			 * replacement. This is meant to be used in conjuction with
			 * bagging.
			 * \param n the number of indices to generate.
			 */
			void buildIndicesWithReplacement(uint n);

			/**Build the index vector. This puts all the indices of the DataSet
			 * into a vector in random order. This doesn't check whether we
			 * have already added the generated index. Thus we sample with
			 * replacement. This is meant to be used in conjuction with
			 * bagging.
			 * \param orig the indices to generate from.
			 */
			void buildIndicesWithReplacement(std::vector<uint>& orig);

			/**Print the built indices.
			 * \param os the output stream to write to.
			 */
			void printIndices(std::ostream& os);
			
			/**A vector containing an index for every data point in the
			 * original DataSet.
			 */
			std::vector<uint> indices;

			/**Toggle wheather to do the split in a randomised manner or not.
			 * Default is true. When set to false data selection is done
			 * sequentially.
			 */
			bool isRandom;
	};
}

#endif
