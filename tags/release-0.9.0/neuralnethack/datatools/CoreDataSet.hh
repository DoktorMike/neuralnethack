/*$Id: CoreDataSet.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __CoreDataSet_hh__
#define __CoreDataSet_hh__

#include "Pattern.hh"

#include <vector>
#include <iostream>

namespace DataTools
{
	/**A class representing data storage.
	 * \sa DataSet, Pattern
	 */
	class CoreDataSet
	{
		public:
			/**Basic constructor. */
			CoreDataSet();

			/**Copy constructor.
			 * \param coreDataSet the data set to copy from.
			 */
			CoreDataSet(const CoreDataSet& coreDataSet);

			/**Basic destructor. */
			~CoreDataSet();

			/**Assignment operator.
			 * \param coreDataSet the data set to assign from.
			 */
			CoreDataSet& operator=(const CoreDataSet& coreDataSet);

			/**Returns the pattern at index.
			 * \param index the index for the pattern to fetch.
			 * \return the specified pattern.
			 */
			Pattern& pattern(uint index);
			
			/**Fetch all patterns residing in this CoreDataSet stored in a vector.
			 * \return the pattern vector.
			 */
			std::vector<Pattern>& patternVector();

			/**Adds a pattern to this data set.
			 * \param pattern the pattern to add.
			 */
			void addPattern(const Pattern& pattern);

			/**Fetch the number of inputs a pattern uses.
			 * \return the number of inputs.
			 */
			uint nInput() const;

			/**Fetch the number of outputs a pattern uses.
			 * \return the number of outputs.
			 */
			uint nOutput() const;

			/**Return the number of patterns residing in this data set.
			 * \return the total number of patterns.
			 */
			uint size() const;

			/**Print the data set to output stream.
			 * \param os the output stream to print to.
			 */
			void print(std::ostream& os) const;

		private:
			/**Holds the patterns. */
			std::vector<Pattern> patterns;

			/**The patterns iterator. */
			std::vector<Pattern>::iterator itp;
	};
}
#endif
