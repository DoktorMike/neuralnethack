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
			void print(std::ostream& os);

		private:
			/**Holds the patterns. */
			std::vector<Pattern> patterns;

			/**The patterns iterator. */
			std::vector<Pattern>::iterator itp;
	};
}
#endif
