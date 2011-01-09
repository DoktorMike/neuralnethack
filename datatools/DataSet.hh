#ifndef __DataSet_hh__
#define __DataSet_hh__

#include "CoreDataSet.hh"

#include <vector>
#include <iostream>

namespace DataTools
{
	/**A class functioning as layer between a user and the actual CoreDataSet.
	 * The entries in this class will only contain indices to the actual data.
	 * Thus we can have several different DataSet(s) without copying the
	 * actual data.
	 * \sa CoreData, Pattern
	 */
	class DataSet
	{
		public:
			/**Basic constructor. */
			DataSet();

			/**Copy constructor.
			 * \param dataSet the data set to copy from.
			 */
			DataSet(const DataSet& dataSet);

			/**Basic destructor. */
			~DataSet();

			/**Assignment operator.
			 * \param dataSet the data set to assign from.
			 */
			DataSet& operator=(const DataSet& dataSet);

			/**Return the indices mapping this DataSet to the CoreDataSet.
			 * \return the indices.
			 */
			std::vector<uint>& indices();

			/**Set the indices mapping this DataSet to the CoreDataSet. Note
			 * that the size of the vector i cannot be larger than the size of
			 * the CoreDataSet. Also i may not contain an index greater than
			 * the size of the CoreDataSet -1.
			 * \param i the indices to use.
			 */
			void indices(std::vector<uint>& i);

			/**Return the pattern at index.
			 * \param index the index of the pattern to return.
			 * \return the indexed pattern.
			 */
			Pattern& pattern(uint index);

			/**Return the CoreDataSet this DataSet operates on.
			 * \return the CoreDataSet.
			 */
			CoreDataSet& coreDataSet();
			
			/**Set the CoreDataSet this DataSet should operate on. This also
			 * reinitialiises theIndices vector to the size of the
			 * CoreDataSet, with elements 0 up to size-1.
			 * \param cds the CoreDataSet to set.
			 */
			void coreDataSet(CoreDataSet& cds);
			
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
			/**Holds the indices. Each and every one of the indices refers to
			 * a particular Pattern in CoreDataSet.
			 */
			std::vector<uint> theIndices;

			/**The index iterator. */
			std::vector<uint>::iterator itp;

			/**The CoreDataSet the indices operate on. */
			CoreDataSet* theCoreDataSet;
	};
}
#endif
