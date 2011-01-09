#include "DataSet.hh"

namespace DataTools
{
	/**A class representing splitting one large DataSet into K DataSet(s). 
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
			bool random();

			/**Set state of the data selection mode.
			 * \param rnd controlling randomised split on/off.
			 */
			void random(bool rnd);
			
			/**Split the DataSet into 2 parts, Training and Testing. 
			 * The reason for returning a
			 * real vector instead of a reference is that DataSet is a rather
			 * lightweight structure.
			 * \param ds the DataSet to split.
			 * \param ratio the ratio between training and testing.
			 * \return a vector of the k created DataSet(s).
			 */
			vector<DataSet> split(DataSet& ds, double ratio);

			/**Split the DataSet into k parts. The reason for returning a
			 * real vector instead of a reference is that DataSet is a rather
			 * lightweight structure.
			 * \param ds the DataSet to split.
			 * \param k the number of splits.
			 * \return a vector of the k created DataSet(s).
			 */
			vector<DataSet> split(DataSet& ds, uint k);

			/**Concatenate all DataSet(s) in the vector to one DataSet.
			 * \param splits the DataSet(s) to concatenate.
			 * \return one DataSet consisting of the concatenated DataSet(s).
			 */
			DataSet join(vector<DataSet>& splits);
			
		private:
			/**Build the index vector. This puts all the indices of the DataSet
			 * into a vector in random order.
			 * \param n the number of indices to generate.
			 */
			void buildIndices(uint n);

			/**Print the built indices. */
			void printIndices();
			
			/**A vector containing an index for every data point in the
			 * original DataSet.
			 */
			vector<uint> indices;

			/**Toggle wheather to do the split in a randomised manner or not.
			 * Default is true. When set to false data selection is done
			 * sequentially.
			 */
			bool isRandom;
	};
}
