#ifndef __DataSet_hh__
#define __DataSet_hh__

#include "CoreDataSet.hh"

#include <iostream>
#include <memory>
#include <vector>

namespace DataTools {
/**A view layer over a CoreDataSet. Each DataSet holds a `shared_ptr` to the
 * underlying CoreDataSet plus its own index vector, so multiple views (train
 * splits, bootstrap samples, etc.) can share the same underlying patterns
 * without copying them.
 *
 * Lifetime is automatic: the CoreDataSet is destroyed when the last DataSet
 * referencing it goes out of scope.
 *
 * \sa CoreDataSet, Pattern
 */
class DataSet {
  public:
	DataSet();
	DataSet(const DataSet& dataSet);
	DataSet(DataSet&&) noexcept = default;
	~DataSet();
	DataSet& operator=(const DataSet& dataSet);
	DataSet& operator=(DataSet&&) noexcept = default;

	/**Return the indices mapping this DataSet to the CoreDataSet. */
	std::vector<uint>& indices();

	/**Set the indices mapping this DataSet to the CoreDataSet. */
	void indices(std::vector<uint>& i);

	/**Return the pattern at index. */
	Pattern& pattern(uint index);

	/**Reference to the underlying CoreDataSet. */
	CoreDataSet& coreDataSet();

	/**Bind this DataSet to a CoreDataSet. Resets indices to {0, 1, ..., N-1}.
	 *
	 * Pass a `std::shared_ptr<CoreDataSet>` so ownership is unambiguous and
	 * the data outlives every DataSet that views it. Use
	 * `std::make_shared<CoreDataSet>()` to construct one.
	 */
	void coreDataSet(std::shared_ptr<CoreDataSet> cds);

	/**Return the shared_ptr to the underlying CoreDataSet. Useful for
	 * building another DataSet view onto the same data.
	 */
	std::shared_ptr<CoreDataSet> sharedCoreDataSet() const;

	uint nInput() const;
	uint nOutput() const;
	uint size() const;
	void print(std::ostream& os) const;

  private:
	std::vector<uint> theIndices;
	std::vector<uint>::iterator itp;
	std::shared_ptr<CoreDataSet> theCoreDataSet;
};
} // namespace DataTools
#endif
