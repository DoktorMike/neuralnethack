#ifndef __DataSet_hh__
#define __DataSet_hh__

#include "Pattern.hh"

namespace DataTools
{
    ///A class representing the entire dataset available to this MLP.
    class DataSet
    {
	public:
	    ///Basic constructor.
	    DataSet();

	    ///Copy constructor.
	    ///\param dataSet the data set to copy from.
	    DataSet(const DataSet& dataSet);

	    ///The destructor.
	    ~DataSet();

	    ///Assignment operator.
	    ///\param dataSet the data set to assign from.
	    DataSet& operator=(const DataSet& dataSet);

	    ///Return the remaining number of patterns in this data set.
	    int remaining();

	    ///Returns the next pattern.
	    Pattern& nextPattern();

	    ///Returns the current pattern.
	    Pattern& currentPattern();

	    ///Returns the previous pattern.
	    Pattern& previousPattern();

	    ///Adds a pattern to this data set.
	    ///\param pattern the pattern to add.
	    void addPattern(const Pattern& pattern);

	    /**Return the number of inputs that each patterns holds. */
	    uint nInput();

	    /**Return the number of outputs that each patterns holds. */
	    uint nOutput();

	    ///Set the iterator to the last element and the counter to size.
	    void reset();

	    ///Return the number of patterns residing in this data set.
	    uint size();

	    ///Print the data set to output stream.
	    ///\param os the output stream to print to.
	    void print(ostream& os);

	private:
	    ///The number of data points remaining in the set. 
	    ///The int refers to the number of elements after 
	    ///the current one.
	    int nLeft;

	    ///Holds the patterns.
	    vector<Pattern> patterns;

	    ///The patterns iterator.
	    vector<Pattern>::iterator itp;
    };
}
#endif
