#ifndef __Supervisor_hh__
#define __Supervisor_hh__

#include "Trainer.hh"
#include "Config.hh"

namespace NetHack
{
    ///Class responsible for supervising the training of a committee of MLPs.
    class Supervisor
    {
	public:
	    ///Basic constructor.
	    ///\param config holds configuration variables i.e. batch size etc.
	    ///\param dset the data set used for this session.
	    Supervisor(Config& config, DataSet& dset);

	    ///Basic destructor.
	    ~Supervisor();

	    ///Train the committee!
	    void train();
	    
	    ///Evaluate the committee.
	    void test();
	    
	private:
	    ///Copy constructor.
	    ///\param sup the object to copy from.
	    Supervisor(const Supervisor& sup);

	    ///Assignment operator.
	    ///\param sup the object to copy from.
	    Supervisor& operator=(const Supervisor& sup);

	    ///A committee of MLPs.
	    Committee* theCommittee;

	    ///A set of data points.
	    DataSet* dataSet;

	    ///A trainer using a learning algorithm.
	    Trainer* trainer;

	    ///A configuration class.
	    ///This keeps track of all those annoying constants
	    ///needed in ANN development.
	    Config* config;
    };
}
#endif
