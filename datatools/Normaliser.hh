#ifndef __Normaliser_hh__
#define __Normaliser_hh__

#include "DataSet.hh"

namespace DataTools{

	/**A class that can normalise a data set. The normalisation is done by 
	 * calculating the mean and standard deviation of the variables. 
	 * \f[z=\frac{x-\mu_x}{\sigma_x}\f] where \f$z\f$ is 
	 * the score, \f$\mu_x\f$ is the mean, and \f$\sigma_x\f$ is the sample 
	 * standard deviation for the random variable \f$x\f$.
	 * \sa DataSet
	 */
	class Normaliser
	{
		public:
			/**Basic constructor. */
			Normaliser();

			/**Copy constructor.
			 * \param n the Normaliser to copy from.
			 */
			Normaliser(const Normaliser& n);

			/**Basic destructor. */
			~Normaliser();

			/**Assignment operator.
			 * \param n the Normaliser to assign from.
			 */
			Normaliser& operator=(const Normaliser& n);

			/**Normalise a data set. In the normal case every variable in the
			 * data set gets normalised. However this method is able to search
			 * through the data set to see if any variables are binary and if
			 * so skip rescaling them. Only variables varying between [-1,1]
			 * or [0,1] is skipped. If a variable is determined to be binary
			 * it's mean is set to 0 and it's standard deviation is set to 1
			 * which effectively cancels their rescaling.
			 * \param d the data set to normalise.
			 * \param doSkip flag whether to do skipping or not.
			 * \return the normalised data set.
			 */
			DataSet& normalise(DataSet& d, bool doSkip=false);

			/**Unnormalise a data set.
			 * \param d the data set to unnormalise.
			 * \return the unnormalised data set.
			 */
			DataSet& unnormalise(DataSet& d);

			/**Return the standard deviations for the current data set.
			 * This is the standard deviations of the data set as calculated 
			 * before normalisation.
			 * \return the standard deviations of the data set.
			 */
			vector<double>& stdDev();

			/**Set the standard deviations for the current data set.
			 * \param v the standard deviations to set.
			 * \deprecated This lacks purpose since this variable has to be
			 * determined from the data set at hand.
			 */
			void stdDev(vector<double>& v);

			/**Return the means for the current data set.
			 * This is the means of the data set as calculated before
			 * normalisation.
			 * \return the means of the data set.
			 */
			vector<double>& mean();

			/**Set the means for the current data set.
			 * \param m the means to set.
			 * \deprecated This lacks purpose since this variable has to be
			 * determined from the data set at hand.
			 */
			void mean(vector<double>& m);

		private:
			/**The standard deviation for each variable in the original 
			 * data set. This will be used to restore a data set to its 
			 * original form.
			 */
			vector<double> theStdDev;

			/**The mean for each variable in the original data set. 
			 * This will be used to restore a data set to its original form.
			 */
			vector<double> theMean;

			/**Contains the indeces of the variables to skip in the data set.
			 * Thus the values in the columns which index is in theSkip will
			 * not be rescaled.
			 */
			vector<bool> theSkip;

			/**Calculate the mean for each variable in the data set.
			 * \param d the data set.
			 */
			void calcMean(DataSet& d);

			/**Calculate the standard deviation for each variable in the data
			 * set.
			 * \param d the data set.
			 */
			void calcStdDev(DataSet& d);

			/**Builds up the skip vector by searching for binary variables. It
			 * only looks for variables using [-1,1] or [0,1] as outputs in
			 * the data set.
			 * \param d the data set.
			 */
			void findSkip(DataSet& d);

			/**Checks the skip status of a value. This checks if the value is
			 * skipable in the [0,1] case.
			 * \param val the value to check.
			 * \return true if the value indicates skipping and false otherwise.
			 */
			bool skipBin(double val);

			/**Checks the skip status of a value. This checks if the value is
			 * skipable in the [-1,1] case.
			 * \param val the value to check.
			 * \return true if the value indicates skipping and false otherwise.
			 */
			bool skipSig(double val);
	};		
}
#endif
