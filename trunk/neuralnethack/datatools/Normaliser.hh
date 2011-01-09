/*$Id: Normaliser.hh 1656 2007-07-05 14:02:29Z michael $*/

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


#ifndef __Normaliser_hh__
#define __Normaliser_hh__

#include "DataSet.hh"

#include <vector>

namespace DataTools
{
	/**A class that can normalise a DataSet using Z-Normalisation. 
	 * The normalisation is done by calculating the mean and standard 
	 * deviation of the variables. 
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

			/**Constructor for fully defining the state of the Normaliser. 
			 * \param means the vector of means.
			 * \param stds the vector of standard deviations.
			 * \param skips the vector of variables to skip.
			 */
			Normaliser(std::vector<double>& stds, std::vector<double>& means, std::vector<bool>& skips);

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

			/**Normalise a DataSet. Normalise a DataSet using already
			 * predefined mean ans std vectors.
			 * \param d the DataSet to normalise.
			 * \param doSkip flag whether to do skipping or not.
			 * \return the normalised DataSet.
			 */
			DataSet& normalise(DataSet& d);

			/**Normalise a DataSet. In the normal case every variable in the
			 * DataSet gets normalised. However this method is able to search
			 * through the DataSet to see if any variables are binary and if
			 * so skip rescaling them. Only variables varying between [-1,1]
			 * or [0,1] is skipped. If a variable is determined to be binary
			 * it's mean is set to 0 and it's standard deviation is set to 1
			 * which effectively cancels their rescaling.
			 * \param d the DataSet to normalise.
			 * \param doSkip flag whether to do skipping or not.
			 * \return the normalised DataSet.
			 */
			DataSet& calcAndNormalise(DataSet& d, bool doSkip=false);

			/**Normalise a Pattern.
			 * \param p the Pattern to normalise.
			 * \return the normalised Pattern.
			 */
			Pattern& normalise(Pattern& p);

			/**Normalise an input vector.
			 * \param i the input vector to normalise.
			 * \return the normalized input vector.
			 */
			std::vector<double>& normaliseInput(std::vector<double>& i);

			/**Unnormalise a DataSet.
			 * \param d the DataSet to unnormalise.
			 * \return the unnormalised DataSet.
			 */
			DataSet& unnormalise(DataSet& d);

			/**Unnormalise a Pattern.
			 * \param p the Pattern to unnormalise.
			 * \return the unnormalised Pattern.
			 */
			Pattern& unnormalise(Pattern& p);

			/**Return the standard deviations for the current DataSet.
			 * This is the standard deviations of the DataSet as calculated 
			 * before normalisation.
			 * \return the standard deviations of the DataSet.
			 */
			std::vector<double>& stdDev();

			/**Set the standard deviations for the current DataSet.
			 * \param v the standard deviations to set.
			 */
			void stdDev(std::vector<double>& v);

			/**Return the means for the current DataSet.
			 * This is the means of the DataSet as calculated before
			 * normalisation.
			 * \return the means of the DataSet.
			 */
			std::vector<double>& mean();

			/**Set the means for the current DataSet.
			 * \param m the means to set.
			 */
			void mean(std::vector<double>& m);

			/**Return the skips for the current DataSet.
			 * This is the skips of the DataSet as calculated before
			 * normalisation.
			 * \return the skips of the DataSet.
			 */
			std::vector<bool>& skip();

			/**Set the skips for the current DataSet.
			 * \param m the skips to set.
			 */
			void skip(std::vector<bool>& m);

		private:
			/**The standard deviation for each variable in the original 
			 * DataSet. This will be used to restore a DataSet to its 
			 * original form.
			 */
			std::vector<double> theStdDev;

			/**The mean for each variable in the original DataSet. 
			 * This will be used to restore a DataSet to its original form.
			 */
			std::vector<double> theMean;

			/**Contains the indices of the variables to skip in the DataSet.
			 * Thus the values in the columns which index is in theSkip will
			 * not be rescaled.
			 */
			std::vector<bool> theSkip;

			/**Calculate the mean for each variable in the DataSet.
			 * \param d the DataSet.
			 */
			void calcMean(DataSet& d);

			/**Calculate the standard deviation for each variable in the data
			 * set.
			 * \param d the DataSet.
			 */
			void calcStdDev(DataSet& d);

			/**Builds up the skip vector by searching for binary variables. It
			 * only looks for variables using [-1,1] or [0,1] as outputs in
			 * the DataSet.
			 * \param d the DataSet.
			 */
			void findSkip(DataSet& d);

			/**Checks the skip status of a value. This checks if the value is
			 * skipable in the [0,1] case.
			 * \param val the value to check.
			 * \return true if the value indicates skipping and false otherwise.
			 */
			bool skipBin(double val) const;

			/**Checks the skip status of a value. This checks if the value is
			 * skipable in the [-1,1] case.
			 * \param val the value to check.
			 * \return true if the value indicates skipping and false otherwise.
			 */
			bool skipSig(double val) const;

			/**Transform [0,1] binary encoding to [-1,1]. This is useful for
			 * the network learning process. Currently only the inputs are
			 * changed. The target stays at [0,1].
			 * \param data the DataSet to transform.
			 */
			void transformBinaryCoding(DataSet& data);

			/**Transform [0,1] binary encoding to [-1,1]. This is useful for
			 * the network learning process. This method only operates on an
			 * input vector directly.
			 * \param data the DataSet to transform.
			 */
			void transformBinaryCoding(std::vector<double>& input);
	};		
}
#endif
