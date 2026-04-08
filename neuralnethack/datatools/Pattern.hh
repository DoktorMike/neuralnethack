/*$Id: Pattern.hh 3344 2009-03-13 00:04:02Z michael $*/

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


#ifndef __Pattern_hh__
#define __Pattern_hh__

#include "DataTools.hh"

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

namespace DataTools
{
	/**A class representing a pattern. A pattern consists of two elements.
	 * (i)The data point i.e. input to the classifier.
	 * (ii)The classification i.e. the target output of the classifier.
	 */
	class Pattern
	{
		public:
			/**Basic constructor.
			 * \param id the identification string for this pattern.
			 * \param in the input portion of the pattern.
			 * \param out the output portion of the pattern.
			 */
			Pattern(std::string id, std::vector<double>& in, std::vector<double>& out);

			/**Empty constructor. */
			Pattern();

			/**Copy constructor.
			 * \param pattern the pattern to copy from.
			 */
			Pattern(const Pattern& pattern);

			/**The destructor. */
			~Pattern();

			/**Assignment operator.
			 * \param pattern the Pattern to assign from.
			 */
			Pattern& operator=(const Pattern& pattern);

			/**Returns the identification string for this pattern.
			 * \return the identification string.
			 */
			std::string& idstring();

			/**Sets the identification string for this pattern.
			 * \param id the identification string to use.
			 */
			void idstring(const std::string& id);

			/**Returns the input vector. */
			std::vector<double>& input();

			/**Sets the input vector.
			 * \param in the input vector to use.
			 */
			void input(std::vector<double>& in);

			/**Fetch the number of inputs this pattern uses.
			 * \return the number of inputs.
			 */
			uint nInput() const;

			/**Returns the output vector.
			 * \return the output vector.
			 */
			std::vector<double>& output();

			/**Sets the output vector.
			 * \param out the output vector to use.
			 */
			void output(std::vector<double>& out);

			/**Return the number of outputs this pattern uses.
			 * \return the number of outputs.
			 */
			uint nOutput() const;

			/**Print this pattern. */
			void print(std::ostream& os) const;

		private:
			/**The identification string of this pattern. */
			std::string id;

			/**The input portion of this pattern. */
			std::vector<double> in;

			/**The output portion of this pattern. */
			std::vector<double> out;
	};

	inline std::string& Pattern::idstring() { return id; }
	inline void Pattern::idstring(const std::string& id) { this->id = id; }
}
#endif
