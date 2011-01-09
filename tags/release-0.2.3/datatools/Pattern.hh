#ifndef __Pattern_hh__
#define __Pattern_hh__

#include "DataTools.hh"

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
			 * \param in the input portion of the pattern.
			 * \param out the output portion of the pattern.
			 */
			Pattern(vector<double>& in, vector<double>& out);

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

			/**Returns the input vector. */
			vector<double>& input();

			/**Sets the input vector.
			 * \param in the input vector to use.
			 */
			void input(vector<double>& in);

			/**Fetch the number of inputs this pattern uses.
			 * \return the number of inputs.
			 */
			uint nInput() const;

			/**Returns the output vector.
			 * \return the output vector.
			 */
			vector<double>& output();

			/**Sets the output vector.
			 * \param out the output vector to use.
			 */
			void output(vector<double>& out);

			/**Return the number of outputs this pattern uses.
			 * \return the number of outputs.
			 */
			uint nOutput() const;

			/**Print this pattern. */
			void print(ostream& os);

		private:
			/**The input portion of this pattern. */
			vector<double> in;

			/**The output portion of this pattern. */
			vector<double> out;
	};
}
#endif
