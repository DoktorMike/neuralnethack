/*$Id: NetworkParser.hh 1605 2007-01-24 20:47:36Z michael $*/

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


#ifndef __NetworkParser_hh__
#define __NetworkParser_hh__

#include "../mlp/Mlp.hh"
#include "Ensemble.hh"
#include "../datatools/Normaliser.hh"

#include <vector>
#include <utility>
#include <istream>
#include <string>

namespace NeuralNetHack
{
	/**This class encloses the network XML parsing.
	 * It contains parsing methods for the saved committees of mlps in XML
	 * form.
	 */
	class NetworkParser
	{
		public:

			/**Default constructor.
			 */
			NetworkParser();

			/**Parse the entire xml data and return the committee and
			 * normaliser.
			 * \param is the input stream to read from
			 * \return the committees and the normaliser
			 */
			std::pair<std::vector<Ensemble*>, DataTools::Normaliser*> parseXML(std::istream& is);

			/**Combine all the ensembles into one big ensemble.
			 * \param committees the vector of committees to combine
			 * \return the full committee
			 */
			Ensemble* buildEnsemble(std::vector<Ensemble*>& committees);

			/**Delete all the pointers.
			 */
			void killAll(std::vector<Ensemble*>& committees, Ensemble* committee, DataTools::Normaliser* normalisation);

		private:

			/**Parse a vector of uints from a stream.
			 * \param is the input stream to read from
			 * \param vec the vector to parse into
			 * \param stop the token that signals the parsing to stop
			 */
			void parseXMLvector(std::istream& is, std::vector<uint>& vec, std::string stop);

			/**Parse a vector of doubles from a stream.
			 * \param is the input stream to read from
			 * \param vec the vector to parse into
			 * \param stop the token that signals the parsing to stop
			 */
			void parseXMLvector(std::istream& is, std::vector<double>& vec, std::string stop);

			/**Parse a vector of strings from a stream.
			 * \param is the input stream to read from
			 * \param vec the vector to parse into
			 * \param stop the token that signals the parsing to stop
			 */
			void parseXMLvector(std::istream& is, std::vector<std::string>& vec, std::string stop);

			/**Parse an Mlp from a stream.
			 * \param is the input stream to read from
			 * \return a pointer to an Mlp
			 */
			MultiLayerPerceptron::Mlp* parseXMLmlp(std::istream& is);

			/**Parse a Ensemble of Mlps from a stream.
			 * \param is the input stream to read from
			 * \return a pointer to a Ensemble
			 */
			Ensemble* parseXMLcommittee(std::istream& is);

			/**Parse a Normalization from a stream.
			 * \param is the input stream to read from
			 * \return a pointer to a Normalization
			 */
			DataTools::Normaliser* parseXMLnormalisation(std::istream& is);

	};
}
#endif
