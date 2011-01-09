/*$Id: EnsembleBuilder.hh 1627 2007-05-08 16:40:20Z michael $*/

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


#ifndef __EnsembleBuilder_hh__
#define __EnsembleBuilder_hh__

#include "mlp/Trainer.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"
#include "datatools/DataManager.hh"
#include "datatools/Sampler.hh"

namespace NeuralNetHack
{
	/**The result of a training session. A training session can
	 * include several trainings and Mlp(s).*/
	class Session
	{
		public:
			Session():committee(0), trnData(0), valData(0){}

			Session(Ensemble* c, DataTools::DataSet* trn, DataTools::DataSet* val)
				:committee(c), trnData(trn), valData(val){}

			Session(const Session& s) {*this = s;}

			~Session()
			{
				if(committee != 0) delete committee; 
				if(trnData != 0) delete trnData; 
				if(valData != 0) delete valData;
			}

			Session& operator=(const Session& s)
			{
				if(this != &s){
					committee = new Ensemble(*s.committee);
					trnData = new DataTools::DataSet(*s.trnData);
					valData = new DataTools::DataSet(*s.valData);
				}
				return *this;
			}

			/**The pointer to the ensemble that holds all the models. */
			Ensemble* committee;

			/**The pointer to the DataSet used for training. */
			DataTools::DataSet* trnData;

			/**The pointer to the DataSet used for validation. */
			DataTools::DataSet* valData;
	};

	/**A base class representing the different ensemble builders.
	 * \sa CrossSplitter, Bagger
	 */
	class EnsembleBuilder
	{
		public:
			/**Basic constructor. */
			EnsembleBuilder();

			/**Copy constructor. 
			 * \param eb the EnsembleBuilder to copy from.
			 */
			EnsembleBuilder(const EnsembleBuilder& eb);

			/**Assignment operator.
			 * \param eb the EnsembleBuilder to assign from.
			 * \return the EnsembleBuilder assigned to.
			 */
			virtual EnsembleBuilder& operator=(const EnsembleBuilder& eb);

			/**Basic destructor. */
			virtual ~EnsembleBuilder();

			/**Accessor for the Trainer pointer.
			 * \return the pointer to the Trainer.
			 */
			MultiLayerPerceptron::Trainer* trainer() const;

			/**Mutator for the Trainer pointer.
			 * \param t the pointer to the Trainer to set.
			 */
			void trainer(MultiLayerPerceptron::Trainer* t);

			/**Accessor for the Sampler pointer.
			 * \return the pointer to the Sampler.
			 */
			DataTools::Sampler* sampler() const;

			/**Mutator for the Sampler pointer.
			 * \param s the pointer to the Sampler to set.
			 */
			void sampler(DataTools::Sampler* s);

			/**Accessor for the Session vector.
			 * \return the Session vector.
			 */
			std::vector<Session>& sessions();

			/**Mutator for the Session vector.
			 * \param s the Session vector to set.
			 */
			void sessions(std::vector<Session>& s);

			/**Method that will fetch the ensemble that was built previously.
			 * \return the built ensemble.
			 */
			Ensemble* getEnsemble();

			/**Method that will build the ensemble using the current Sampler.
			 * \sa Sampler
			 * \return the built ensemble.
			 */
			Ensemble* buildEnsemble();

		private:
			/**Utility to check whether everything is ok. This checks so that
			 * all pointers are pointing at something.
			 * \return true if good to go, false otherwise.
			 */
			bool isValid() const;

			/**Pointer to the Trainer. */
			MultiLayerPerceptron::Trainer* theTrainer;

			/**Pointer to the Sampler. */
			DataTools::Sampler* theSampler;

			/**A vector of Estimations. */
			std::vector<Session> theSessions;
	};
}
#endif
