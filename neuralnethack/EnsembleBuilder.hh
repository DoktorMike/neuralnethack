#ifndef __EnsembleBuilder_hh__
#define __EnsembleBuilder_hh__

#include "mlp/Trainer.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"
#include "datatools/DataManager.hh"
#include "datatools/Sampler.hh"

#include <memory>

namespace NeuralNetHack
{
	/**The result of a training session. A training session can
	 * include several trainings and Mlp(s).*/
	class Session
	{
		public:
			Session():ensemble(nullptr), trnData(nullptr), valData(nullptr){}

			Session(std::unique_ptr<Ensemble> c,
					std::unique_ptr<DataTools::DataSet> trn,
					std::unique_ptr<DataTools::DataSet> val)
				:ensemble(std::move(c)), trnData(std::move(trn)), valData(std::move(val)){}

			Session(const Session& s)
				:ensemble(s.ensemble ? std::make_unique<Ensemble>(*s.ensemble) : nullptr),
				 trnData(s.trnData ? std::make_unique<DataTools::DataSet>(*s.trnData) : nullptr),
				 valData(s.valData ? std::make_unique<DataTools::DataSet>(*s.valData) : nullptr)
			{}

			Session(Session&&) noexcept = default;

			~Session() = default;

			Session& operator=(const Session& s)
			{
				if(this != &s){
					ensemble = s.ensemble ? std::make_unique<Ensemble>(*s.ensemble) : nullptr;
					trnData = s.trnData ? std::make_unique<DataTools::DataSet>(*s.trnData) : nullptr;
					valData = s.valData ? std::make_unique<DataTools::DataSet>(*s.valData) : nullptr;
				}
				return *this;
			}

			Session& operator=(Session&&) noexcept = default;

			/**The ensemble that holds all the models. */
			std::unique_ptr<Ensemble> ensemble;

			/**The DataSet used for training. */
			std::unique_ptr<DataTools::DataSet> trnData;

			/**The DataSet used for validation. */
			std::unique_ptr<DataTools::DataSet> valData;
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
