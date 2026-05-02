#ifndef __EnsembleBuilder_hh__
#define __EnsembleBuilder_hh__

#include "mlp/Trainer.hh"
#include "Ensemble.hh"
#include "datatools/DataSet.hh"
#include "datatools/DataManager.hh"
#include "datatools/Sampler.hh"

#include <cstdint>
#include <functional>
#include <memory>

namespace NeuralNetHack {
/**The result of a training session. A training session can
 * include several trainings and Mlp(s).*/
class Session {
  public:
	Session() : ensemble(nullptr), trnData(nullptr), valData(nullptr) {}

	Session(std::unique_ptr<Ensemble> c, std::unique_ptr<DataTools::DataSet> trn,
	        std::unique_ptr<DataTools::DataSet> val)
	    : ensemble(std::move(c)), trnData(std::move(trn)), valData(std::move(val)) {}

	Session(const Session& s)
	    : ensemble(s.ensemble ? std::make_unique<Ensemble>(*s.ensemble) : nullptr),
	      trnData(s.trnData ? std::make_unique<DataTools::DataSet>(*s.trnData) : nullptr),
	      valData(s.valData ? std::make_unique<DataTools::DataSet>(*s.valData) : nullptr) {}

	Session(Session&&) noexcept = default;

	~Session() = default;

	Session& operator=(const Session& s) {
		if (this != &s) {
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
class EnsembleBuilder {
  public:
	EnsembleBuilder();
	EnsembleBuilder(const EnsembleBuilder&) = delete;
	EnsembleBuilder& operator=(const EnsembleBuilder&) = delete;
	EnsembleBuilder(EnsembleBuilder&&) noexcept = default;
	EnsembleBuilder& operator=(EnsembleBuilder&&) noexcept = default;
	virtual ~EnsembleBuilder();

	/**Accessor for the Trainer (read-only). */
	MultiLayerPerceptron::Trainer* trainer() const;

	/**Take ownership of a Trainer. */
	void trainer(std::unique_ptr<MultiLayerPerceptron::Trainer> t);

	/**Accessor for the Sampler (read-only). */
	DataTools::Sampler* sampler() const;

	/**Take ownership of a Sampler. */
	void sampler(std::unique_ptr<DataTools::Sampler> s);

	/**Set a factory that returns a fresh, fully-owned Trainer for each
	 * ensemble member. When set, `buildEnsemble` parallelises member
	 * training; without a factory it falls back to serial training using
	 * the single shared Trainer set via `trainer()`.
	 *
	 * The factory receives the per-member training DataSet. Required if
	 * you want OpenMP-parallel ensemble training, since a single Trainer
	 * holds mutable state (Mlp weights, Adam moments, ...) that would
	 * race across threads.
	 */
	using TrainerFactory =
	    std::function<std::unique_ptr<MultiLayerPerceptron::Trainer>(DataTools::DataSet&)>;
	void trainerFactory(TrainerFactory f);

	/**Base seed for per-member RNG seeding inside parallel buildEnsemble.
	 * Each iteration seeds its worker's thread-local RNG with
	 * baseSeed + member_index, so weight init is deterministic and
	 * diverse across members regardless of thread scheduling.
	 */
	void baseSeed(uint64_t s);

	/**Base path for per-member learning-curve files. When set, each
	 * ensemble member's trainer writes a gnuplot-friendly file named
	 * `<stem>_<NNN>.<ext>` (NNN = zero-padded member index, 1-based).
	 * The member's out-of-bag DataSet (samples[i].second) is wired up
	 * as validation so the file gets a `valErr` column too. Empty
	 * disables. Setter on its own does nothing without a sampler that
	 * yields meaningful out-of-bag splits (Bootstrap, CrossSplit, ...).
	 */
	void learningCurvePathBase(const std::string& base);
	const std::string& learningCurvePathBase() const;

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

	/**Owned Trainer. */
	std::unique_ptr<MultiLayerPerceptron::Trainer> theTrainer;

	/**Owned Sampler. */
	std::unique_ptr<DataTools::Sampler> theSampler;

	/**Optional factory for building a fresh Trainer per member. */
	TrainerFactory theTrainerFactory;

	/**Base seed for per-member RNG seeding under parallel training. */
	uint64_t theBaseSeed{1};

	/**Base path for per-member learning-curve files (empty = disabled). */
	std::string theLearningCurvePathBase;

	/**A vector of Estimations. */
	std::vector<Session> theSessions;
};
} // namespace NeuralNetHack
#endif
