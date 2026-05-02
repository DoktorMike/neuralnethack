#ifndef __Trainer_hh__
#define __Trainer_hh__

#include "Error.hh"
#include "../datatools/DataSet.hh"

#include <fstream>
#include <memory>
#include <string>
#include <ostream>

namespace MultiLayerPerceptron {
/**A base class representing the training of an MLP. */
class Trainer {
  public:
	/**Basic destructor. */
	virtual ~Trainer();

	/**Return a pointer to the Mlp being trained.
	 * \return the mlp pointer.
	 */
	Mlp* mlp();

	/**Set the mlp pointer that shall be trained.
	 * \param mlp the pointer to the mlp that shall be trained.
	 */
	void mlp(Mlp* mlp);

	/**Get the DataSet this Trainer should use for training.
	 * \return the pointer to the DataSet.
	 */
	DataTools::DataSet* data();

	/**Set the DataSet this Trainer should use for training.
	 * \param d the pointer to the DataSet.
	 */
	void data(DataTools::DataSet* d);

	/**Return the error function this Trainer is using.
	 * \return the error function.
	 */
	Error* error();

	/**Set the error function to train this committee with.
	 * \param e the error function.
	 */
	void error(Error* e);

	/**Return the number of epochs to train for.
	 * \return the number of epochs.
	 */
	uint numEpochs() const;

	/**Set the number of epochs to train for.
	 * \param ne the number of epochs.
	 */
	void numEpochs(uint ne);

	/**Return the training error for this trainer. */
	double trainingError() const;

	/**Set the training error for this trainer.
	 * \param te the training error.
	 */
	void trainingError(double te);

	/**Get the batch size for this trainer.
	 * \return the batch size.
	 */
	uint batchSize() const;

	/**Set the batch size for this trainer.
	 * \param bs the batch size.
	 */
	void batchSize(uint bs);

	/**Optional validation DataSet used for the learning-curve file.
	 * Setting this on its own does nothing; pair with learningCurveFile.
	 */
	void validationData(DataTools::DataSet* v);
	DataTools::DataSet* validationData() const;

	/**When set, the trainer writes a gnuplot-friendly learning-curve
	 * file as it trains: one row per recorded epoch, columns
	 * `epoch  trainErr` (and `valErr` when validationData is also set).
	 * Pass an empty string to disable. Cloned trainers do not inherit
	 * the path so they don't all clobber the same file.
	 */
	void learningCurveFile(const std::string& path);
	const std::string& learningCurveFile() const;

	/**Tells wether this trainer has everything it needs in order to
	 * perform training.
	 * \return true if everything is ready, false if something is missing.
	 */
	bool isValid() const;

	/**Method used to train an MLP. This method sets the member
	 * attributes Mlp and DataSet and then performs training.
	 * \param mlp the MLP to train.
	 * \param dset the data set to train the MLP on.
	 * \param os the ostream to print the training process to.
	 */
	void train(Mlp& mlp, DataTools::DataSet& dset, std::ostream& os);

	/**Method used to train a copy of the current MLP.
	 * This creates a copy of the current
	 * Mlp, trains it and returns it to the caller.
	 * The Error function in the Trainer is the same.
	 * \param dset the data set to train the MLP on.
	 * \param os the ostream to print the training process to.
	 * \return a unique_ptr to the newly trained Mlp.
	 */
	std::unique_ptr<Mlp> trainNew(DataTools::DataSet& dset, std::ostream& os);

	/**Method used to train a copy of the current MLP.
	 * This creates a copy of the current
	 * Mlp, trains it and returns it to the caller.
	 * The DataSet and the Error function in the Trainer is the same.
	 * \param os the ostream to print the training process to.
	 * \return a unique_ptr to the newly trained Mlp.
	 */
	std::unique_ptr<Mlp> trainNew(std::ostream& os);

	/**Method used to train an MLP. This uses the Mlp the DataSet
	 * and the Error function in the Trainer.
	 * \param os the ostream to print the training process to.
	 */
	virtual void train(std::ostream& os) = 0;

	/**Method that clones the Trainer and returns a copy of it.
	 * \return a unique_ptr to the clone of this Trainer.
	 */
	virtual std::unique_ptr<Trainer> clone() const = 0;

  protected:
	/**Non-owning constructor: caller keeps mlp/error alive. */
	Trainer(Mlp& mlp, DataTools::DataSet& data, Error& error, double te, uint bs);

	/**Owning constructor: this Trainer takes ownership of the Error
	 * (and transitively whatever the Error owns, like the Mlp).
	 */
	Trainer(std::unique_ptr<Error> error, DataTools::DataSet& data, double te, uint bs);

	/**Copy constructor.
	 * \param trainer the Trainer object to copy.
	 */
	Trainer(const Trainer& trainer);

	/**Assignment operator.
	 * \param trainer the Trainer object to copy.
	 */
	Trainer& operator=(const Trainer& trainer);

	/**Check if convergence criterion is reached.
	 * \param ecurr the error in the current epoch.
	 * \param eprev the error from the previous epoch.
	 * \return true if criterion is met, false otherwise.
	 */
	bool hasConverged(double ecurr, double eprev) const;

	/**Append one row to the learning-curve file. No-op if no path is
	 * set. Lazy-opens the file on first call (truncating any existing
	 * content) and writes a `# epoch  trainErr [valErr]` header. If
	 * validationData is set, also computes and emits the val error
	 * (restoring the Error's internal mlp/dset pointers afterwards so
	 * subsequent gradient() calls still target the training data).
	 */
	void recordLearningPoint(uint epoch, double trainErr);

	/**The Mlp this Trainer is using. */
	Mlp* theMlp;

	/**The DataSet this Trainer will be using. */
	DataTools::DataSet* theData;

	/**The error function. */
	Error* theError;

	/**Optional ownership of the Error. Set when constructed via the
	 * owning ctor; null when constructed from a reference.
	 */
	std::unique_ptr<Error> theOwnedError;

	/**The number of epochs to train for. */
	uint theNumEpochs;

	/**The error required to stop training. */
	double theTrainingError;

	/**The number of patterns to use every epoch. */
	uint theBatchSize;

	/**Optional non-owning pointer to a validation DataSet. */
	DataTools::DataSet* theValData = nullptr;

	/**Path for the learning-curve file. Empty = disabled. */
	std::string theLearningCurvePath;

	/**Lazy-opened stream for the learning-curve file. */
	std::unique_ptr<std::ofstream> theLearningCurveStream;

  private:
};
} // namespace MultiLayerPerceptron
#endif
