#ifndef __PrintUtils_hh__
#define __PrintUtils_hh__

#include "datatools/DataSet.hh"
#include "datatools/Normaliser.hh"
#include "mlp/Mlp.hh"
#include "Ensemble.hh"
#include "ModelEstimator.hh"
#include "EnsembleBuilder.hh"
#include "Config.hh"

#include <ostream>
#include <string>

namespace NeuralNetHack {
/**This namespace encloses functions for printing various data.
 */
namespace PrintUtils {
/**Print the targets of the DataSet.
 * \param os the stream to write to.
 * \param id the string to print before the data on a separate line.
 * \param data the DataSet containing the targets to print.
 */
void printTargetList(std::ostream& os, std::string id, DataTools::DataSet& data);

/**Print the outputs of the Mlp on each Pattern in the DataSet.
 * \param os the stream to write to.
 * \param id the string to print before the data on a separate line.
 * \param mlp the Mlp whose output to write.
 * \param data the DataSet containing the patterns to propagate.
 */
void printOutputList(std::ostream& os, std::string id, MultiLayerPerceptron::Mlp& mlp,
                     DataTools::DataSet& data);

/**Print the outputs of the Ensemble on each Pattern in the DataSet.
 * \param os the stream to write to.
 * \param id the string to print before the data on a separate line.
 * \param c the Ensemble whose output to write.
 * \param data the DataSet containing the patterns to propagate.
 */
void printOutputList(std::ostream& os, std::string id, Ensemble& c, DataTools::DataSet& data);

/**Print the outputs and targets of the Ensemble created via either
 * the EnsembleBuilder or the ModelEstimator. This also checks if the
 * user has specified to actually do the printing.
 * \param os the stream to write to.
 * \param sessions the training and testing networks and their data.
 * \param trnData the training data.
 * \param tstData the data used for testing the model.
 * \param config the Config class containing all information.
 */
void printTstEnslist(std::ostream& os, std::vector<Session>& sessions, DataTools::DataSet& trnData,
                     DataTools::DataSet& tstData, const Config& config);

/**Print the outputs and targets of the Ensemble created via either
 * the EnsembleBuilder or the ModelEstimator. This also checks if the
 * user has specified to actually do the printing.
 * \param os the stream to write to.
 * \param sessions the training and testing networks and their data.
 * \param trnData the training data.
 * \param tstData the data used for testing the model.
 * \param config the Config class containing all information.
 */
void printValEnslist(std::ostream& os, std::vector<Session>& sessions, DataTools::DataSet& trnData,
                     DataTools::DataSet& tstData, const Config& config);

/**Print the importance of each variable using the Saliency measure.
 * \param os the stream to write to.
 * \param sessions the training and testing networks and their data.
 * \param config the Config class containing all information.
 */
void printSaliencies(std::ostream& os, std::vector<Session>& sessions, const Config& config);

/**Print the entire network structure. Basically prints all useful
 * information about the NeuralNetHack session i.e. each trained
 * network and it's architechture and weights.
 * \param os the stream to write to.
 * \param sessions the training and testing networks and their data.
 * \param norm the Normaliser to use.
 * \param config the Config class containing all information.
 */
void printXML(std::ostream& os, std::vector<Session>& sessions, DataTools::Normaliser& norm,
              const Config& config);

/**Print the entire network structure. Basically prints all useful
 * information about the NeuralNetHack session i.e. each trained
 * network and it's architechture and weights.
 * \param os the stream to write to.
 * \param ensemble the ensemble to print.
 * \param norm the Normaliser to use.
 * \param config the Config class containing all information.
 */
void printXML(std::ostream& os, Ensemble& ensemble, DataTools::Normaliser& norm,
              const Config& config);
} // namespace PrintUtils
} // namespace NeuralNetHack

#endif
