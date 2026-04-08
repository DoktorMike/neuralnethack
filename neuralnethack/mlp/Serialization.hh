#ifndef __Serialization_hh__
#define __Serialization_hh__

#include "Mlp.hh"
#include "../Ensemble.hh"

#include <memory>
#include <string>
#include <iostream>

namespace MultiLayerPerceptron {
/**Save an Mlp in binary format.
 * \param mlp the Mlp to save.
 * \param os the output stream.
 */
void saveMlpBinary(const Mlp& mlp, std::ostream& os);

/**Load an Mlp from binary format.
 * \param is the input stream.
 * \return a unique_ptr to the loaded Mlp.
 */
std::unique_ptr<Mlp> loadMlpBinary(std::istream& is);

/**Save an Mlp to a file.
 * \param mlp the Mlp to save.
 * \param path the file path.
 */
void saveMlpBinary(const Mlp& mlp, const std::string& path);

/**Load an Mlp from a file.
 * \param path the file path.
 * \return a unique_ptr to the loaded Mlp.
 */
std::unique_ptr<Mlp> loadMlpBinary(const std::string& path);

/**Save an Ensemble in binary format.
 * \param ens the Ensemble to save.
 * \param os the output stream.
 */
void saveEnsembleBinary(const NeuralNetHack::Ensemble& ens, std::ostream& os);

/**Load an Ensemble from binary format.
 * \param is the input stream.
 * \return a unique_ptr to the loaded Ensemble.
 */
std::unique_ptr<NeuralNetHack::Ensemble> loadEnsembleBinary(std::istream& is);

/**Save an Ensemble to a file.
 * \param ens the Ensemble to save.
 * \param path the file path.
 */
void saveEnsembleBinary(const NeuralNetHack::Ensemble& ens, const std::string& path);

/**Load an Ensemble from a file.
 * \param path the file path.
 * \return a unique_ptr to the loaded Ensemble.
 */
std::unique_ptr<NeuralNetHack::Ensemble> loadEnsembleBinary(const std::string& path);
} // namespace MultiLayerPerceptron

#endif
