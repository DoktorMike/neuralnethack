#include "Serialization.hh"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <cstring>

using namespace MultiLayerPerceptron;
using namespace std;

static void writeUint32(ostream& os, uint32_t v) {
	os.write(reinterpret_cast<const char*>(&v), 4);
}
static uint32_t readUint32(istream& is) {
	uint32_t v;
	is.read(reinterpret_cast<char*>(&v), 4);
	return v;
}
static void writeDouble(ostream& os, double v) {
	os.write(reinterpret_cast<const char*>(&v), 8);
}
static double readDouble(istream& is) {
	double v;
	is.read(reinterpret_cast<char*>(&v), 8);
	return v;
}

void MultiLayerPerceptron::saveMlpBinary(const Mlp& mlp, ostream& os) {
	// Magic: NNH1 = plain MLP, NNH2 = per-channel adstock stage, NNH3 =
	// boxed adstock stage (meta block between softmax flag and weights;
	// the weight vector then carries the adstock params at its tail, as
	// Mlp::weights() emits).
	const Adstock* ads = mlp.adstock();
	os.write(!ads ? "NNH1" : (ads->boxed() ? "NNH3" : "NNH2"), 4);

	// Architecture
	// Need const access to arch — use const_cast since arch() isn't const-qualified
	Mlp& mref = const_cast<Mlp&>(mlp);
	vector<uint>& arch = mref.arch();
	writeUint32(os, arch.size());
	for (auto a : arch)
		writeUint32(os, a);

	// Types
	vector<string>& types = mref.types();
	writeUint32(os, types.size());
	for (auto& t : types) {
		writeUint32(os, t.size());
		os.write(t.data(), t.size());
	}

	// Softmax
	uint8_t sm = mref.softmax() ? 1 : 0;
	os.write(reinterpret_cast<const char*>(&sm), 1);

	// Adstock meta (NNH2/NNH3)
	if (ads) {
		writeUint32(os, ads->nChannels());
		writeUint32(os, ads->nLags());
		writeUint32(os, ads->nPassthrough());
		const string tag = kernelToTag(ads->kernel());
		writeUint32(os, tag.size());
		os.write(tag.data(), tag.size());
		if (ads->boxed()) {
			writeUint32(os, ads->nBoxes());
			uint8_t sat = ads->saturation() == Adstock::Saturation::Hill ? 1 : 0;
			os.write(reinterpret_cast<const char*>(&sat), 1);
			writeDouble(os, ads->temperature());
		}
	}

	// Weights
	vector<double> w = mlp.weights();
	writeUint32(os, w.size());
	os.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(double));
}

unique_ptr<Mlp> MultiLayerPerceptron::loadMlpBinary(istream& is) {
	// Magic
	char magic[4];
	is.read(magic, 4);
	const bool hasAdstock = memcmp(magic, "NNH2", 4) == 0 || memcmp(magic, "NNH3", 4) == 0;
	const bool boxedAdstock = memcmp(magic, "NNH3", 4) == 0;
	if (!hasAdstock && memcmp(magic, "NNH1", 4) != 0)
		throw runtime_error("Invalid MLP binary format: bad magic");

	// Architecture
	uint32_t archSize = readUint32(is);
	vector<uint> arch(archSize);
	for (uint32_t i = 0; i < archSize; ++i)
		arch[i] = readUint32(is);

	// Types
	uint32_t typesSize = readUint32(is);
	vector<string> types(typesSize);
	for (uint32_t i = 0; i < typesSize; ++i) {
		uint32_t len = readUint32(is);
		types[i].resize(len);
		is.read(&types[i][0], len);
	}

	// Softmax
	uint8_t sm;
	is.read(reinterpret_cast<char*>(&sm), 1);

	// Create Mlp (this randomizes weights)
	auto mlp = make_unique<Mlp>(arch, types, sm != 0);

	// Adstock meta (NNH2/NNH3); params arrive with the weight vector
	if (hasAdstock) {
		uint32_t channels = readUint32(is);
		uint32_t lags = readUint32(is);
		uint32_t pass = readUint32(is);
		uint32_t len = readUint32(is);
		string tag(len, '\0');
		is.read(&tag[0], len);
		if (boxedAdstock) {
			uint32_t boxes = readUint32(is);
			uint8_t sat;
			is.read(reinterpret_cast<char*>(&sat), 1);
			double tau = readDouble(is);
			Adstock a(channels, lags, pass, kernelFromTag(tag), boxes,
			          sat ? Adstock::Saturation::Hill : Adstock::Saturation::None);
			a.temperature(tau);
			mlp->adstock(a);
		} else {
			mlp->adstock(Adstock(channels, lags, pass, kernelFromTag(tag)));
		}
	}

	// Read and set weights
	uint32_t nWeights = readUint32(is);
	vector<double> w(nWeights);
	is.read(reinterpret_cast<char*>(w.data()), nWeights * sizeof(double));
	mlp->weights(w);

	return mlp;
}

void MultiLayerPerceptron::saveMlpBinary(const Mlp& mlp, const string& path) {
	ofstream os(path, ios::binary);
	if (!os) throw runtime_error("Cannot open file for writing: " + path);
	saveMlpBinary(mlp, os);
}

unique_ptr<Mlp> MultiLayerPerceptron::loadMlpBinary(const string& path) {
	ifstream is(path, ios::binary);
	if (!is) throw runtime_error("Cannot open file for reading: " + path);
	return loadMlpBinary(is);
}

void MultiLayerPerceptron::saveEnsembleBinary(const NeuralNetHack::Ensemble& ens, ostream& os) {
	os.write("ENS1", 4);
	writeUint32(os, ens.size());
	for (uint i = 0; i < ens.size(); ++i) {
		writeDouble(os, ens.scale(i));
		saveMlpBinary(ens.mlp(i), os);
	}
}

unique_ptr<NeuralNetHack::Ensemble> MultiLayerPerceptron::loadEnsembleBinary(istream& is) {
	char magic[4];
	is.read(magic, 4);
	if (memcmp(magic, "ENS1", 4) != 0)
		throw runtime_error("Invalid Ensemble binary format: bad magic");

	uint32_t n = readUint32(is);
	auto ens = make_unique<NeuralNetHack::Ensemble>();
	for (uint32_t i = 0; i < n; ++i) {
		double scale = readDouble(is);
		auto mlp = loadMlpBinary(is);
		ens->addMlp(std::move(mlp), scale);
	}
	return ens;
}

void MultiLayerPerceptron::saveEnsembleBinary(const NeuralNetHack::Ensemble& ens,
                                              const string& path) {
	ofstream os(path, ios::binary);
	if (!os) throw runtime_error("Cannot open file for writing: " + path);
	saveEnsembleBinary(ens, os);
}

unique_ptr<NeuralNetHack::Ensemble> MultiLayerPerceptron::loadEnsembleBinary(const string& path) {
	ifstream is(path, ios::binary);
	if (!is) throw runtime_error("Cannot open file for reading: " + path);
	return loadEnsembleBinary(is);
}
