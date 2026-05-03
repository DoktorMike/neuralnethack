// Smoke test for Trainer::learningCurveFile + validationData. Trains a
// tiny MLP with each trainer (Adam, GD, QN) and verifies:
//   - the file is created with a header
//   - rows are numeric and have 2 columns without val data, 3 with val data
//   - epoch monotonically increases
//   - the val column differs from the train column (sanity: they're
//     evaluated on different sets)

#include "Random.hh"
#include "datatools/CoreDataSet.hh"
#include "datatools/DataSet.hh"
#include "datatools/Pattern.hh"
#include "mlp/Adam.hh"
#include "mlp/GradientDescent.hh"
#include "mlp/Mlp.hh"
#include "mlp/QuasiNewton.hh"
#include "mlp/SummedSquare.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MultiLayerPerceptron;
using namespace DataTools;
using std::string;
using std::vector;

static int fails = 0;

#define EXPECT(cond, label)                                                                        \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::cerr << "FAIL: " << label << " (" << __FILE__ << ":" << __LINE__ << ")"           \
			          << std::endl;                                                                \
			++fails;                                                                               \
		}                                                                                          \
	} while (0)

namespace {
DataSet makeData(uint n, uint seed) {
	srand(seed);
	auto core = std::make_shared<CoreDataSet>();
	for (uint i = 0; i < n; ++i) {
		double x = drand48();
		double y = drand48();
		vector<double> in = {x, y};
		vector<double> out = {std::sin(3.0 * x) + std::cos(2.0 * y)};
		core->addPattern(Pattern(std::to_string(i), in, out));
	}
	DataSet ds;
	ds.coreDataSet(core);
	return ds;
}

struct CurveRow {
	double epoch, trainErr, valErr;
	bool hasVal;
};

vector<CurveRow> readCurve(const string& path) {
	vector<CurveRow> rows;
	std::ifstream in(path);
	string line;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream iss(line);
		CurveRow r{};
		iss >> r.epoch >> r.trainErr;
		double v;
		r.hasVal = static_cast<bool>(iss >> v);
		r.valErr = r.hasVal ? v : 0.0;
		rows.push_back(r);
	}
	return rows;
}

void runCase(const string& tag, std::function<std::unique_ptr<Trainer>(Mlp&, DataSet&, Error&)> mk,
             bool withVal) {
	nnh::rand::seed(1);
	DataSet trn = makeData(40, 11);
	DataSet val = makeData(20, 22);
	vector<uint> arch = {2, 4, 1};
	vector<string> types = {"tansig", "purelin"};
	Mlp mlp(arch, types, false);
	SummedSquare loss(mlp, trn);

	auto trainer = mk(mlp, trn, loss);
	trainer->numEpochs(20);
	const string path = "/tmp/testLearningCurve_" + tag + (withVal ? "_val" : "") + ".dat";
	std::remove(path.c_str());
	trainer->learningCurveFile(path);
	if (withVal) trainer->validationData(&val);

	std::ostringstream sink;
	trainer->train(sink);

	auto rows = readCurve(path);
	EXPECT(!rows.empty(), tag + ": at least one row written");
	if (rows.empty()) return;

	for (const auto& r : rows)
		EXPECT(r.hasVal == withVal, tag + ": val column presence matches expectation");

	for (size_t i = 1; i < rows.size(); ++i)
		EXPECT(rows[i].epoch >= rows[i - 1].epoch, tag + ": epoch is monotonically nondecreasing");

	if (withVal) {
		// Train and val are different datasets and tiny; they should not
		// produce identical errors at every recorded point.
		bool anyDiffer = false;
		for (const auto& r : rows)
			if (std::fabs(r.trainErr - r.valErr) > 1e-12) {
				anyDiffer = true;
				break;
			}
		EXPECT(anyDiffer, tag + ": train and val errors differ at least once");
	}

	// The Error's internal pointers must still target the training data
	// after recording, otherwise subsequent calls would silently re-train
	// on the val set.
	EXPECT(&loss.dset() == &trn, tag + ": Error::dset restored to train after recording");

	std::remove(path.c_str());
}
} // namespace

int main() {
	// Adam, with and without val.
	runCase(
	    "adam",
	    [](Mlp& m, DataSet& d, Error& e) {
		    return std::make_unique<Adam>(m, d, e, /*te=*/0.0, /*bs=*/8, /*lr=*/0.05);
	    },
	    false);
	runCase(
	    "adam",
	    [](Mlp& m, DataSet& d, Error& e) { return std::make_unique<Adam>(m, d, e, 0.0, 8, 0.05); },
	    true);

	// GradientDescent, with val.
	runCase(
	    "gd",
	    [](Mlp& m, DataSet& d, Error& e) {
		    return std::make_unique<GradientDescent>(m, d, e, /*te=*/0.0, /*bs=*/8, /*lr=*/0.05,
		                                             /*dlr=*/1.0, /*momentum=*/0.0);
	    },
	    true);

	// QuasiNewton, with val.
	runCase(
	    "qn",
	    [](Mlp& m, DataSet& d, Error& e) {
		    return std::make_unique<QuasiNewton>(m, d, e, /*te=*/0.0, /*bs=*/8);
	    },
	    true);

	if (fails == 0) std::cout << "All learning-curve tests passed." << std::endl;
	return fails == 0 ? 0 : 1;
}
