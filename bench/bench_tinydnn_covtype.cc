// Covertype benchmark for tiny-dnn. 54-128-7 softmax via tinydnn's
// `softmax_layer` + `cross_entropy_multiclass` loss.

#include "tiny_dnn/tiny_dnn.h"

#include "covtype_loader.hh"
#include "timing.hh"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tiny_dnn;

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "datasets/covtype";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 5;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 3;

	bench::CovType trn = bench::loadCovType(root + "/covtype.trn.csv");
	bench::CovType tst = bench::loadCovType(root + "/covtype.tst.csv");
	bench::zNormaliseContinuous(trn, tst);

	std::vector<vec_t> Xtrn, Xtst;
	std::vector<label_t> Ytrn;
	for (auto& r : trn.X) Xtrn.push_back(vec_t(r.begin(), r.end()));
	for (auto& r : tst.X) Xtst.push_back(vec_t(r.begin(), r.end()));
	for (int v : trn.y) Ytrn.push_back(static_cast<label_t>(v));

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif

	for (int t = 0; t < trials; ++t) {
		set_random_seed(static_cast<unsigned int>(42 + t));
		network<sequential> net;
		net << layers::fc(54, 128) << tanh_layer() << layers::fc(128, 7) << softmax_layer();
		adam optimiser;
		optimiser.alpha = float_t(0.01);

		const auto t0 = bench::clk::now();
		net.train<cross_entropy_multiclass>(optimiser, Xtrn, Ytrn, batch, epochs);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		std::size_t correct = 0;
		const auto i0 = bench::clk::now();
		for (std::size_t i = 0; i < Xtst.size(); ++i) {
			const auto y = net.predict(Xtst[i]);
			std::size_t a = 0;
			for (std::size_t k = 1; k < y.size(); ++k)
				if (y[k] > y[a]) a = k;
			if (static_cast<int>(a) == tst.y[i]) ++correct;
		}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / Xtst.size();
		const double acc = static_cast<double>(correct) / Xtst.size();

		bench::emit("tiny-dnn", "covtype", "54-128-7", epochs, batch, threads, "none", t + 1,
		            train_s, infer_us, acc);
	}
	return 0;
}
