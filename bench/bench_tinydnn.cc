// Pima benchmark for tiny-dnn. Same architecture (8-32-1, tansig + logsig),
// same optimizer (Adam, lr=0.01), same epochs/batch as bench_nnh.cc.

#include "tiny_dnn/tiny_dnn.h"

#include "pima_loader.hh"
#include "timing.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tiny_dnn;

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "test/pima-indians-diabetes";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 100;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 10;

	bench::Pima trn = bench::loadPima(root + "/pima.trn.tab");
	bench::Pima tst = bench::loadPima(root + "/pima.tst.tab");
	bench::zNormalise(trn, tst);

	std::vector<vec_t> Xtrn, Xtst;
	std::vector<vec_t> Ytrn;
	for (auto& r : trn.X) Xtrn.push_back(vec_t(r.begin(), r.end()));
	for (auto& r : tst.X) Xtst.push_back(vec_t(r.begin(), r.end()));
	for (int v : trn.y) Ytrn.push_back(vec_t{static_cast<float_t>(v)});

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif

	for (int t = 0; t < trials; ++t) {
		set_random_seed(static_cast<unsigned int>(42 + t));
		network<sequential> net;
		net << layers::fc(8, 32) << tanh_layer() << layers::fc(32, 1) << sigmoid_layer();
		adam optimiser;
		optimiser.alpha = float_t(0.01);

		const auto t0 = bench::clk::now();
		net.fit<mse>(optimiser, Xtrn, Ytrn, batch, epochs);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		int correct = 0;
		const int reps = 20;
		const auto i0 = bench::clk::now();
		for (int r = 0; r < reps; ++r)
			for (std::size_t i = 0; i < Xtst.size(); ++i) {
				const auto y = net.predict(Xtst[i]);
				if (r == 0 && (y[0] >= float_t(0.5) ? 1 : 0) == tst.y[i]) ++correct;
			}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / (reps * Xtst.size());
		const double acc = double(correct) / Xtst.size();

		bench::emit("tiny-dnn", "pima", "8-32-1", epochs, batch, threads, "none", t + 1, train_s,
		            infer_us, acc);
	}
	return 0;
}
