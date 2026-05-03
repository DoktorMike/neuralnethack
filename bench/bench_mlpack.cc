// Pima benchmark for mlpack. FFN<MeanSquaredError, ...> with the same
// 8-32-1 architecture, Adam optimiser, same epochs/batch.
//
// Build deps: mlpack >= 4.7, armadillo, ensmallen, OpenBLAS.
// On Arch: paru -S mlpack (PKGBUILD must use CMAKE_CXX_STANDARD=17).

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <ensmallen.hpp>

#include "pima_loader.hh"
#include "timing.hh"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mlpack;

namespace {

arma::mat toMat(const std::vector<std::vector<double>>& X) {
	arma::mat M(X.front().size(), X.size());
	for (std::size_t i = 0; i < X.size(); ++i)
		for (std::size_t j = 0; j < X.front().size(); ++j) M(j, i) = X[i][j];
	return M;
}

arma::mat toRow(const std::vector<int>& y) {
	arma::mat M(1, y.size());
	for (std::size_t i = 0; i < y.size(); ++i) M(0, i) = y[i];
	return M;
}

} // namespace

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "datasets/pima";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 100;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 10;

	bench::Pima trn = bench::loadPima(root + "/pima.trn.tab");
	bench::Pima tst = bench::loadPima(root + "/pima.tst.tab");
	bench::zNormalise(trn, tst);

	arma::mat Xtrn = toMat(trn.X);
	arma::mat Ytrn = toRow(trn.y);
	arma::mat Xtst = toMat(tst.X);

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif

	for (int t = 0; t < trials; ++t) {
		arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(42 + t));
		mlpack::FFN<mlpack::MeanSquaredError, mlpack::GlorotInitialization> net;
		net.Add<mlpack::Linear>(32);
		net.Add<mlpack::TanH>();
		net.Add<mlpack::Linear>(1);
		net.Add<mlpack::Sigmoid>();

		// ensmallen counts maxIterations in batch steps, not in samples,
		// so for `epochs` passes over the data we need
		// epochs * ceil(n_train / batch) iterations -- not epochs *
		// n_train, which would silently train ~n_train / batch times
		// longer than the other harnesses and inflate accuracy.
		const std::size_t batches_per_epoch =
		    (Xtrn.n_cols + static_cast<std::size_t>(batch) - 1) / static_cast<std::size_t>(batch);
		ens::Adam optimiser(0.01, batch, 0.9, 0.999, 1e-8,
		                    static_cast<std::size_t>(epochs) * batches_per_epoch, 1e-8, true);

		const auto t0 = bench::clk::now();
		net.Train(Xtrn, Ytrn, optimiser);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		arma::mat preds;
		int correct = 0;
		const int reps = 20;
		const auto i0 = bench::clk::now();
		for (int r = 0; r < reps; ++r) {
			net.Predict(Xtst, preds);
			if (r == 0)
				for (std::size_t i = 0; i < tst.y.size(); ++i)
					if ((preds(0, i) >= 0.5 ? 1 : 0) == tst.y[i]) ++correct;
		}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / (reps * tst.y.size());
		const double acc = double(correct) / tst.y.size();

		bench::emit("mlpack", "pima", "8-32-1", epochs, batch, threads, "openblas", t + 1,
		            train_s, infer_us, acc);
	}
	return 0;
}
