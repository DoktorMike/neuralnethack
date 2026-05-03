// Covertype benchmark for mlpack. 54-128-7 with LogSoftMax + NLL.

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>
#include <ensmallen.hpp>

#include "covtype_loader.hh"
#include "timing.hh"

#include <cstddef>
#include <cstdlib>
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

arma::mat toLabelRow(const std::vector<int>& y) {
	arma::mat M(1, y.size());
	for (std::size_t i = 0; i < y.size(); ++i) M(0, i) = y[i];
	return M;
}

} // namespace

int main(int argc, char** argv) {
	const std::string root = (argc > 1) ? argv[1] : "bench/datasets/covtype";
	const int epochs = (argc > 2) ? std::atoi(argv[2]) : 5;
	const int batch = (argc > 3) ? std::atoi(argv[3]) : 32;
	const int trials = (argc > 4) ? std::atoi(argv[4]) : 3;

	bench::CovType trn = bench::loadCovType(root + "/covtype.trn.csv");
	bench::CovType tst = bench::loadCovType(root + "/covtype.tst.csv");
	bench::zNormaliseContinuous(trn, tst);

	arma::mat Xtrn = toMat(trn.X);
	arma::mat Ytrn = toLabelRow(trn.y);
	arma::mat Xtst = toMat(tst.X);

#ifdef _OPENMP
	const int threads = omp_get_max_threads();
#else
	const int threads = 1;
#endif

	for (int t = 0; t < trials; ++t) {
		arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(42 + t));
		FFN<NegativeLogLikelihood, GlorotInitialization> net;
		net.Add<Linear>(128);
		net.Add<TanH>();
		net.Add<Linear>(7);
		net.Add<LogSoftMax>();

		const std::size_t batches_per_epoch =
		    (Xtrn.n_cols + static_cast<std::size_t>(batch) - 1) / static_cast<std::size_t>(batch);
		ens::Adam optimiser(0.01, batch, 0.9, 0.999, 1e-8,
		                    static_cast<std::size_t>(epochs) * batches_per_epoch, 1e-8, true);

		const auto t0 = bench::clk::now();
		net.Train(Xtrn, Ytrn, optimiser);
		const auto t1 = bench::clk::now();
		const double train_s = bench::seconds(t0, t1);

		arma::mat preds;
		std::size_t correct = 0;
		const auto i0 = bench::clk::now();
		net.Predict(Xtst, preds);
		for (std::size_t i = 0; i < tst.y.size(); ++i) {
			arma::uword a = 0;
			double best = preds(0, i);
			for (arma::uword k = 1; k < preds.n_rows; ++k)
				if (preds(k, i) > best) {
					best = preds(k, i);
					a = k;
				}
			if (static_cast<int>(a) == tst.y[i]) ++correct;
		}
		const auto i1 = bench::clk::now();
		const double infer_us = 1e6 * bench::seconds(i0, i1) / tst.y.size();
		const double acc = static_cast<double>(correct) / tst.y.size();

		bench::emit("mlpack", "covtype", "54-128-7", epochs, batch, threads, "openblas", t + 1,
		            train_s, infer_us, acc);
	}
	return 0;
}
