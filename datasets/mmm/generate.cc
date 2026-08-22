// Generator for the synthetic MMM dataset in this directory. The data
// process is byte-identical to examples/mmm_boxed.cc (same seed, same
// nnh::rand call order): 50 insertion types = 10 media x 5 messages,
// weekly over 3 years, 5 carryover regimes (four geometric decays plus
// one delayed peak), per-regime Hill saturation (one S-shaped), and
// covariates (annual seasonality, unemployment random walk, linear
// trend). See README.md here for the column layout and ground truth.
//
// Build & run from the repo root:
//   g++ -std=c++23 -O2 -Ineuralnethack datasets/mmm/generate.cc \
//       build/libneuralnethack.a -o /tmp/gen_mmm && (cd datasets/mmm && /tmp/gen_mmm)
//
// Writes: mmm.raw.tab, mmm.trn.tab, mmm.tst.tab

#include "Random.hh"

#include <cmath>
#include <cstdio>
#include <vector>

using uint = unsigned int;

namespace {

constexpr uint MEDIA = 10, MSGS = 5;
constexpr uint C = MEDIA * MSGS; // 50 insertion types
constexpr uint L = 13;           // one quarter of weekly lags
constexpr uint T = 156 + L - 1;  // enough weeks for 156 usable rows

uint regimeOf(uint media) { return media / 2; }

std::vector<double> trueKernel(uint r) {
	std::vector<double> w(L);
	if (r < 4) {
		const double lambda[4] = {0.05, 0.30, 0.60, 0.85};
		for (uint l = 0; l < L; ++l)
			w[l] = std::pow(lambda[r], l);
	} else {
		for (uint l = 0; l < L; ++l)
			w[l] = std::pow(l + 1.0, 2.0) * std::exp(-(l + 1.0) / 3.0); // peak ~ week 5
	}
	double S = 0;
	for (auto v : w)
		S += v;
	for (auto& v : w)
		v /= S;
	return w;
}

// Spends are in currency units (~10k weekly scale, like a real MMM);
// Hill half-saturations live on the adstocked-spend scale accordingly.
constexpr double SPEND_SCALE = 10000.0;
const double trueHalf[5] = {0.6 * SPEND_SCALE, 0.8 * SPEND_SCALE, 1.0 * SPEND_SCALE,
                            1.2 * SPEND_SCALE, 0.9 * SPEND_SCALE};
const double trueExp[5] = {1.0, 1.0, 1.0, 2.0, 1.0};

double hillFn(double a, double s, double n) {
	const double an = std::pow(a, n), sn = std::pow(s, n);
	return an / (an + sn);
}

} // namespace

int main() {
	nnh::rand::seed(4711);

	std::vector<std::vector<double>> spend(C, std::vector<double>(T, 0.0));
	for (uint c = 0; c < C; ++c) {
		bool on = false;
		for (uint t = 0; t < T; ++t) {
			if (nnh::rand::uniform() < 0.18) on = !on;
			spend[c][t] = on ? SPEND_SCALE * (1.0 + 2.0 * nnh::rand::uniform()) : 0.0;
		}
	}

	std::vector<double> unemployment(T);
	double u = 0.0;
	for (uint t = 0; t < T; ++t) {
		u = 0.98 * u + 0.1 * (2.0 * nnh::rand::uniform() - 1.0);
		unemployment[t] = u;
	}

	const double msgBeta[MSGS] = {0.6, 0.8, 1.0, 1.2, 1.4};
	std::vector<std::vector<double>> tk(5);
	for (uint r = 0; r < 5; ++r)
		tk[r] = trueKernel(r);

	std::vector<double> season(T), trend(T), sales(T, 0.0);
	for (uint t = 0; t < T; ++t) {
		season[t] = std::sin(2.0 * M_PI * (t % 52) / 52.0);
		trend[t] = static_cast<double>(t) / T;
	}

	// Windowed rows (channel-major lags, lag 0 = current week), matching
	// examples/mmm_boxed.cc exactly, split chronologically 80/20.
	const uint n = T - (L - 1);
	const uint nTrn = (n * 4) / 5;
	FILE* trn = std::fopen("mmm.trn.tab", "w");
	FILE* tst = std::fopen("mmm.tst.tab", "w");
	for (uint t = L - 1; t < T; ++t) {
		FILE* out = (t - (L - 1) < nTrn) ? trn : tst;
		double s = 2.0; // base
		for (uint c = 0; c < C; ++c) {
			const uint media = c / MSGS, msg = c % MSGS, r = regimeOf(media);
			double a = 0;
			for (uint l = 0; l < L; ++l) {
				const double x = spend[c][t - l];
				std::fprintf(out, "%.6g\t", x);
				a += tk[r][l] * x;
			}
			s += 0.8 * msgBeta[msg] * hillFn(a, trueHalf[r], trueExp[r]);
		}
		s += 0.5 * season[t] - 0.8 * unemployment[t] + 1.0 * trend[t];
		s += 0.03 * (2.0 * nnh::rand::uniform() - 1.0);
		sales[t] = s;
		std::fprintf(out, "%.6g\t%.6g\t%.6g\t%.6g\n", season[t], unemployment[t], trend[t], s);
	}
	std::fclose(trn);
	std::fclose(tst);

	// Raw weekly series: week, 50 spends, season, unemployment, trend,
	// sales (sales 0 for the first L-1 warmup weeks, which have no
	// complete lag window).
	FILE* raw = std::fopen("mmm.raw.tab", "w");
	for (uint t = 0; t < T; ++t) {
		std::fprintf(raw, "%u\t", t);
		for (uint c = 0; c < C; ++c)
			std::fprintf(raw, "%.6g\t", spend[c][t]);
		std::fprintf(raw, "%.6g\t%.6g\t%.6g\t%.6g\n", season[t], unemployment[t], trend[t],
		             sales[t]);
	}
	std::fclose(raw);

	double sum = 0;
	for (uint t = L - 1; t < T; ++t)
		sum += sales[t];
	std::printf("rows: %u train, %u test; sum(sales) = %.6f\n", nTrn, n - nTrn, sum);
	return 0;
}
