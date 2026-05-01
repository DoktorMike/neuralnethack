#include <evaltools/EvalTools.hh>

#include <cmath>
#include <iostream>
#include <vector>

using EvalTools::ErrorMeasures::mae;
using EvalTools::ErrorMeasures::mape;
using EvalTools::ErrorMeasures::r2;
using EvalTools::ErrorMeasures::rmse;
using EvalTools::ErrorMeasures::smape;
using std::vector;

static bool nearly(double a, double b, double tol = 1e-9) {
	return std::fabs(a - b) <= tol;
}

static int fails = 0;

#define EXPECT(cond, label)                                                               \
	do {                                                                                  \
		if (!(cond)) {                                                                    \
			std::cerr << "FAIL: " << label << " (" << __FILE__ << ":" << __LINE__ << ")"  \
			          << std::endl;                                                       \
			++fails;                                                                      \
		}                                                                                 \
	} while (0)

int main() {
	// Perfect predictions: every metric is 0 (or 1 for R²).
	{
		vector<double> y = {1.0, 2.0, 3.0, 4.0};
		vector<double> t = {1.0, 2.0, 3.0, 4.0};
		EXPECT(nearly(mae(y, t), 0.0), "MAE perfect = 0");
		EXPECT(nearly(rmse(y, t), 0.0), "RMSE perfect = 0");
		EXPECT(nearly(mape(y, t), 0.0), "MAPE perfect = 0");
		EXPECT(nearly(smape(y, t), 0.0), "sMAPE perfect = 0");
		EXPECT(nearly(r2(y, t), 1.0), "R2 perfect = 1");
	}

	// Constant offset of 1: MAE=1, RMSE=1, R² < 0 because residuals exceed
	// what predicting the mean would give.
	{
		vector<double> t = {1.0, 2.0, 3.0, 4.0, 5.0};
		vector<double> y = {2.0, 3.0, 4.0, 5.0, 6.0};
		EXPECT(nearly(mae(y, t), 1.0), "MAE constant offset 1");
		EXPECT(nearly(rmse(y, t), 1.0), "RMSE constant offset 1");
		// SS_res = 5, SS_tot = sum((t - 3)^2) = 4 + 1 + 0 + 1 + 4 = 10. R² = 1 - 5/10 = 0.5.
		EXPECT(nearly(r2(y, t), 0.5), "R2 constant offset 1 = 0.5");
	}

	// MAPE skips zero targets; sMAPE handles them via the symmetric denominator.
	{
		vector<double> t = {0.0, 100.0, 200.0};
		vector<double> y = {5.0, 110.0, 180.0};
		// MAPE: skip i=0. (10/100 + 20/200) / 2 * 100 = (0.1 + 0.1)/2 * 100 = 10.
		EXPECT(nearly(mape(y, t), 10.0), "MAPE 10%");
		// sMAPE: i=0: 2*5 / (0+5) = 2. i=1: 2*10/210 ≈ 0.0952. i=2: 2*20/380 ≈ 0.1053.
		// mean = (2 + 0.0952 + 0.1053)/3 = 0.7335. * 100 = 73.35.
		double sm = smape(y, t);
		EXPECT(sm > 70.0 && sm < 76.0, "sMAPE in expected band");
	}

	// MAPE on all-zero targets: NaN.
	{
		vector<double> t = {0.0, 0.0, 0.0};
		vector<double> y = {1.0, 2.0, 3.0};
		EXPECT(std::isnan(mape(y, t)), "MAPE all-zero targets = NaN");
	}

	// R² on constant target is undefined.
	{
		vector<double> t = {7.0, 7.0, 7.0};
		vector<double> y = {7.0, 7.0, 7.0};
		EXPECT(std::isnan(r2(y, t)), "R2 constant target = NaN");
	}

	// Predicting the mean gives R² = 0.
	{
		vector<double> t = {1.0, 2.0, 3.0, 4.0, 5.0};
		vector<double> y(5, 3.0); // mean of t
		EXPECT(nearly(r2(y, t), 0.0), "R2 predict-mean = 0");
	}

	// Worse-than-mean fit gives R² < 0.
	{
		vector<double> t = {1.0, 2.0, 3.0, 4.0, 5.0};
		vector<double> y = {5.0, 4.0, 3.0, 2.0, 1.0}; // anti-correlated
		EXPECT(r2(y, t) < 0.0, "R2 anti-correlated < 0");
	}

	// RMSE >= MAE always (Cauchy-Schwarz).
	{
		vector<double> t = {1.0, 5.0, 2.0, 8.0, 3.0};
		vector<double> y = {1.5, 4.0, 3.0, 6.0, 2.5};
		EXPECT(rmse(y, t) >= mae(y, t) - 1e-12, "RMSE >= MAE");
	}

	if (fails == 0) std::cout << "All regression metric tests passed." << std::endl;
	return fails == 0 ? 0 : 1;
}
