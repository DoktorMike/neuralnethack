#include <evaltools/ConfusionMatrix.hh>
#include <evaltools/EvalTools.hh>

#include <cmath>
#include <iostream>
#include <vector>

using EvalTools::ConfusionMatrix;
using EvalTools::ErrorMeasures::accuracy;
using EvalTools::ErrorMeasures::balancedAccuracy;
using EvalTools::ErrorMeasures::f1;
using EvalTools::ErrorMeasures::macroF1;
using EvalTools::ErrorMeasures::macroPrecision;
using EvalTools::ErrorMeasures::macroRecall;
using EvalTools::ErrorMeasures::mcc;
using EvalTools::ErrorMeasures::precision;
using EvalTools::ErrorMeasures::recall;
using std::vector;

static int fails = 0;

static bool nearly(double a, double b, double tol = 1e-9) {
	return std::fabs(a - b) <= tol;
}

#define EXPECT(cond, label)                                                                        \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::cerr << "FAIL: " << label << " (" << __FILE__ << ":" << __LINE__ << ")"           \
			          << std::endl;                                                                \
			++fails;                                                                               \
		}                                                                                          \
	} while (0)

int main() {
	// Binary, perfect predictions.
	{
		vector<double> y = {0.05, 0.9, 0.1, 0.95, 0.2, 0.8};
		vector<uint> t = {0, 1, 0, 1, 0, 1};
		auto cm = ConfusionMatrix::fromBinary(y, t);
		EXPECT(cm.tp() == 3, "perfect tp");
		EXPECT(cm.tn() == 3, "perfect tn");
		EXPECT(cm.fp() == 0 && cm.fn() == 0, "perfect off-diag");
		EXPECT(nearly(accuracy(cm), 1.0), "perfect accuracy");
		EXPECT(nearly(precision(cm), 1.0), "perfect precision");
		EXPECT(nearly(recall(cm), 1.0), "perfect recall");
		EXPECT(nearly(f1(cm), 1.0), "perfect F1");
		EXPECT(nearly(mcc(cm), 1.0), "perfect MCC");
		EXPECT(nearly(balancedAccuracy(cm), 1.0), "perfect BA");
	}

	// Binary, anti-correlated.
	{
		vector<double> y = {0.9, 0.1, 0.95, 0.05};
		vector<uint> t = {0, 1, 0, 1};
		auto cm = ConfusionMatrix::fromBinary(y, t);
		EXPECT(cm.tp() == 0 && cm.tn() == 0, "anti tp/tn = 0");
		EXPECT(cm.fp() == 2 && cm.fn() == 2, "anti fp/fn = 2");
		EXPECT(nearly(accuracy(cm), 0.0), "anti accuracy = 0");
		EXPECT(nearly(mcc(cm), -1.0), "anti MCC = -1");
	}

	// Binary, imbalanced realistic case.
	// 100 negatives, 10 positives. Predict 95 of the negatives correctly,
	// catch 8 of 10 positives but with 5 false alarms.
	// TP=8, FN=2, FP=5, TN=95. Total=110.
	// precision = 8/13 ≈ 0.6154, recall = 8/10 = 0.8, accuracy = 103/110 ≈ 0.9364.
	{
		ConfusionMatrix cm(2);
		for (uint i = 0; i < 95; ++i)
			cm.add(0, 0); // TN
		for (uint i = 0; i < 5; ++i)
			cm.add(0, 1); // FP
		for (uint i = 0; i < 8; ++i)
			cm.add(1, 1); // TP
		for (uint i = 0; i < 2; ++i)
			cm.add(1, 0); // FN
		EXPECT(nearly(precision(cm), 8.0 / 13.0), "imbalanced precision");
		EXPECT(nearly(recall(cm), 0.8), "imbalanced recall");
		EXPECT(nearly(accuracy(cm), 103.0 / 110.0), "imbalanced accuracy");
		// F1 = 2 * P * R / (P + R)
		double p = 8.0 / 13.0, r = 0.8;
		EXPECT(nearly(f1(cm), 2.0 * p * r / (p + r)), "imbalanced F1");
	}

	// Multi-class 3x3, perfect.
	{
		vector<vector<double>> y = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 0, 1}};
		vector<vector<double>> t = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 0, 1}};
		auto cm = ConfusionMatrix::fromMulticlass(y, t);
		EXPECT(cm.nClasses() == 3, "3 classes detected");
		EXPECT(nearly(accuracy(cm), 1.0), "multiclass perfect accuracy");
		EXPECT(nearly(macroPrecision(cm), 1.0), "macro P perfect");
		EXPECT(nearly(macroRecall(cm), 1.0), "macro R perfect");
		EXPECT(nearly(macroF1(cm), 1.0), "macro F1 perfect");
	}

	// Multi-class, partial errors. Build matrix directly:
	//        pred 0  pred 1  pred 2
	// act 0:    5      1       0
	// act 1:    0      4       2
	// act 2:    1      0       3
	// total = 16, correct = 12 → accuracy = 0.75
	{
		ConfusionMatrix cm(3);
		for (uint i = 0; i < 5; ++i)
			cm.add(0, 0);
		cm.add(0, 1);
		for (uint i = 0; i < 4; ++i)
			cm.add(1, 1);
		cm.add(1, 2);
		cm.add(1, 2);
		cm.add(2, 0);
		for (uint i = 0; i < 3; ++i)
			cm.add(2, 2);

		EXPECT(cm.total() == 16, "total = 16");
		EXPECT(cm.correct() == 12, "correct = 12");
		EXPECT(nearly(accuracy(cm), 12.0 / 16.0), "multiclass accuracy");

		// recall(0) = 5/6, recall(1) = 4/6, recall(2) = 3/4.
		EXPECT(nearly(recall(cm, 0), 5.0 / 6.0), "recall class 0");
		EXPECT(nearly(recall(cm, 1), 4.0 / 6.0), "recall class 1");
		EXPECT(nearly(recall(cm, 2), 3.0 / 4.0), "recall class 2");

		// precision(0) = 5/6, precision(1) = 4/5, precision(2) = 3/5.
		EXPECT(nearly(precision(cm, 0), 5.0 / 6.0), "precision class 0");
		EXPECT(nearly(precision(cm, 1), 4.0 / 5.0), "precision class 1");
		EXPECT(nearly(precision(cm, 2), 3.0 / 5.0), "precision class 2");

		// MCC must be NaN for non-binary.
		EXPECT(std::isnan(mcc(cm)), "MCC NaN for 3-class");
	}

	// Edge: precision NaN when class never predicted.
	{
		ConfusionMatrix cm(2);
		cm.add(0, 0);
		cm.add(1, 0);
		cm.add(1, 0); // nothing ever predicted as 1
		EXPECT(std::isnan(precision(cm, 1)), "precision NaN when class never predicted");
		// Recall is defined: class 1 appeared twice, both missed → 0.
		EXPECT(nearly(recall(cm, 1), 0.0), "recall = 0 when all missed");
		// F1 NaN because precision is NaN.
		EXPECT(std::isnan(f1(cm, 1)), "F1 NaN follows precision NaN");
	}

	// Edge: recall NaN when class never appears in actuals.
	{
		ConfusionMatrix cm(2);
		cm.add(0, 0);
		cm.add(0, 1); // no actual class 1
		EXPECT(std::isnan(recall(cm, 1)), "recall NaN when class absent");
	}

	// Edge: empty matrix.
	{
		ConfusionMatrix cm(2);
		EXPECT(std::isnan(accuracy(cm)), "accuracy NaN for empty");
		EXPECT(std::isnan(mcc(cm)), "MCC NaN for empty");
	}

	if (fails == 0) std::cout << "All ConfusionMatrix tests passed." << std::endl;
	return fails == 0 ? 0 : 1;
}
