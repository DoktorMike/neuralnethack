#ifndef __EvalTools_hh__
#define __EvalTools_hh__

#include "ConfusionMatrix.hh"
#include "datatools/DataSet.hh"
#include "Ensemble.hh"

#include <vector>

/**This namespace encloses a bunch of functions needed to
 * evaluate the performance of a classifier or a regressor.
 *
 * Classification metrics: crossEntropy, auc, gof, plus ConfusionMatrix-derived
 * accuracy, precision, recall, f1, mcc, balancedAccuracy, and macro variants.
 * Regression metrics: mae, mape, smape, rmse, r2.
 * Generic: summedSquare.
 */
namespace EvalTools {
namespace ErrorMeasures {
/**Measures the CrossEntropyError for an entire DataSet.
 * This is also known as the kullback leibler error measure.
 * This measure is intended to be used with classification
 * problems for two class single output problems. In the
 * special case of one output and two classes the following will be used:
 * \f[E=-\frac{1}{N}\sum_{n}\left(d_n\ln y_n + (1-d_n)\ln (1-y_n)\right)\f]
 * Otherwise we use:
 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
 * \param committee the ensemble of Mlp to estimate the error for.
 * \param data the DataSet to use for error measure.
 * \return the cross entropy error.
 */
double crossEntropy(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Measures the CrossEntropyError for one data point.
 * This is also known as the kullback leibler error measure.
 * This measure is intended to be used with classification
 * problems for two class single output problems. In the
 * special case of one output and two classes the following will be used:
 * \f[E=-\frac{1}{N}\sum_{n}\left(d_n\ln y_n + (1-d_n)\ln (1-y_n)\right)\f]
 * Otherwise we use:
 * \f[E=-\frac{1}{N}\sum_{n}\sum_{i}\left(d_i\ln\left(\frac{y_i}{d_i}\right)\right)\f]
 * \param output the output vector of the classifier.
 * \param target the target.
 * \return the cross entropy error.
 */
double crossEntropy(const std::vector<double>& output, const std::vector<double>& target);

/**Measures the SummedSquareError for an entire DataSet.
 * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
 * \param committee the ensemble of Mlp to estimate the error for.
 * \param data the DataSet to use for error measure.
 * \return the summed square error.
 */
double summedSquare(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Measures the SummedSquareError for an entire DataSet.
 * \f[E=\frac{1}{2N}\sum_{n}\sum_{i}(d_i-y_i)^2\f]
 * \param output the output vector of the classifier.
 * \param target the target.
 * \return the summed square error.
 */
double summedSquare(const std::vector<double>& output, const std::vector<double>& target);

/**Measures the AUC(Area Under Curve) for the ROC(Receicer Operating
 * Characteristics).
 * \param committee the ensemble of Mlp to estimate the error for.
 * \param data the DataSet to use for error measure.
 * \return the area under curve.
 */
double auc(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Measures the Hosmer Lemeshow goodness of fit statistics.
 * \f[\chi ^2 = \sum_{j=1}^{G}\frac{(o_j - n_j \bar{\pi}_j)^2}{ n_j \bar{\pi}_j (1 - \bar{\pi}_j)
 * }\f] Where \f$o_j\f$ is the number of observed positives in bin j, and
 * \f$\bar{\pi}_j\f$ is the mean average predicted value in bin j. G is
 * the number of bins meanwhile \f$n_j\f$ is the number of samples in the
 * bin.This test statistics follow the chi square statistics with a (G-2)
 * deegree of freedom.
 * \param committee the ensemble of Mlp to estimate the error for.
 * \param data the DataSet to use for error measure.
 * \return the hosmer lemeshow statistics.
 */
double gof(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

void buildOutputTargetVectors(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data,
                              std::vector<double>& output, std::vector<uint>& target);

void buildOutputTargetVectors(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data,
                              std::vector<std::vector<double>>& output,
                              std::vector<std::vector<double>>& target);

/**Flatten a DataSet of multi-output predictions and targets into a single
 * pair of vectors, concatenated pattern by pattern. Used by the regression
 * metrics below.
 */
void buildFlatRegressionVectors(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data,
                                std::vector<double>& output, std::vector<double>& target);

// ---- Regression metrics --------------------------------------------------
//
// All take "flat" vectors: predictions and actuals concatenated across the
// whole dataset. Use buildFlatRegressionVectors (or the Ensemble overloads
// below) when you have a model + DataSet.

/**Mean absolute error: \f$\frac{1}{N}\sum_i |t_i - y_i|\f$.*/
double mae(const std::vector<double>& output, const std::vector<double>& target);
double mae(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Mean absolute percentage error in percent:
 * \f$\frac{100}{N}\sum_i |t_i - y_i| / |t_i|\f$.
 *
 * Elements with \f$|t_i|\f$ below a small epsilon are skipped (MAPE is
 * undefined at zero). Returns NaN if every element is skipped.
 */
double mape(const std::vector<double>& output, const std::vector<double>& target);
double mape(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Symmetric MAPE in percent:
 * \f$\frac{200}{N}\sum_i |t_i - y_i| / (|t_i| + |y_i|)\f$.
 *
 * Bounded between 0 and 200. Better behaved than MAPE near zero, since the
 * denominator only vanishes when both target and prediction are zero (in
 * which case that element is skipped).
 */
double smape(const std::vector<double>& output, const std::vector<double>& target);
double smape(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Root mean squared error: \f$\sqrt{\frac{1}{N}\sum_i (t_i - y_i)^2}\f$.*/
double rmse(const std::vector<double>& output, const std::vector<double>& target);
double rmse(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

/**Coefficient of determination:
 * \f$R^2 = 1 - \frac{\sum_i (t_i - y_i)^2}{\sum_i (t_i - \bar t)^2}\f$.
 *
 * Returns 1 for a perfect fit, 0 for a model that's no better than
 * predicting the target mean, and can go negative for a worse-than-mean
 * model. Returns NaN if the targets are constant (denominator is zero).
 */
double r2(const std::vector<double>& output, const std::vector<double>& target);
double r2(NeuralNetHack::Ensemble& committee, DataTools::DataSet& data);

// ---- Confusion-matrix derived metrics -----------------------------------
//
// Build a ConfusionMatrix (binary, multi-class, or via fromEnsemble) and
// pass it to any of these. Per-class metrics default to class 1 (the
// positive class in the binary convention).

/**Overall accuracy: correct / total. Returns NaN if the matrix is empty.*/
double accuracy(const ConfusionMatrix& cm);

/**Precision for class `cls`: TP_cls / (TP_cls + FP_cls).
 * Returns NaN if no instance was predicted as `cls`.
 */
double precision(const ConfusionMatrix& cm, uint cls = 1);

/**Recall (sensitivity) for class `cls`: TP_cls / (TP_cls + FN_cls).
 * Returns NaN if `cls` never appears in the ground truth.
 */
double recall(const ConfusionMatrix& cm, uint cls = 1);

/**F1 score for class `cls`: harmonic mean of precision and recall.
 * Returns NaN if either input is undefined or both are zero.
 */
double f1(const ConfusionMatrix& cm, uint cls = 1);

/**Matthews correlation coefficient for binary classification.
 * Range -1 (anti-correlated) to +1 (perfect). Returns NaN for non-binary
 * matrices or when the denominator is zero.
 */
double mcc(const ConfusionMatrix& cm);

/**Mean recall across classes. Robust to class imbalance.*/
double balancedAccuracy(const ConfusionMatrix& cm);

/**Unweighted mean of per-class precision / recall / F1 across all classes.
 * Classes whose individual metric is NaN are skipped from the mean.
 */
double macroPrecision(const ConfusionMatrix& cm);
double macroRecall(const ConfusionMatrix& cm);
double macroF1(const ConfusionMatrix& cm);
} // namespace ErrorMeasures
} // namespace EvalTools

#endif
