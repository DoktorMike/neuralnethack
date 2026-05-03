// UCI Covertype loader. CSV with 54 features + class label (1..7) per row.
// First 10 columns are continuous (we z-normalise), remaining 44 are
// already 0/1 indicator variables (left as-is to keep their semantics).
#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

struct CovType {
	std::vector<std::vector<double>> X; // [N][54]
	std::vector<int> y;                 // [N], 0..6
};

inline CovType loadCovType(const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open " + path);
	CovType d;
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		std::vector<double> row;
		row.reserve(55);
		std::stringstream ss(line);
		std::string field;
		while (std::getline(ss, field, ',')) row.push_back(std::stod(field));
		if (row.size() != 55)
			throw std::runtime_error("Covtype row must have 55 fields, got " +
			                         std::to_string(row.size()));
		d.X.push_back({row.begin(), row.begin() + 54});
		d.y.push_back(static_cast<int>(row[54]) - 1); // 1..7 -> 0..6
	}
	return d;
}

// Z-normalise the first 10 (continuous) features against train means/stds.
// The remaining 44 columns are already 0/1 indicators and are left alone.
inline void zNormaliseContinuous(CovType& train, CovType& test) {
	constexpr std::size_t CONT_DIMS = 10;
	std::vector<double> mean(CONT_DIMS, 0.0), sd(CONT_DIMS, 0.0);
	for (const auto& r : train.X)
		for (std::size_t j = 0; j < CONT_DIMS; ++j) mean[j] += r[j];
	for (std::size_t j = 0; j < CONT_DIMS; ++j) mean[j] /= train.X.size();
	for (const auto& r : train.X)
		for (std::size_t j = 0; j < CONT_DIMS; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
	for (std::size_t j = 0; j < CONT_DIMS; ++j) sd[j] = std::sqrt(sd[j] / train.X.size());
	for (auto* d : {&train, &test})
		for (auto& r : d->X)
			for (std::size_t j = 0; j < CONT_DIMS; ++j)
				if (sd[j] > 1e-12) r[j] = (r[j] - mean[j]) / sd[j];
}

} // namespace bench
