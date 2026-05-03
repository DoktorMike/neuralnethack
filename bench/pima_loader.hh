// Shared Pima loader for the three bench harnesses. Reads tab-separated
// rows of 9 numbers (8 features + binary label) from the existing
// neuralnethack test fixture, optionally Z-normalises features against
// the training set's mean/std.
#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

struct Pima {
	std::vector<std::vector<double>> X; // [N][8]
	std::vector<int> y;                 // [N], 0 or 1
};

inline Pima loadPima(const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open " + path);
	Pima d;
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		std::vector<double> row;
		double v;
		while (iss >> v) row.push_back(v);
		if (row.size() != 9)
			throw std::runtime_error("Pima row must have 9 values, got " +
			                         std::to_string(row.size()));
		d.X.push_back({row.begin(), row.begin() + 8});
		d.y.push_back(static_cast<int>(row[8]));
	}
	return d;
}

// Compute per-feature mean / std on `train` and apply to both train and test.
inline void zNormalise(Pima& train, Pima& test) {
	const std::size_t D = train.X.front().size();
	std::vector<double> mean(D, 0.0), sd(D, 0.0);
	for (const auto& r : train.X)
		for (std::size_t j = 0; j < D; ++j) mean[j] += r[j];
	for (std::size_t j = 0; j < D; ++j) mean[j] /= train.X.size();
	for (const auto& r : train.X)
		for (std::size_t j = 0; j < D; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
	for (std::size_t j = 0; j < D; ++j) sd[j] = std::sqrt(sd[j] / train.X.size());
	for (auto* d : {&train, &test})
		for (auto& r : d->X)
			for (std::size_t j = 0; j < D; ++j)
				if (sd[j] > 1e-12) r[j] = (r[j] - mean[j]) / sd[j];
}

} // namespace bench
