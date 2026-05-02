#include "Random.hh"

#include <random>

namespace nnh::rand {

namespace {
std::mt19937_64& tls_generator() {
	static thread_local std::mt19937_64 gen(1);
	return gen;
}
} // namespace

void seed(uint64_t s) {
	tls_generator().seed(s);
}

double uniform() {
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	return dist(tls_generator());
}

uint64_t uint_below(uint64_t n) {
	std::uniform_int_distribution<uint64_t> dist(0, n - 1);
	return dist(tls_generator());
}

std::mt19937_64& generator() {
	return tls_generator();
}

} // namespace nnh::rand
