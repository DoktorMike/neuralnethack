#ifndef __nnh_Random_hh__
#define __nnh_Random_hh__

#include <cstdint>
#include <random>

namespace nnh {

/**Thread-local RNG abstraction. Each OS thread keeps its own seeded
 * mt19937 state, so parallel ensemble training (or any other
 * embarrassingly parallel use) doesn't race the way the previous
 * `drand48()` calls did.
 *
 * Seed each thread (or each parallel iteration) explicitly via
 * `rand_seed(seed)` at the top of its work — like `srand48` but
 * thread-local. Threads that never seed get a default seed (1), the
 * same way `drand48()` behaved.
 */
namespace rand {

/**Reseed the calling thread's generator. */
void seed(uint64_t s);

/**Uniform double in [0, 1). Drop-in for `drand48()`. */
double uniform();

/**Uniform integer in [0, n). */
uint64_t uint_below(uint64_t n);

/**Reference to the calling thread's generator, for use with
 * std::shuffle and other generator-aware algorithms.
 */
std::mt19937_64& generator();

} // namespace rand
} // namespace nnh

#endif
