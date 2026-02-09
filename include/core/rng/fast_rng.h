#ifndef COAST_FAST_RNG_H
#define COAST_FAST_RNG_H

#include "core/rng/rng_wrapper.h"
#include "core/rng/mt19937_simd.h"
#include <random>

namespace core::rng {

/// Fast RNG using SIMD-optimized Mersenne Twister
using fast_rng_t = rng_t<mt19937_simd>;

/// Standard RNG using std::mt19937 (for compatibility/comparison)
using standard_rng_t = rng_t<std::mt19937>;

} // namespace core::rng

// Convenience: bring fast_rng_t into global namespace
using core::rng::fast_rng_t;

#endif // COAST_FAST_RNG_H
