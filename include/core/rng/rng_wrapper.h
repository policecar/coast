#ifndef COAST_RNG_WRAPPER_H
#define COAST_RNG_WRAPPER_H

#include <type_traits>

namespace core::rng {

/// Generic wrapper for uniform random bit generators.
/// Provides a uniform interface allowing easy swapping of RNG implementations.
template<typename Generator>
class rng_t {
    Generator gen_;

public:
    using result_type = typename Generator::result_type;

    rng_t() = default;

    explicit rng_t(result_type seed) : gen_(seed) {}

    static constexpr result_type min() { return Generator::min(); }

    static constexpr result_type max() { return Generator::max(); }

    result_type operator()() { return gen_(); }

    void seed(result_type value) { gen_.seed(value); }

    void discard(unsigned long long z) { gen_.discard(z); }

    /// Access the underlying generator (for advanced use)
    Generator& generator() { return gen_; }
    const Generator& generator() const { return gen_; }
};

} // namespace core::rng

#endif // COAST_RNG_WRAPPER_H
