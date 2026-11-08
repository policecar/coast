#ifndef COAST_PHILOX_4X32_H
#define COAST_PHILOX_4X32_H

#include <cstdint>
#include <limits>

namespace core::rng {

/// Counter-based PRNG: Philox-4x32-10
///
/// State: ~44 bytes per instance (vs 5120 bytes for mt19937_simd).
/// No periodic state-refresh ("twist") — every call independently computes
/// output from a 128-bit counter and a fixed 64-bit key.
///
/// Statistical quality: passes BigCrush, PractRand, TestU01 Crush.
/// Reference: Salmon et al., "Random123: A Library of Counter-Based Random
///            Number Generators", SC 2011.
///
/// Layout designed for future AVX-512 extension: 4 independent Philox-4x32
/// instances can be mapped to a single 512-bit register for bulk generation
/// of 16 values per SIMD call.
class philox_4x32 {
public:
    using result_type = uint32_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type{0}; }

    philox_4x32() { seed(default_seed); }
    explicit philox_4x32(result_type s) { seed(s); }

    void seed(result_type s) {
        // Expand 32-bit seed to 64-bit key via one splitmix64 step,
        // providing good avalanche even for seeds that differ by 1.
        uint64_t z = static_cast<uint64_t>(s) + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= (z >> 31);
        key_[0] = static_cast<uint32_t>(z);
        key_[1] = static_cast<uint32_t>(z >> 32);
        ctr_[0] = ctr_[1] = ctr_[2] = ctr_[3] = 0;
        idx_ = 4;  // mark buffer empty; refill on first call
    }

    result_type operator()() {
        if (idx_ >= 4) { refill(); }
        return buf_[idx_++];
    }

    void discard(unsigned long long n) {
        const unsigned long long avail = 4ULL - static_cast<unsigned long long>(idx_);
        if (n < avail) {
            idx_ += static_cast<int>(n);
            return;
        }
        n -= avail;
        // Skip n/4 complete blocks
        advance_counter(n / 4);
        // Handle the partial block remainder
        const int rem = static_cast<int>(n % 4);
        if (rem > 0) {
            refill();
            idx_ = rem;
        } else {
            idx_ = 4;  // will refill lazily on next operator()
        }
    }

private:
    static constexpr uint32_t default_seed = 5489u;

    // Philox-4x32-10 constants (from the Random123 specification)
    static constexpr uint32_t M0 = 0xD2511F53u;  // multiplier for lanes 0,1
    static constexpr uint32_t M1 = 0xCD9E8D57u;  // multiplier for lanes 2,3
    static constexpr uint32_t W0 = 0x9E3779B9u;  // Weyl increment (golden ratio)
    static constexpr uint32_t W1 = 0xBB67AE85u;  // Weyl increment (sqrt(3)-1)

    uint32_t ctr_[4] = {};  // 128-bit monotonic counter
    uint32_t key_[2] = {};  // 64-bit key derived from seed (never mutated)
    uint32_t buf_[4] = {};  // 4-value output buffer
    int      idx_    =  4;  // read index into buf_; 4 = buffer empty

    void refill() {
        // Apply Philox-4x32-10: 10 rounds of bijective mixing
        uint32_t x0 = ctr_[0], x1 = ctr_[1], x2 = ctr_[2], x3 = ctr_[3];
        uint32_t k0 = key_[0], k1 = key_[1];

        for (int r = 0; r < 10; ++r) {
            const uint64_t t0 = static_cast<uint64_t>(M0) * x0;
            const uint64_t t1 = static_cast<uint64_t>(M1) * x2;
            const uint32_t hi0 = static_cast<uint32_t>(t0 >> 32);
            const uint32_t lo0 = static_cast<uint32_t>(t0);
            const uint32_t hi1 = static_cast<uint32_t>(t1 >> 32);
            const uint32_t lo1 = static_cast<uint32_t>(t1);
            x0 = hi1 ^ x1 ^ k0;
            x1 = lo1;
            x2 = hi0 ^ x3 ^ k1;
            x3 = lo0;
            k0 += W0;
            k1 += W1;
        }

        buf_[0] = x0;  buf_[1] = x1;  buf_[2] = x2;  buf_[3] = x3;
        idx_ = 0;
        advance_counter(1);
    }

    // Add n to the 128-bit counter with carry propagation
    void advance_counter(uint64_t n) {
        const uint64_t lo_old = static_cast<uint64_t>(ctr_[0])
                              | (static_cast<uint64_t>(ctr_[1]) << 32);
        const uint64_t hi_old = static_cast<uint64_t>(ctr_[2])
                              | (static_cast<uint64_t>(ctr_[3]) << 32);
        const uint64_t lo_new = lo_old + n;
        const uint64_t hi_new = hi_old + (lo_new < lo_old ? 1ULL : 0ULL);
        ctr_[0] = static_cast<uint32_t>(lo_new);
        ctr_[1] = static_cast<uint32_t>(lo_new >> 32);
        ctr_[2] = static_cast<uint32_t>(hi_new);
        ctr_[3] = static_cast<uint32_t>(hi_new >> 32);
    }
};

} // namespace core::rng

#endif // COAST_PHILOX_4X32_H
