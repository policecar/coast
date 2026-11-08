#ifndef COAST_PHILOX_4X32_AVX512_H
#define COAST_PHILOX_4X32_AVX512_H

#include <cstdint>
#include <algorithm>
#include <limits>

// SIMD detection
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #define COAST_PHILOX_AVX512 1
    #include <immintrin.h>
#endif

namespace core::rng {

/// SIMD-buffered Philox-4x32-10 PRNG.
///
/// Runs 16 independent Philox-4x32 instances in parallel using AVX-512,
/// producing 64 random uint32 values per refill. Values are handed out
/// one at a time via operator()() — identical hot path to mt19937_simd
/// (sequential load + increment from a cache buffer).
///
/// Key design differences from mt19937_simd:
///   - State: ~544 bytes  vs  5120 bytes for mt19937_simd
///   - Refill: every 64 calls (64 values)  vs  every 624 calls (twist writes 10 KB)
///   - Each refill writes 256 bytes  vs  the mt19937_simd twist's 10 KB write
///   - No cache-polluting bulk state update; refill is compute-only, not memory-heavy
///
/// Each instance has a unique key derived from the seed via the Weyl sequence,
/// guaranteeing 16 non-overlapping random streams.
///
/// Falls back to 4 scalar Philox-4x32 instances (16-value buffer) on platforms
/// without AVX-512.
class philox_4x32_avx512 {
public:
    using result_type = uint32_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type{0}; }

    philox_4x32_avx512() { seed(default_seed); }
    explicit philox_4x32_avx512(result_type s) { seed(s); }

    void seed(result_type s) {
        // Expand 32-bit seed to 64-bit base key via splitmix64
        uint64_t z = static_cast<uint64_t>(s) + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= (z >> 31);
        const uint32_t k0 = static_cast<uint32_t>(z);
        const uint32_t k1 = static_cast<uint32_t>(z >> 32);

        // Derive N_INST unique keys via the Philox Weyl sequence.
        // Instance i uses key (k0 + i*W0, k1 + i*W1).  Since the Weyl sequence
        // is a bijection on uint32, all 16 keys are distinct, guaranteeing
        // non-overlapping streams even when all instances share counter = 0.
        for (int i = 0; i < N_INST; i++) {
            key_k0_[i] = k0 + static_cast<uint32_t>(i) * W0;
            key_k1_[i] = k1 + static_cast<uint32_t>(i) * W1;
        }

        // All instances start with counter = 0
        std::fill(ctr_lo_, ctr_lo_ + N_INST, uint32_t{0});
        std::fill(ctr_hi_, ctr_hi_ + N_INST, uint32_t{0});

        idx_ = BUF_SIZE;  // mark buffer empty; refill on first call
    }

    result_type operator()() {
        if (idx_ >= BUF_SIZE) { refill(); }
        return buf_[idx_++];
    }

    void discard(unsigned long long n) {
        const unsigned long long avail = BUF_SIZE - static_cast<unsigned long long>(idx_);
        if (n < avail) { idx_ += static_cast<int>(n); return; }
        n -= avail;
        // Skip n / BUF_SIZE complete refills
        advance_counters(n / BUF_SIZE);
        // Partial refill for remainder
        const int rem = static_cast<int>(n % BUF_SIZE);
        if (rem > 0) { refill(); idx_ = rem; }
        else         { idx_ = BUF_SIZE; }
    }

private:
    // Philox-4x32 constants
    static constexpr uint32_t M0 = 0xD2511F53u;  // multiplier for lanes 0,2
    static constexpr uint32_t M1 = 0xCD9E8D57u;  // multiplier for lanes 1,3
    static constexpr uint32_t W0 = 0x9E3779B9u;  // Weyl increment (golden ratio)
    static constexpr uint32_t W1 = 0xBB67AE85u;  // Weyl increment (sqrt(3)-1)

    static constexpr uint32_t default_seed = 5489u;

#ifdef COAST_PHILOX_AVX512
    static constexpr int N_INST    = 16;   // AVX-512: 16 instances in parallel
    static constexpr int BUF_SIZE  = 64;   // 16 instances × 4 outputs = 64 values
#else
    static constexpr int N_INST    = 4;    // scalar fallback: 4 instances
    static constexpr int BUF_SIZE  = 16;   // 4 × 4 = 16 values
#endif

    // Per-instance unique keys (derived from seed)
    alignas(64) uint32_t key_k0_[N_INST] = {};
    alignas(64) uint32_t key_k1_[N_INST] = {};

    // 64-bit per-instance counters (lo and hi 32 bits, with carry)
    alignas(64) uint32_t ctr_lo_[N_INST] = {};
    alignas(64) uint32_t ctr_hi_[N_INST] = {};

    // Output cache: N_INST instances × 4 output words = BUF_SIZE uint32
    // Layout: [word0 of all instances, word1 of all, word2 of all, word3 of all]
    alignas(64) uint32_t buf_[BUF_SIZE] = {};

    int idx_ = BUF_SIZE;

    // -----------------------------------------------------------------------
    // Scalar fallback (4 instances, 16-value buffer)
    // -----------------------------------------------------------------------
    void scalar_philox_round(uint32_t& x0, uint32_t& x1,
                             uint32_t& x2, uint32_t& x3,
                             uint32_t  k0, uint32_t  k1) const {
        const uint64_t t0 = static_cast<uint64_t>(M0) * x0;
        const uint64_t t1 = static_cast<uint64_t>(M1) * x2;
        x0 = static_cast<uint32_t>(t1 >> 32) ^ x1 ^ k0;
        x1 = static_cast<uint32_t>(t1);
        x2 = static_cast<uint32_t>(t0 >> 32) ^ x3 ^ k1;
        x3 = static_cast<uint32_t>(t0);
    }

#ifdef COAST_PHILOX_AVX512
    // -----------------------------------------------------------------------
    // AVX-512 path: 16 instances in parallel
    // -----------------------------------------------------------------------

    // Multiply all 16 uint32 lanes and return the HIGH 32 bits of each product.
    // Uses two _mm512_mul_epu32 calls (which operate on even 64-bit lanes only)
    // to cover all 16 lanes.
    [[nodiscard]] static __m512i mulhi_epu32(__m512i a, __m512i b) noexcept {
        // Even 32-bit positions (0,2,4,...,14) — mul_epu32 uses low 32 of each 64-bit lane
        const __m512i even   = _mm512_mul_epu32(a, b);
        // Odd 32-bit positions (1,3,5,...,15) — shift to bring them into even positions
        const __m512i a_odd  = _mm512_srli_epi64(a, 32);
        const __m512i b_odd  = _mm512_srli_epi64(b, 32);
        const __m512i odd    = _mm512_mul_epu32(a_odd, b_odd);
        // Extract hi32 from each 64-bit product
        const __m512i hi_evn = _mm512_srli_epi64(even, 32);         // [hi0,0, hi2,0, ...]
        const __m512i hi_odd = _mm512_slli_epi64(                    // [0,hi1, 0,hi3, ...]
            _mm512_srli_epi64(odd, 32), 32);
        return _mm512_or_si512(hi_evn, hi_odd);                      // [hi0,hi1,hi2,hi3,...]
    }

    void refill() noexcept {
        // Load 16-instance counter vectors (x2, x3 are always zero — 64-bit counter suffices)
        __m512i x0 = _mm512_load_si512(ctr_lo_);
        __m512i x1 = _mm512_load_si512(ctr_hi_);
        __m512i x2 = _mm512_setzero_si512();
        __m512i x3 = _mm512_setzero_si512();

        // Load 16 unique keys
        __m512i k0 = _mm512_load_si512(key_k0_);
        __m512i k1 = _mm512_load_si512(key_k1_);

        const __m512i vM0 = _mm512_set1_epi32(static_cast<int>(M0));
        const __m512i vM1 = _mm512_set1_epi32(static_cast<int>(M1));
        const __m512i vW0 = _mm512_set1_epi32(static_cast<int>(W0));
        const __m512i vW1 = _mm512_set1_epi32(static_cast<int>(W1));

        // 10 Philox rounds: mix → bump key
        for (int r = 0; r < 10; ++r) {
            const __m512i lo0 = _mm512_mullo_epi32(vM0, x0);
            const __m512i hi0 = mulhi_epu32(vM0, x0);
            const __m512i lo1 = _mm512_mullo_epi32(vM1, x2);
            const __m512i hi1 = mulhi_epu32(vM1, x2);

            x0 = _mm512_xor_si512(_mm512_xor_si512(hi1, x1), k0);
            x1 = lo1;
            x2 = _mm512_xor_si512(_mm512_xor_si512(hi0, x3), k1);
            x3 = lo0;

            k0 = _mm512_add_epi32(k0, vW0);
            k1 = _mm512_add_epi32(k1, vW1);
        }

        // Store 64 results: word0..word3 of all 16 instances
        _mm512_store_si512(&buf_[ 0], x0);
        _mm512_store_si512(&buf_[16], x1);
        _mm512_store_si512(&buf_[32], x2);
        _mm512_store_si512(&buf_[48], x3);

        // Advance all 16 counters by 1 (64-bit, with carry from lo to hi)
        const __m512i one    = _mm512_set1_epi32(1);
        const __m512i old_lo = _mm512_load_si512(ctr_lo_);
        const __m512i new_lo = _mm512_add_epi32(old_lo, one);
        _mm512_store_si512(ctr_lo_, new_lo);

        // Carry: where new_lo wrapped to 0, increment ctr_hi_
        const __mmask16 carry = _mm512_cmpeq_epu32_mask(new_lo, _mm512_setzero_si512());
        if (carry) {
            __m512i hi = _mm512_load_si512(ctr_hi_);
            hi = _mm512_mask_add_epi32(hi, carry, hi, one);
            _mm512_store_si512(ctr_hi_, hi);
        }

        idx_ = 0;
    }

    void advance_counters(uint64_t n) {
        for (int i = 0; i < N_INST; i++) {
            const uint64_t cur = static_cast<uint64_t>(ctr_lo_[i])
                               | (static_cast<uint64_t>(ctr_hi_[i]) << 32);
            const uint64_t nxt = cur + n;
            ctr_lo_[i] = static_cast<uint32_t>(nxt);
            ctr_hi_[i] = static_cast<uint32_t>(nxt >> 32);
        }
        idx_ = BUF_SIZE;  // mark buffer stale
    }

#else
    // -----------------------------------------------------------------------
    // Scalar fallback: 4 instances, 16-value buffer
    // -----------------------------------------------------------------------
    void refill() {
        for (int inst = 0; inst < N_INST; inst++) {
            uint32_t x0 = ctr_lo_[inst], x1 = ctr_hi_[inst], x2 = 0, x3 = 0;
            uint32_t k0 = key_k0_[inst], k1 = key_k1_[inst];
            for (int r = 0; r < 10; ++r) {
                scalar_philox_round(x0, x1, x2, x3, k0, k1);
                k0 += W0; k1 += W1;
            }
            buf_[inst]            = x0;
            buf_[inst + N_INST]   = x1;
            buf_[inst + N_INST*2] = x2;
            buf_[inst + N_INST*3] = x3;

            // Advance 64-bit counter for this instance
            const uint64_t cur = static_cast<uint64_t>(ctr_lo_[inst])
                               | (static_cast<uint64_t>(ctr_hi_[inst]) << 32);
            const uint64_t nxt = cur + 1;
            ctr_lo_[inst] = static_cast<uint32_t>(nxt);
            ctr_hi_[inst] = static_cast<uint32_t>(nxt >> 32);
        }
        idx_ = 0;
    }

    void advance_counters(uint64_t n) {
        for (int i = 0; i < N_INST; i++) {
            const uint64_t cur = static_cast<uint64_t>(ctr_lo_[i])
                               | (static_cast<uint64_t>(ctr_hi_[i]) << 32);
            const uint64_t nxt = cur + n;
            ctr_lo_[i] = static_cast<uint32_t>(nxt);
            ctr_hi_[i] = static_cast<uint32_t>(nxt >> 32);
        }
        idx_ = BUF_SIZE;
    }
#endif
};

} // namespace core::rng

#endif // COAST_PHILOX_4X32_AVX512_H
