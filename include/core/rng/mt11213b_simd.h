#ifndef COAST_MT11213B_SIMD_H
#define COAST_MT11213B_SIMD_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

// SIMD detection and includes — use distinct macros to avoid collision with mt19937_simd.h
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #define COAST_MT11213B_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define COAST_MT11213B_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define COAST_MT11213B_NEON 1
    #include <arm_neon.h>
#endif

namespace core::rng {

/// SIMD-optimized MT11213b generator.
///
/// MT11213b is a member of the Mersenne Twister family with parameters
///   (n=351, m=175, r=19, a=0xE4BD75F5).  Period: 2^11213 − 1.
/// Statistical quality comparable to MT19937; passes BigCrush.
/// Boost.Random documents identical parameters for boost::mt11213b.
///
/// Interface and behaviour are identical to mt19937_simd — drop-in replacement
/// via fast_rng.h.  The only observable difference is the random stream.
///
/// Key differences from mt19937_simd:
///   state_size : 351   vs  624    (44 % smaller)
///   twist write: 2.8 KB vs 5.0 KB per refill
///   refill every: 351  vs  624    values
///   amortised instruction cost/output: ~constant (both scale with state_size)
///   expected benefit: reduced L2/L3 footprint from smaller sequential writes
///
/// Upper/lower masks reflect r = 19:
///   upper = 0xFFF80000  (bits 31..19)
///   lower = 0x0007FFFF  (bits 18..0)
///
/// Tempering parameters are MT11213b-specific (differ from MT19937):
///   b = 0x655E5280  (vs 0x9D2C5680)
///   c = 0xFFD58000  (vs 0xEFC60000)
///   l = 17          (vs 18)
///
class mt11213b_simd {
public:
    using result_type = std::uint32_t;

    // MT11213b parameters — n*w - r = 351*32 - 19 = 11213 → period 2^11213-1
    static constexpr std::size_t state_size   = 351;
    static constexpr std::size_t shift_size   = 175;
    static constexpr result_type upper_mask   = 0xFFF80000u;  // (~0u) << 19
    static constexpr result_type lower_mask   = 0x0007FFFFu;  // ~upper_mask
    static constexpr result_type xor_mask     = 0xE4BD75F5u;  // a
    static constexpr result_type default_seed = 5489u;

    // Tempering parameters (MT11213b-specific; from Boost.Random / original MT papers)
    // u=11 (same as MT19937), s=7, t=15 (same), l=17, b and c differ
    static constexpr result_type temper_b = 0x655E5280u;
    static constexpr result_type temper_c = 0xFFD58000u;

    // Batch size based on available SIMD
#if defined(COAST_MT11213B_AVX512)
    static constexpr std::size_t batch_size = 16;
#elif defined(COAST_MT11213B_AVX2)
    static constexpr std::size_t batch_size = 8;
#elif defined(COAST_MT11213B_NEON)
    static constexpr std::size_t batch_size = 4;
#else
    static constexpr std::size_t batch_size = 1;
#endif

private:
    // Padding of batch_size covers the mt_[i+1] read at i near state_size (same pattern
    // as mt19937_simd).  For MT11213b the second twist loop stops before needing it, but
    // we keep the padding for consistency and safety.
    alignas(64) std::array<result_type, state_size + batch_size> mt_;
    alignas(64) std::array<result_type, state_size>              tempered_;
    std::size_t mti_ = state_size;

public:
    mt11213b_simd() { seed(default_seed); }
    explicit mt11213b_simd(result_type value) { seed(value); }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    void seed(result_type value) {
        mt_[0] = value;
        for (std::size_t i = 1; i < state_size; ++i) {
            mt_[i] = static_cast<result_type>(
                1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + i
            );
        }
        mti_ = state_size;
    }

    result_type operator()() {
        if (mti_ >= state_size) [[unlikely]] {
            generate_and_temper();
        }
        return tempered_[mti_++];
    }

    void discard(unsigned long long count) {
        for (unsigned long long i = 0; i < count; ++i) {
            operator()();
        }
    }

private:
    void generate_and_temper() {
#if defined(COAST_MT11213B_AVX512)
        twist_avx512();
        temper_all_avx512();
#elif defined(COAST_MT11213B_AVX2)
        twist_avx2();
        temper_all_avx2();
#elif defined(COAST_MT11213B_NEON)
        twist_neon();
        temper_all_neon();
#else
        twist_scalar();
        temper_all_scalar();
#endif
        mti_ = 0;
    }

    // Scalar temper helper — used as the SIMD-tail fallback in every path.
    [[nodiscard]] static result_type temper_one(result_type y) noexcept {
        y ^=  (y >> 11);
        y ^=  (y <<  7) & temper_b;
        y ^=  (y << 15) & temper_c;
        y ^=  (y >> 17);          // l=17 for MT11213b (vs 18 for MT19937)
        return y;
    }

    // -----------------------------------------------------------------------
    // AVX-512 path
    // -----------------------------------------------------------------------
#if defined(COAST_MT11213B_AVX512)

    void twist_avx512() {
        const __m512i v_upper = _mm512_set1_epi32(static_cast<int>(upper_mask));
        const __m512i v_lower = _mm512_set1_epi32(static_cast<int>(lower_mask));
        const __m512i v_xor   = _mm512_set1_epi32(static_cast<int>(xor_mask));
        const __m512i one     = _mm512_set1_epi32(1);

        // First section: i = 0 .. (state_size - shift_size - 1) = 0..175
        //   mt[i] = mt[i + shift_size] ^ (y>>1) ^ mag01[y & 1]
        //   where y = (mt[i] & upper_mask) | (mt[i+1] & lower_mask)
        //
        // 176 elements = 11 × 16: the SIMD loop covers it exactly, no scalar tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            __m512i mt_i    = _mm512_load_si512( reinterpret_cast<const __m512i*>(&mt_[i]));
            __m512i mt_i1   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + 1]));
            __m512i mt_shft = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + shift_size]));

            __m512i y       = _mm512_or_si512(_mm512_and_si512(mt_i,  v_upper),
                                               _mm512_and_si512(mt_i1, v_lower));
            __m512i mag_val = _mm512_and_si512(_mm512_sub_epi32(_mm512_setzero_si512(),
                                                                  _mm512_and_si512(y, one)),
                                                v_xor);
            __m512i result  = _mm512_xor_si512(_mm512_xor_si512(mt_shft,
                                                                  _mm512_srli_epi32(y, 1)),
                                                mag_val);
            _mm512_store_si512(reinterpret_cast<__m512i*>(&mt_[i]), result);
        }
        // Scalar tail (none expected for MT11213b since 176 is divisible by 16)
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second section: i = 176 .. 349
        //   mt[i] = mt[i + shift_size - state_size] ^ (y>>1) ^ mag01[y & 1]
        //   Note: mt_[176] is 64-byte aligned (176×4 = 704 = 11×64), so aligned
        //   store would be valid; storeu kept for consistency with mt19937_simd.
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            __m512i mt_i    = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i]));
            __m512i mt_i1   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + 1]));
            __m512i mt_shft = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + shift_size - state_size]));

            __m512i y       = _mm512_or_si512(_mm512_and_si512(mt_i,  v_upper),
                                               _mm512_and_si512(mt_i1, v_lower));
            __m512i mag_val = _mm512_and_si512(_mm512_sub_epi32(_mm512_setzero_si512(),
                                                                  _mm512_and_si512(y, one)),
                                                v_xor);
            __m512i result  = _mm512_xor_si512(_mm512_xor_si512(mt_shft,
                                                                  _mm512_srli_epi32(y, 1)),
                                                mag_val);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&mt_[i]), result);
        }
        // Scalar tail: indices 336..349 (14 elements)
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Last element (always scalar — wraps around)
        result_type y        = (mt_[state_size - 1] & upper_mask) | (mt_[0] & lower_mask);
        mt_[state_size - 1]  = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_avx512() {
        const __m512i mask1 = _mm512_set1_epi32(static_cast<int>(temper_b));
        const __m512i mask2 = _mm512_set1_epi32(static_cast<int>(temper_c));

        // SIMD covers full 16-element batches; state_size=351 leaves a 15-element tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size; i += batch_size) {
            __m512i y = _mm512_load_si512(reinterpret_cast<const __m512i*>(&mt_[i]));
            y = _mm512_xor_si512(y, _mm512_srli_epi32(y, 11));
            y = _mm512_xor_si512(y, _mm512_and_si512(_mm512_slli_epi32(y,  7), mask1));
            y = _mm512_xor_si512(y, _mm512_and_si512(_mm512_slli_epi32(y, 15), mask2));
            y = _mm512_xor_si512(y, _mm512_srli_epi32(y, 17));   // l=17 for MT11213b
            _mm512_store_si512(reinterpret_cast<__m512i*>(&tempered_[i]), y);
        }
        // Scalar tail: indices 336..350 (15 elements)
        for (; i < state_size; ++i) {
            tempered_[i] = temper_one(mt_[i]);
        }
    }

#endif // COAST_MT11213B_AVX512

    // -----------------------------------------------------------------------
    // AVX2 path
    // -----------------------------------------------------------------------
#if defined(COAST_MT11213B_AVX2)

    void twist_avx2() {
        const __m256i v_upper = _mm256_set1_epi32(static_cast<int>(upper_mask));
        const __m256i v_lower = _mm256_set1_epi32(static_cast<int>(lower_mask));
        const __m256i v_xor   = _mm256_set1_epi32(static_cast<int>(xor_mask));
        const __m256i one     = _mm256_set1_epi32(1);

        // First section: 176 elements = 22 × 8, no scalar tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            __m256i mt_i    = _mm256_load_si256( reinterpret_cast<const __m256i*>(&mt_[i]));
            __m256i mt_i1   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + 1]));
            __m256i mt_shft = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + shift_size]));

            __m256i y       = _mm256_or_si256(_mm256_and_si256(mt_i,  v_upper),
                                               _mm256_and_si256(mt_i1, v_lower));
            __m256i mag_val = _mm256_and_si256(_mm256_sub_epi32(_mm256_setzero_si256(),
                                                                  _mm256_and_si256(y, one)),
                                                v_xor);
            __m256i result  = _mm256_xor_si256(_mm256_xor_si256(mt_shft,
                                                                  _mm256_srli_epi32(y, 1)),
                                                mag_val);
            _mm256_store_si256(reinterpret_cast<__m256i*>(&mt_[i]), result);
        }
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second section: 174 elements, 21 × 8 + 6 scalar tail.
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            __m256i mt_i    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i]));
            __m256i mt_i1   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + 1]));
            __m256i mt_shft = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + shift_size - state_size]));

            __m256i y       = _mm256_or_si256(_mm256_and_si256(mt_i,  v_upper),
                                               _mm256_and_si256(mt_i1, v_lower));
            __m256i mag_val = _mm256_and_si256(_mm256_sub_epi32(_mm256_setzero_si256(),
                                                                  _mm256_and_si256(y, one)),
                                                v_xor);
            __m256i result  = _mm256_xor_si256(_mm256_xor_si256(mt_shft,
                                                                  _mm256_srli_epi32(y, 1)),
                                                mag_val);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&mt_[i]), result);
        }
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        result_type y       = (mt_[state_size - 1] & upper_mask) | (mt_[0] & lower_mask);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_avx2() {
        const __m256i mask1 = _mm256_set1_epi32(static_cast<int>(temper_b));
        const __m256i mask2 = _mm256_set1_epi32(static_cast<int>(temper_c));

        // 351 = 43 × 8 + 7, so 7-element scalar tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size; i += batch_size) {
            __m256i y = _mm256_load_si256(reinterpret_cast<const __m256i*>(&mt_[i]));
            y = _mm256_xor_si256(y, _mm256_srli_epi32(y, 11));
            y = _mm256_xor_si256(y, _mm256_and_si256(_mm256_slli_epi32(y,  7), mask1));
            y = _mm256_xor_si256(y, _mm256_and_si256(_mm256_slli_epi32(y, 15), mask2));
            y = _mm256_xor_si256(y, _mm256_srli_epi32(y, 17));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&tempered_[i]), y);
        }
        for (; i < state_size; ++i) {
            tempered_[i] = temper_one(mt_[i]);
        }
    }

#endif // COAST_MT11213B_AVX2

    // -----------------------------------------------------------------------
    // NEON path
    // -----------------------------------------------------------------------
#if defined(COAST_MT11213B_NEON)

    void twist_neon() {
        const uint32x4_t v_upper = vdupq_n_u32(upper_mask);
        const uint32x4_t v_lower = vdupq_n_u32(lower_mask);
        const uint32x4_t v_xor   = vdupq_n_u32(xor_mask);
        const uint32x4_t one     = vdupq_n_u32(1);
        const uint32x4_t zero    = vdupq_n_u32(0);

        // First section: 176 elements = 44 × 4, no scalar tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            uint32x4_t mt_i    = vld1q_u32(&mt_[i]);
            uint32x4_t mt_i1   = vld1q_u32(&mt_[i + 1]);
            uint32x4_t mt_shft = vld1q_u32(&mt_[i + shift_size]);

            uint32x4_t y       = vorrq_u32(vandq_u32(mt_i,  v_upper),
                                            vandq_u32(mt_i1, v_lower));
            uint32x4_t mag_val = vandq_u32(vsubq_u32(zero, vandq_u32(y, one)), v_xor);
            uint32x4_t result  = veorq_u32(veorq_u32(mt_shft, vshrq_n_u32(y, 1)), mag_val);
            vst1q_u32(&mt_[i], result);
        }
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second section: 174 elements, 43 × 4 + 2 scalar tail.
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            uint32x4_t mt_i    = vld1q_u32(&mt_[i]);
            uint32x4_t mt_i1   = vld1q_u32(&mt_[i + 1]);
            uint32x4_t mt_shft = vld1q_u32(&mt_[i + shift_size - state_size]);

            uint32x4_t y       = vorrq_u32(vandq_u32(mt_i,  v_upper),
                                            vandq_u32(mt_i1, v_lower));
            uint32x4_t mag_val = vandq_u32(vsubq_u32(zero, vandq_u32(y, one)), v_xor);
            uint32x4_t result  = veorq_u32(veorq_u32(mt_shft, vshrq_n_u32(y, 1)), mag_val);
            vst1q_u32(&mt_[i], result);
        }
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        result_type y       = (mt_[state_size - 1] & upper_mask) | (mt_[0] & lower_mask);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_neon() {
        const uint32x4_t mask1 = vdupq_n_u32(temper_b);
        const uint32x4_t mask2 = vdupq_n_u32(temper_c);

        // 351 = 87 × 4 + 3, so 3-element scalar tail.
        std::size_t i = 0;
        for (; i + batch_size <= state_size; i += batch_size) {
            uint32x4_t y = vld1q_u32(&mt_[i]);
            y = veorq_u32(y, vshrq_n_u32(y, 11));
            y = veorq_u32(y, vandq_u32(vshlq_n_u32(y,  7), mask1));
            y = veorq_u32(y, vandq_u32(vshlq_n_u32(y, 15), mask2));
            y = veorq_u32(y, vshrq_n_u32(y, 17));
            vst1q_u32(&tempered_[i], y);
        }
        for (; i < state_size; ++i) {
            tempered_[i] = temper_one(mt_[i]);
        }
    }

#endif // COAST_MT11213B_NEON

    // -----------------------------------------------------------------------
    // Scalar fallback (also used when no SIMD is detected)
    // -----------------------------------------------------------------------
#if !defined(COAST_MT11213B_AVX512) && !defined(COAST_MT11213B_AVX2) && !defined(COAST_MT11213B_NEON)

    void twist_scalar() {
        std::size_t i = 0;
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }
        result_type y       = (mt_[state_size - 1] & upper_mask) | (mt_[0] & lower_mask);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_scalar() {
        for (std::size_t i = 0; i < state_size; ++i) {
            tempered_[i] = temper_one(mt_[i]);
        }
    }

#endif // scalar fallback
};

} // namespace core::rng

#endif // COAST_MT11213B_SIMD_H
