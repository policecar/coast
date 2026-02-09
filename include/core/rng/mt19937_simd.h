#ifndef COAST_MT19937_SIMD_H
#define COAST_MT19937_SIMD_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

// SIMD detection and includes
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #define COAST_RNG_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define COAST_RNG_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define COAST_RNG_NEON 1
    #include <arm_neon.h>
#endif

namespace core::rng {

/// SIMD-optimized Mersenne Twister 19937 generator.
/// Produces identical sequences to std::mt19937 with the same seed.
/// Uses AVX512/AVX2/NEON SIMD to parallelize both twist and tempering steps.
class mt19937_simd {
public:
    using result_type = std::uint32_t;

    // MT19937 parameters
    static constexpr std::size_t state_size = 624;
    static constexpr std::size_t shift_size = 397;
    static constexpr result_type xor_mask = 0x9908b0df;
    static constexpr result_type default_seed = 5489u;

    // Batch size based on available SIMD
#if defined(COAST_RNG_AVX512)
    static constexpr std::size_t batch_size = 16;
#elif defined(COAST_RNG_AVX2)
    static constexpr std::size_t batch_size = 8;
#elif defined(COAST_RNG_NEON)
    static constexpr std::size_t batch_size = 4;
#else
    static constexpr std::size_t batch_size = 1;
#endif

private:
    // Pad state array to allow unaligned SIMD loads at boundaries
    // We need to read mt_[i+1] which at i=623 would be mt_[624], so add padding
    alignas(64) std::array<result_type, state_size + batch_size> mt_;
    alignas(64) std::array<result_type, state_size> tempered_;
    std::size_t mti_ = state_size;

public:
    mt19937_simd() { seed(default_seed); }

    explicit mt19937_simd(result_type value) { seed(value); }

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
    /// Generate next batch of state values and temper them all
    void generate_and_temper() {
#if defined(COAST_RNG_AVX512)
        twist_avx512();
        temper_all_avx512();
#elif defined(COAST_RNG_AVX2)
        twist_avx2();
        temper_all_avx2();
#elif defined(COAST_RNG_NEON)
        twist_neon();
        temper_all_neon();
#else
        twist_scalar();
        temper_all_scalar();
#endif
        mti_ = 0;
    }

#if defined(COAST_RNG_AVX512)
    void twist_avx512() {
        const __m512i upper_mask = _mm512_set1_epi32(static_cast<int>(0x80000000u));
        const __m512i lower_mask = _mm512_set1_epi32(static_cast<int>(0x7fffffffu));
        const __m512i xor_mask_vec = _mm512_set1_epi32(static_cast<int>(xor_mask));
        const __m512i one = _mm512_set1_epi32(1);

        // First loop: i = 0 to state_size - shift_size - 1 (0 to 226)
        // mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ mag01[y & 1]
        // where y = (mt_[i] & upper) | (mt_[i+1] & lower)
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            __m512i mt_i = _mm512_load_si512(reinterpret_cast<const __m512i*>(&mt_[i]));
            __m512i mt_i1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + 1]));
            __m512i mt_shift = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + shift_size]));

            // y = (mt_[i] & upper_mask) | (mt_[i+1] & lower_mask)
            __m512i y = _mm512_or_si512(
                _mm512_and_si512(mt_i, upper_mask),
                _mm512_and_si512(mt_i1, lower_mask)
            );

            // mag01[y & 1]: if lowest bit is 1, use xor_mask, else 0
            // Create mask: -(y & 1) gives 0 or 0xFFFFFFFF
            __m512i low_bit = _mm512_and_si512(y, one);
            __m512i mag_mask = _mm512_sub_epi32(_mm512_setzero_si512(), low_bit);
            __m512i mag_val = _mm512_and_si512(mag_mask, xor_mask_vec);

            // result = mt_[i + shift_size] ^ (y >> 1) ^ mag_val
            __m512i result = _mm512_xor_si512(mt_shift, _mm512_srli_epi32(y, 1));
            result = _mm512_xor_si512(result, mag_val);

            _mm512_store_si512(reinterpret_cast<__m512i*>(&mt_[i]), result);
        }

        // Handle remaining elements in first section (scalar)
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second loop: i = state_size - shift_size to state_size - 2 (227 to 622)
        // mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ mag01[y & 1]
        // Note: i is not aligned here, so use unaligned load/store
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            __m512i mt_i = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i]));
            __m512i mt_i1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + 1]));
            __m512i mt_shift = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&mt_[i + shift_size - state_size]));

            __m512i y = _mm512_or_si512(
                _mm512_and_si512(mt_i, upper_mask),
                _mm512_and_si512(mt_i1, lower_mask)
            );

            __m512i low_bit = _mm512_and_si512(y, one);
            __m512i mag_mask = _mm512_sub_epi32(_mm512_setzero_si512(), low_bit);
            __m512i mag_val = _mm512_and_si512(mag_mask, xor_mask_vec);

            __m512i result = _mm512_xor_si512(mt_shift, _mm512_srli_epi32(y, 1));
            result = _mm512_xor_si512(result, mag_val);

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&mt_[i]), result);
        }

        // Handle remaining elements in second section (scalar)
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Last element wraps around to mt_[0]
        result_type y = (mt_[state_size - 1] & 0x80000000u) | (mt_[0] & 0x7fffffffu);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_avx512() {
        const __m512i mask1 = _mm512_set1_epi32(static_cast<int>(0x9d2c5680u));
        const __m512i mask2 = _mm512_set1_epi32(static_cast<int>(0xefc60000u));

        for (std::size_t i = 0; i < state_size; i += batch_size) {
            __m512i y = _mm512_load_si512(reinterpret_cast<const __m512i*>(&mt_[i]));

            y = _mm512_xor_si512(y, _mm512_srli_epi32(y, 11));
            y = _mm512_xor_si512(y, _mm512_and_si512(_mm512_slli_epi32(y, 7), mask1));
            y = _mm512_xor_si512(y, _mm512_and_si512(_mm512_slli_epi32(y, 15), mask2));
            y = _mm512_xor_si512(y, _mm512_srli_epi32(y, 18));

            _mm512_store_si512(reinterpret_cast<__m512i*>(&tempered_[i]), y);
        }
    }
#endif

#if defined(COAST_RNG_AVX2)
    void twist_avx2() {
        const __m256i upper_mask = _mm256_set1_epi32(static_cast<int>(0x80000000u));
        const __m256i lower_mask = _mm256_set1_epi32(static_cast<int>(0x7fffffffu));
        const __m256i xor_mask_vec = _mm256_set1_epi32(static_cast<int>(xor_mask));
        const __m256i one = _mm256_set1_epi32(1);

        // First loop: i = 0 to state_size - shift_size - 1 (0 to 226)
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            __m256i mt_i = _mm256_load_si256(reinterpret_cast<const __m256i*>(&mt_[i]));
            __m256i mt_i1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + 1]));
            __m256i mt_shift = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + shift_size]));

            __m256i y = _mm256_or_si256(
                _mm256_and_si256(mt_i, upper_mask),
                _mm256_and_si256(mt_i1, lower_mask)
            );

            __m256i low_bit = _mm256_and_si256(y, one);
            __m256i mag_mask = _mm256_sub_epi32(_mm256_setzero_si256(), low_bit);
            __m256i mag_val = _mm256_and_si256(mag_mask, xor_mask_vec);

            __m256i result = _mm256_xor_si256(mt_shift, _mm256_srli_epi32(y, 1));
            result = _mm256_xor_si256(result, mag_val);

            _mm256_store_si256(reinterpret_cast<__m256i*>(&mt_[i]), result);
        }

        // Handle remaining elements in first section (scalar)
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second loop: i = state_size - shift_size to state_size - 2 (227 to 622)
        // Note: i is not aligned here, so use unaligned load/store
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            __m256i mt_i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i]));
            __m256i mt_i1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + 1]));
            __m256i mt_shift = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mt_[i + shift_size - state_size]));

            __m256i y = _mm256_or_si256(
                _mm256_and_si256(mt_i, upper_mask),
                _mm256_and_si256(mt_i1, lower_mask)
            );

            __m256i low_bit = _mm256_and_si256(y, one);
            __m256i mag_mask = _mm256_sub_epi32(_mm256_setzero_si256(), low_bit);
            __m256i mag_val = _mm256_and_si256(mag_mask, xor_mask_vec);

            __m256i result = _mm256_xor_si256(mt_shift, _mm256_srli_epi32(y, 1));
            result = _mm256_xor_si256(result, mag_val);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&mt_[i]), result);
        }

        // Handle remaining elements in second section (scalar)
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Last element wraps around
        result_type y = (mt_[state_size - 1] & 0x80000000u) | (mt_[0] & 0x7fffffffu);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_avx2() {
        const __m256i mask1 = _mm256_set1_epi32(static_cast<int>(0x9d2c5680u));
        const __m256i mask2 = _mm256_set1_epi32(static_cast<int>(0xefc60000u));

        for (std::size_t i = 0; i < state_size; i += batch_size) {
            __m256i y = _mm256_load_si256(reinterpret_cast<const __m256i*>(&mt_[i]));

            y = _mm256_xor_si256(y, _mm256_srli_epi32(y, 11));
            y = _mm256_xor_si256(y, _mm256_and_si256(_mm256_slli_epi32(y, 7), mask1));
            y = _mm256_xor_si256(y, _mm256_and_si256(_mm256_slli_epi32(y, 15), mask2));
            y = _mm256_xor_si256(y, _mm256_srli_epi32(y, 18));

            _mm256_store_si256(reinterpret_cast<__m256i*>(&tempered_[i]), y);
        }
    }
#endif

#if defined(COAST_RNG_NEON)
    void twist_neon() {
        const uint32x4_t upper_mask = vdupq_n_u32(0x80000000u);
        const uint32x4_t lower_mask = vdupq_n_u32(0x7fffffffu);
        const uint32x4_t xor_mask_vec = vdupq_n_u32(xor_mask);
        const uint32x4_t one = vdupq_n_u32(1);
        const uint32x4_t zero = vdupq_n_u32(0);

        // First loop: i = 0 to state_size - shift_size - 1 (0 to 226)
        std::size_t i = 0;
        for (; i + batch_size <= state_size - shift_size; i += batch_size) {
            uint32x4_t mt_i = vld1q_u32(&mt_[i]);
            uint32x4_t mt_i1 = vld1q_u32(&mt_[i + 1]);
            uint32x4_t mt_shift = vld1q_u32(&mt_[i + shift_size]);

            // y = (mt_[i] & upper_mask) | (mt_[i+1] & lower_mask)
            uint32x4_t y = vorrq_u32(
                vandq_u32(mt_i, upper_mask),
                vandq_u32(mt_i1, lower_mask)
            );

            // mag01[y & 1]: if lowest bit is 1, use xor_mask, else 0
            // Create mask: -(y & 1) gives 0 or 0xFFFFFFFF
            uint32x4_t low_bit = vandq_u32(y, one);
            uint32x4_t mag_mask = vsubq_u32(zero, low_bit);
            uint32x4_t mag_val = vandq_u32(mag_mask, xor_mask_vec);

            // result = mt_[i + shift_size] ^ (y >> 1) ^ mag_val
            uint32x4_t result = veorq_u32(mt_shift, vshrq_n_u32(y, 1));
            result = veorq_u32(result, mag_val);

            vst1q_u32(&mt_[i], result);
        }

        // Handle remaining elements in first section (scalar)
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Second loop: i = state_size - shift_size to state_size - 2 (227 to 622)
        for (; i + batch_size <= state_size - 1; i += batch_size) {
            uint32x4_t mt_i = vld1q_u32(&mt_[i]);
            uint32x4_t mt_i1 = vld1q_u32(&mt_[i + 1]);
            uint32x4_t mt_shift = vld1q_u32(&mt_[i + shift_size - state_size]);

            uint32x4_t y = vorrq_u32(
                vandq_u32(mt_i, upper_mask),
                vandq_u32(mt_i1, lower_mask)
            );

            uint32x4_t low_bit = vandq_u32(y, one);
            uint32x4_t mag_mask = vsubq_u32(zero, low_bit);
            uint32x4_t mag_val = vandq_u32(mag_mask, xor_mask_vec);

            uint32x4_t result = veorq_u32(mt_shift, vshrq_n_u32(y, 1));
            result = veorq_u32(result, mag_val);

            vst1q_u32(&mt_[i], result);
        }

        // Handle remaining elements in second section (scalar)
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & 0x80000000u) | (mt_[i + 1] & 0x7fffffffu);
            mt_[i] = mt_[i + shift_size - state_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }

        // Last element wraps around
        result_type y = (mt_[state_size - 1] & 0x80000000u) | (mt_[0] & 0x7fffffffu);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_neon() {
        const uint32x4_t mask1 = vdupq_n_u32(0x9d2c5680u);
        const uint32x4_t mask2 = vdupq_n_u32(0xefc60000u);

        for (std::size_t i = 0; i < state_size; i += batch_size) {
            uint32x4_t y = vld1q_u32(&mt_[i]);

            // y ^= (y >> 11)
            y = veorq_u32(y, vshrq_n_u32(y, 11));
            // y ^= (y << 7) & mask1
            y = veorq_u32(y, vandq_u32(vshlq_n_u32(y, 7), mask1));
            // y ^= (y << 15) & mask2
            y = veorq_u32(y, vandq_u32(vshlq_n_u32(y, 15), mask2));
            // y ^= (y >> 18)
            y = veorq_u32(y, vshrq_n_u32(y, 18));

            vst1q_u32(&tempered_[i], y);
        }
    }
#endif

#if !defined(COAST_RNG_AVX512) && !defined(COAST_RNG_AVX2) && !defined(COAST_RNG_NEON)
    void twist_scalar() {
        static constexpr result_type upper_mask = 0x80000000u;
        static constexpr result_type lower_mask = 0x7fffffffu;

        std::size_t i = 0;
        for (; i < state_size - shift_size; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + shift_size] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }
        for (; i < state_size - 1; ++i) {
            result_type y = (mt_[i] & upper_mask) | (mt_[i + 1] & lower_mask);
            mt_[i] = mt_[i + (shift_size - state_size)] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
        }
        result_type y = (mt_[state_size - 1] & upper_mask) | (mt_[0] & lower_mask);
        mt_[state_size - 1] = mt_[shift_size - 1] ^ (y >> 1) ^ ((y & 1u) ? xor_mask : 0u);
    }

    void temper_all_scalar() {
        for (std::size_t i = 0; i < state_size; ++i) {
            result_type y = mt_[i];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= (y >> 18);
            tempered_[i] = y;
        }
    }
#endif
};

} // namespace core::rng

#endif // COAST_MT19937_SIMD_H
