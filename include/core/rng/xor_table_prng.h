#ifndef COAST_XOR_TABLE_PRNG_H
#define COAST_XOR_TABLE_PRNG_H

#include <cstdint>
#include <limits>

#if defined(__AVX512F__)
    #define COAST_XTBL_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define COAST_XTBL_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define COAST_XTBL_NEON 1
    #include <arm_neon.h>
#endif

namespace core::rng {

/// XOR-table PRNG: five prime-length read-only tables combined with XOR.
///
/// Idea: maintain five tables rs1..rs5 of prime lengths 73, 79, 83, 89, 97
/// (total ~1.6 KB, permanently L1d-resident).  A running phase p1..p5 steps
/// through each table independently.  Output at step i:
///
///   rs1[i%73] ^ rs2[i%79] ^ rs3[i%83] ^ rs4[i%89] ^ rs5[i%97]
///
/// Period = LCM(73,79,83,89,97) = 73×79×83×89×97 = 4,132,280,413 (all prime).
///
/// SIMD parallelism: because i, i+1, …, i+15 access *consecutive* indices in
/// each table, the 16 lookups per table reduce to a single contiguous SIMD
/// load — no gather instruction is required.  Each table is padded with 16
/// wrap-around copies at the end so the load is always in-bounds regardless
/// of phase alignment.
///
/// Refill (AVX-512): 5 loadu + 4 xor + 1 store — about 10 SIMD instructions
/// for 16 output values.  The tables are read-only after seeding; the only
/// write per refill is the 64-byte output buffer.
///
/// State: ~2.1 KB total (tables + padding + buffer + phases).
/// Refill every 16 output values (BUF_SIZE), no large state rewrites.
///
class xor_table_prng {
public:
    using result_type = uint32_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type{0}; }

    // Five prime table sizes — LCM = 4,132,280,413
    static constexpr int S1 = 73, S2 = 79, S3 = 83, S4 = 89, S5 = 97;

    // Wrap-around padding: 16 elements ensures a 16-wide SIMD load starting
    // at any phase in [0, Si-1] is always within the allocated array.
    static constexpr int PAD = 16;

    // Output buffer: four AVX-512 vectors (BUF_SIZE=64).
    // Refill fires every 64 calls — 4× less often than BUF_SIZE=16 — reducing
    // branch overhead while keeping the same amortised refill cost per output.
    // Each refill does 4 SIMD passes; phases advance by 16 inside each pass.
    // Since 64 < min(S1..S5)=73, a single conditional subtract still handles wrap.
    static constexpr int BUF_SIZE = 64;

    xor_table_prng() { seed(5489u); }
    explicit xor_table_prng(result_type s) { seed(s); }

    void seed(result_type s) {
        // Expand seed with splitmix64 to fill tables with good initial values.
        uint64_t z = static_cast<uint64_t>(s) + 0x9e3779b97f4a7c15ULL;
        auto sm64 = [&]() -> uint32_t {
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z ^= (z >> 31);
            z += 0x9e3779b97f4a7c15ULL;   // keep advancing z
            return static_cast<uint32_t>(z);
        };
        auto fill = [&](uint32_t* t, int n) {
            for (int i = 0; i < n;   ++i) t[i]       = sm64();
            for (int i = 0; i < PAD; ++i) t[n + i]   = t[i]; // wrap-around copies
        };
        fill(rs1_, S1);  fill(rs2_, S2);  fill(rs3_, S3);
        fill(rs4_, S4);  fill(rs5_, S5);
        p1_ = p2_ = p3_ = p4_ = p5_ = 0;
        idx_ = BUF_SIZE;  // trigger refill on first call
    }

    result_type operator()() {
        if (idx_ >= BUF_SIZE) [[unlikely]] { refill(); }
        return buf_[idx_++];
    }

    void discard(unsigned long long n) {
        while (n > 0) {
            const unsigned long long avail =
                static_cast<unsigned long long>(BUF_SIZE - idx_);
            if (n <= avail) { idx_ += static_cast<int>(n); return; }
            n -= avail;
            refill();
        }
    }

private:
    // Tables: Si elements + PAD wrap-around copies.
    // Each starts on a 64-byte boundary; their combined size (~2 KB) fits in L1d.
    alignas(64) uint32_t rs1_[S1 + PAD];
    alignas(64) uint32_t rs2_[S2 + PAD];
    alignas(64) uint32_t rs3_[S3 + PAD];
    alignas(64) uint32_t rs4_[S4 + PAD];
    alignas(64) uint32_t rs5_[S5 + PAD];
    alignas(64) uint32_t buf_[BUF_SIZE];

    int p1_, p2_, p3_, p4_, p5_;  // current phase in each table ∈ [0, Si)
    int idx_;                       // read position in buf_

    // ------------------------------------------------------------------
    // Refill: four passes of 16 values each (BUF_SIZE=64), manually unrolled.
    //
    // Written as four explicit blocks (no loop) so the compiler sees a flat
    // straight-line sequence — same structure as the BUF_SIZE=16 single block.
    // This tests the hypothesis that the +10% instruction overhead of the
    // loop-based BUF_SIZE=64 variant came from loop overhead and loop-carried
    // dependencies on p1_..p5_, not from an intrinsic cost of the 4-pass design.
    //
    // PAD=16 remains sufficient: each block reads rs_[p] to rs_[p+15].
    // Since p ∈ [0, Si-1], max index = Si-1+15 = Si+14 < Si+PAD=Si+16. ✓
    //
    // Scalar fallback uses running counters (p+k offset would OOB:
    // max p+k = 72+63 = 135 > 73+16 = 89).
    // ------------------------------------------------------------------

// Helper macro: one AVX-512 pass — load, XOR, store, advance phases.
// Defined only within this function scope via the pragma/lambda-less approach.
#define XTBL_PASS_AVX512(KOFF)                                              \
    do {                                                                    \
        const __m512i _v = _mm512_xor_si512(                               \
            _mm512_loadu_si512(&rs1_[p1_]),                                 \
            _mm512_xor_si512(                                               \
                _mm512_loadu_si512(&rs2_[p2_]),                             \
                _mm512_xor_si512(                                           \
                    _mm512_loadu_si512(&rs3_[p3_]),                         \
                    _mm512_xor_si512(                                       \
                        _mm512_loadu_si512(&rs4_[p4_]),                     \
                        _mm512_loadu_si512(&rs5_[p5_])))));                 \
        _mm512_store_si512(&buf_[KOFF], _v);                                \
        p1_ += 16; if (p1_ >= S1) p1_ -= S1;                               \
        p2_ += 16; if (p2_ >= S2) p2_ -= S2;                               \
        p3_ += 16; if (p3_ >= S3) p3_ -= S3;                               \
        p4_ += 16; if (p4_ >= S4) p4_ -= S4;                               \
        p5_ += 16; if (p5_ >= S5) p5_ -= S5;                               \
    } while(0)

    void refill() noexcept {
#ifdef COAST_XTBL_AVX512
        XTBL_PASS_AVX512( 0);
        XTBL_PASS_AVX512(16);
        XTBL_PASS_AVX512(32);
        XTBL_PASS_AVX512(48);

#elif defined(COAST_XTBL_AVX2)
        // 8 explicit passes of 8 elements.
        #define XTBL_PASS_AVX2(KOFF)                                            \
            do {                                                                \
                const __m256i _v = _mm256_xor_si256(                           \
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&rs1_[p1_])), \
                    _mm256_xor_si256(                                           \
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&rs2_[p2_])), \
                        _mm256_xor_si256(                                       \
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&rs3_[p3_])), \
                            _mm256_xor_si256(                                   \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&rs4_[p4_])), \
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&rs5_[p5_])))))); \
                _mm256_store_si256(reinterpret_cast<__m256i*>(&buf_[KOFF]), _v); \
                p1_ += 8; if (p1_ >= S1) p1_ -= S1;                            \
                p2_ += 8; if (p2_ >= S2) p2_ -= S2;                            \
                p3_ += 8; if (p3_ >= S3) p3_ -= S3;                            \
                p4_ += 8; if (p4_ >= S4) p4_ -= S4;                            \
                p5_ += 8; if (p5_ >= S5) p5_ -= S5;                            \
            } while(0)
        XTBL_PASS_AVX2( 0); XTBL_PASS_AVX2( 8); XTBL_PASS_AVX2(16); XTBL_PASS_AVX2(24);
        XTBL_PASS_AVX2(32); XTBL_PASS_AVX2(40); XTBL_PASS_AVX2(48); XTBL_PASS_AVX2(56);
        #undef XTBL_PASS_AVX2

#elif defined(COAST_XTBL_NEON)
        // 16 explicit passes of 4 elements.
        #define XTBL_PASS_NEON(KOFF)                                            \
            do {                                                                \
                uint32x4_t _v = veorq_u32(vld1q_u32(&rs1_[p1_]),               \
                    veorq_u32(vld1q_u32(&rs2_[p2_]),                            \
                    veorq_u32(vld1q_u32(&rs3_[p3_]),                            \
                    veorq_u32(vld1q_u32(&rs4_[p4_]),                            \
                              vld1q_u32(&rs5_[p5_])))));                        \
                vst1q_u32(&buf_[KOFF], _v);                                     \
                p1_ += 4; if (p1_ >= S1) p1_ -= S1;                            \
                p2_ += 4; if (p2_ >= S2) p2_ -= S2;                            \
                p3_ += 4; if (p3_ >= S3) p3_ -= S3;                            \
                p4_ += 4; if (p4_ >= S4) p4_ -= S4;                            \
                p5_ += 4; if (p5_ >= S5) p5_ -= S5;                            \
            } while(0)
        XTBL_PASS_NEON( 0); XTBL_PASS_NEON( 4); XTBL_PASS_NEON( 8); XTBL_PASS_NEON(12);
        XTBL_PASS_NEON(16); XTBL_PASS_NEON(20); XTBL_PASS_NEON(24); XTBL_PASS_NEON(28);
        XTBL_PASS_NEON(32); XTBL_PASS_NEON(36); XTBL_PASS_NEON(40); XTBL_PASS_NEON(44);
        XTBL_PASS_NEON(48); XTBL_PASS_NEON(52); XTBL_PASS_NEON(56); XTBL_PASS_NEON(60);
        #undef XTBL_PASS_NEON

#else
        // Scalar: running counters to avoid OOB on padded arrays.
        int q1=p1_, q2=p2_, q3=p3_, q4=p4_, q5=p5_;
        for (int k = 0; k < BUF_SIZE; ++k) {
            buf_[k] = rs1_[q1] ^ rs2_[q2] ^ rs3_[q3] ^ rs4_[q4] ^ rs5_[q5];
            if (++q1 >= S1) q1 = 0;
            if (++q2 >= S2) q2 = 0;
            if (++q3 >= S3) q3 = 0;
            if (++q4 >= S4) q4 = 0;
            if (++q5 >= S5) q5 = 0;
        }
        p1_=q1; p2_=q2; p3_=q3; p4_=q4; p5_=q5;
#endif
        idx_ = 0;
    }

#undef XTBL_PASS_AVX512
};

} // namespace core::rng

#endif // COAST_XOR_TABLE_PRNG_H
