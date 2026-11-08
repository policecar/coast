// rng_hash_bench — Demonstrate-stage micro-benchmark for ngm2.5
// (design/ngm2_5_flat_sparse.md §2, transcription map D1).
//
// Question: ngm2's response sweep measured latency-bound at ~52.9 cyc/synapse,
// and its per-synapse penalty draw is a SEQUENTIAL mt19937 stream — every draw
// depends on the generator state left by the previous one. Does replacing it
// with a stateless per-synapse hash (keyed on structural identity, consumption-
// order-free) relieve that dependency chain, single-threaded, in an ngm2-shaped
// loop?
//
// What this deliberately isolates: the RNG dependency chain in a loop with the
// exact shape of transcription-map region R2 step 3 (threshold skip, segment
// accumulate, conditional penalty with per-synapse clamp). Working set is
// ~2.4 MB (cache-resident single-threaded), so this measures the latency/ILP
// story, NOT the DRAM-bandwidth story — those are separate axes.
//
// Variants (policy template — same mechanism as the design's §8 seam):
//   none      — draw replaced by a constant: the RNG-cost floor. The gap
//               stream_mt -> none bounds what ANY rng change can buy.
//   stream_mt — production baseline: per-dendrite fast_rng_t (mt19937_simd)
//               through std::uniform_real_distribution<float>, as
//               hd_ngm2_dendrite.cpp:253,275.
//   hash_pcg  — the production GPU hash (src/core/gpu/shared_prng.wgsl:12-16)
//               on a (seed, step, dendrite, seg, input_idx) key.
//   hash_sm32 — splitmix32-style finalizer on the same key, for hash selection.
//
// Methodology: variants interleaved round-robin within each round (never
// sequential blocks — see benchmarks/perf_interleaved_ab_required lesson);
// first round discarded as warmup; medians + per-round paired ratios reported.
// Run pinned: taskset -c 4 ./rng_hash_bench
//
// Build:
//   g++ -O3 -march=native -std=c++23 -I include \
//       -o build/rng_hash_bench benchmarks/rng_hash_bench.cpp

#include "core/rng/fast_rng.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// ---- workload shape: matched mnist config at 1M-step maturity ------------
constexpr std::size_t DENDRITES   = 250;     // 50 neurons x 5 dendrites
constexpr std::size_t SYN_PER_D   = 1372;    // ~1.75 leaves/input x 784
constexpr std::size_t N_SYN       = DENDRITES * SYN_PER_D;  // 343,000
constexpr std::size_t INPUT_SIZE  = 784;
constexpr std::size_t SEG_CNT     = 16;      // depth 3: seg ids 1..15
constexpr float       PERM_THRES  = 0.3f;

constexpr int ROUNDS          = 8;           // round 0 = warmup, discarded
constexpr int STEPS_PER_ROUND = 200;

// ---- hashes ----------------------------------------------------------------

// Production hash, bit-identical to src/core/gpu/shared_prng.wgsl:12-16.
inline uint32_t pcg_hash(uint32_t input)
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Low-bias splitmix32-style finalizer (Degski/TheIronBorn constants).
inline uint32_t sm32_hash(uint32_t x)
{
    x += 0x9e3779b9u;
    x ^= x >> 16;  x *= 0x21f0aaadu;
    x ^= x >> 15;  x *= 0x735a2d97u;
    x ^= x >> 15;
    return x;
}

// Structural key, GPU-style multiplicative-xor composition (cf. the noise key
// at hd_ngm3_gpu.wgsl:299). The final key schedule is an implementation
// decision for ngm2.5; what matters here is the cost class.
inline uint32_t syn_key(uint32_t seed, uint32_t step, uint32_t dend,
                        uint32_t seg, uint32_t input_idx)
{
    return seed
         ^ (step      * 3266489917u)
         ^ (dend      * 2246822519u)
         ^ (seg       * 668265263u)
         ^ (input_idx * 374761393u);
}

inline float to_unit(uint32_t h)   // [0,1), matches pcg_uniform's 2^-32 scaling
{
    return static_cast<float>(h) * (1.0f / 4294967296.0f);
}

// ---- draw policies (the §8 seam, in miniature) ------------------------------

struct none_policy {                       // constant draw: RNG-cost floor
    static constexpr const char* name = "none     ";
    float half_max;
    explicit none_policy(uint32_t, float hm) : half_max(hm) {}
    void  begin_dendrite(std::size_t, uint32_t) {}
    float draw(uint32_t, uint32_t, uint32_t) { return 0.5f * half_max; }
};

struct stream_mt_policy {                  // production baseline (D1 as-is)
    static constexpr const char* name = "stream_mt";
    std::vector<fast_rng_t> rgens;         // one per dendrite, as in ngm2
    std::uniform_real_distribution<float> dis;
    fast_rng_t* cur = nullptr;
    stream_mt_policy(uint32_t seed, float hm) : dis(0.0f, hm)
    {
        rgens.reserve(DENDRITES);
        for (std::size_t d = 0; d < DENDRITES; ++d)
            rgens.emplace_back(static_cast<uint32_t>(seed + d));
    }
    void  begin_dendrite(std::size_t d, uint32_t) { cur = &rgens[d]; }
    float draw(uint32_t, uint32_t, uint32_t) { return dis(*cur); }
};

struct hash_pcg_policy {                   // stateless, structural key
    static constexpr const char* name = "hash_pcg ";
    uint32_t seed; float half_max; uint32_t dend = 0;
    hash_pcg_policy(uint32_t s, float hm) : seed(s), half_max(hm) {}
    void  begin_dendrite(std::size_t d, uint32_t) { dend = static_cast<uint32_t>(d); }
    float draw(uint32_t step, uint32_t seg, uint32_t input_idx)
    {
        return to_unit(pcg_hash(syn_key(seed, step, dend, seg, input_idx))) * half_max;
    }
};

struct hash_sm32_policy {
    static constexpr const char* name = "hash_sm32";
    uint32_t seed; float half_max; uint32_t dend = 0;
    hash_sm32_policy(uint32_t s, float hm) : seed(s), half_max(hm) {}
    void  begin_dendrite(std::size_t d, uint32_t) { dend = static_cast<uint32_t>(d); }
    float draw(uint32_t step, uint32_t seg, uint32_t input_idx)
    {
        return to_unit(sm32_hash(syn_key(seed, step, dend, seg, input_idx))) * half_max;
    }
};

// ---- the R2-shaped sweep -----------------------------------------------------

struct workload_t {
    std::vector<float>    perm;        // f32[N_SYN], Poisson(30)/100 as ngm2 init
    std::vector<uint16_t> input_idx;   // u16[N_SYN], input-major per dendrite
    std::vector<uint8_t>  seg;         // u8[N_SYN], mostly 1, some deeper
    std::vector<float>    input;       // f32[INPUT_SIZE], mnist-like sparsity
    float inv_sum = 0.0f, half_max = 0.0f, inv_perm_range = 0.0f;
};

workload_t make_workload(uint32_t seed)
{
    workload_t w;
    std::mt19937 gen(seed);
    std::poisson_distribution<> pois(30);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    w.perm.resize(N_SYN);
    w.input_idx.resize(N_SYN);
    w.seg.resize(N_SYN);
    for (std::size_t d = 0; d < DENDRITES; ++d) {
        // input-major: 784 base synapses then clones cycling through again,
        // approximating grown structure (clones share input_idx, sit deeper).
        for (std::size_t s = 0; s < SYN_PER_D; ++s) {
            const std::size_t i = d * SYN_PER_D + s;
            w.perm[i]      = std::clamp(static_cast<float>(pois(gen)) / 100.0f, 0.0f, 1.0f);
            w.input_idx[i] = static_cast<uint16_t>(s % INPUT_SIZE);
            w.seg[i]       = (s < INPUT_SIZE) ? uint8_t{1}
                                              : static_cast<uint8_t>(2 + (gen() % (SEG_CNT - 2)));
        }
    }
    w.input.resize(INPUT_SIZE);
    float sum = 0.0f, mx = 0.0f;
    for (auto &v : w.input) {
        v = (uni(gen) < 0.8f) ? 0.0f : uni(gen);   // ~80% zeros, mnist-like
        sum += v; mx = std::max(mx, v);
    }
    w.inv_sum        = 1.0f / sum;
    w.half_max       = mx / 2.0f;
    w.inv_perm_range = 1.0f / (1.0f - PERM_THRES);
    return w;
}

struct pass_result_t {
    double   ns_per_syn;
    double   checksum;        // anti-DCE sink; also a cross-run sanity value
    uint64_t penalties;       // penalty-branch count: confirms comparable work
};

template <class Policy>
pass_result_t run_variant(const workload_t &w, Policy &pol, uint32_t step_base)
{
    float seg_activity[SEG_CNT];
    double checksum = 0.0;
    uint64_t penalties = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t step = step_base; step < step_base + STEPS_PER_ROUND; ++step) {
        for (std::size_t d = 0; d < DENDRITES; ++d) {
            pol.begin_dendrite(d, step);
            std::fill(std::begin(seg_activity), std::end(seg_activity), 0.0f);
            const std::size_t base = d * SYN_PER_D;
            for (std::size_t s = 0; s < SYN_PER_D; ++s) {          // R2 step 3
                const float p = w.perm[base + s];
                if (p <= PERM_THRES) continue;
                const uint16_t ii  = w.input_idx[base + s];
                const uint8_t  sg  = w.seg[base + s];
                const float    inp = w.input[ii];
                seg_activity[sg] += inp;
                if (pol.draw(step, sg, ii) > inp) {
                    const float strength = (p - PERM_THRES) * w.inv_perm_range;
                    seg_activity[sg] -= strength * (1.0f - inp * w.inv_sum);
                    if (seg_activity[sg] < 0.0f) seg_activity[sg] = 0.0f;
                    ++penalties;
                }
            }
            for (float v : seg_activity) checksum += v;            // sink
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return { ns / (static_cast<double>(N_SYN) * STEPS_PER_ROUND), checksum, penalties };
}

// ---- hash-quality quick check ------------------------------------------------

template <uint32_t (*H)(uint32_t)>
void quality_check(const char *name, uint32_t seed)
{
    // Realistic key traffic: sweep (step, dend, seg, input_idx) as production
    // would. 2^24 draws; report mean, variance, 256-bucket chi-square
    // (df=255: expect 255 +- ~22.6 at 1 sigma).
    constexpr uint64_t N = 1u << 24;
    double sum = 0.0, sum2 = 0.0;
    std::vector<uint64_t> bucket(256, 0);
    uint64_t n = 0;
    for (uint32_t step = 0; n < N; ++step)
        for (uint32_t d = 0; d < DENDRITES && n < N; ++d)
            for (uint32_t s = 0; s < 64 && n < N; ++s, ++n) {
                const uint32_t h = H(syn_key(seed, step, d, 1 + (s % 15),
                                             (s * 37u) % INPUT_SIZE));
                const double u = to_unit(h);
                sum += u; sum2 += u * u;
                ++bucket[h >> 24];
            }
    const double mean = sum / N;
    const double var  = sum2 / N - mean * mean;
    const double exp_per_bucket = static_cast<double>(N) / 256.0;
    double chi2 = 0.0;
    for (auto b : bucket) {
        const double dlt = static_cast<double>(b) - exp_per_bucket;
        chi2 += dlt * dlt / exp_per_bucket;
    }
    std::printf("  %-10s mean=%.6f (exp 0.5)  var=%.6f (exp %.6f)  chi2[df=255]=%.1f\n",
                name, mean, var, 1.0 / 12.0, chi2);
}

} // namespace

int main()
{
    const uint32_t seed = 0xC0A57u;
    const workload_t w  = make_workload(seed);

    std::printf("rng_hash_bench: %zu dendrites x %zu syn = %zu synapses, "
                "%d steps/round, %d rounds (round 0 = warmup)\n\n",
                DENDRITES, SYN_PER_D, N_SYN, STEPS_PER_ROUND, ROUNDS - 1);

    std::printf("hash quality (2^24 draws, production-shaped keys):\n");
    quality_check<pcg_hash>("pcg_hash", seed);
    quality_check<sm32_hash>("sm32_hash", seed);
    std::printf("\n");

    // Policies persist across rounds (stream_mt keeps its generator state
    // across "steps", as production does).
    none_policy      p_none(seed, w.half_max);
    stream_mt_policy p_mt  (seed, w.half_max);
    hash_pcg_policy  p_pcg (seed, w.half_max);
    hash_sm32_policy p_sm  (seed, w.half_max);

    constexpr int NV = 4;
    double ns[NV][ROUNDS];
    pass_result_t last[NV];

    for (int r = 0; r < ROUNDS; ++r) {                 // interleaved rounds
        const uint32_t step_base = static_cast<uint32_t>(r) * STEPS_PER_ROUND;
        last[0] = run_variant(w, p_none, step_base); ns[0][r] = last[0].ns_per_syn;
        last[1] = run_variant(w, p_mt,   step_base); ns[1][r] = last[1].ns_per_syn;
        last[2] = run_variant(w, p_pcg,  step_base); ns[2][r] = last[2].ns_per_syn;
        last[3] = run_variant(w, p_sm,   step_base); ns[3][r] = last[3].ns_per_syn;
    }

    const char *names[NV] = { none_policy::name, stream_mt_policy::name,
                              hash_pcg_policy::name, hash_sm32_policy::name };

    std::printf("per-round ns/synapse (round 0 = warmup, excluded from stats):\n");
    std::printf("  %-10s", "round");
    for (int r = 1; r < ROUNDS; ++r) std::printf("   r%-5d", r);
    std::printf("   median\n");
    double median[NV];
    for (int v = 0; v < NV; ++v) {
        std::vector<double> m(ns[v] + 1, ns[v] + ROUNDS);
        std::sort(m.begin(), m.end());
        median[v] = m[m.size() / 2];
        std::printf("  %-10s", names[v]);
        for (int r = 1; r < ROUNDS; ++r) std::printf("  %6.3f ", ns[v][r]);
        std::printf("  %6.3f\n", median[v]);
    }

    std::printf("\npaired per-round ratio vs stream_mt (>1 = faster than baseline):\n");
    for (int v = 0; v < NV; ++v) {
        if (v == 1) continue;
        std::printf("  %-10s", names[v]);
        for (int r = 1; r < ROUNDS; ++r) std::printf("  %6.3fx", ns[1][r] / ns[v][r]);
        std::printf("   median %.3fx\n", median[1] / median[v]);
    }

    std::printf("\nwork parity (last round): penalty-branch fraction of %zu syn-steps:\n",
                static_cast<std::size_t>(N_SYN) * STEPS_PER_ROUND);
    for (int v = 0; v < NV; ++v)
        std::printf("  %-10s penalties=%.4f  checksum=%.3f\n", names[v],
                    static_cast<double>(last[v].penalties) / (static_cast<double>(N_SYN) * STEPS_PER_ROUND),
                    last[v].checksum);

    return 0;
}
