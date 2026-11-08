// h5browse stats-kernel throughput benchmark (manual; NOT a ctest).
//
// The T1 experiments measured the identical accumulate source at 5.01 vs
// 3.46 GB/s in two different TU compositions, so the hot kernel's throughput
// is worth re-measuring after changes to stats_kernel / its TU (the DVD
// production rule on multi-lane accumulation). This used to be a ctest
// asserting a 2 GB/s floor (~4.4 GB/s on the reference machine). A
// throughput floor is a verdict that depends on the
// machine and its load (CLAUDE.md "Testing"), so it is a benchmark now: it
// reports the number and checks only that the kernel counted every value.
// A regression to the ~0.5 GB/s scalar path shows here as a ~10x drop.
//
// Usage: bench_h5b_kernel_throughput [reps]      (default 3, best-of)
// Prints the best f32 accumulate throughput in GB/s.

#include "h5browse/scan_stats.h"
#include "h5browse/stats_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

int main(int argc, char** argv)
{
    const int reps = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 3;

    constexpr std::size_t n = 64u << 20; // 64 Mi floats = 256 MB
    std::vector<float>    v(n);
    std::mt19937          rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& x : v) x = dist(rng);

    // Warm-up pass (page-in + frequency ramp), then timed reps, best-of.
    h5b::scan_stats warm;
    h5b::accumulate_stats<float>(warm, std::span<const float>(v));
    if (warm.count != n) {
        std::fprintf(stderr, "kernel counted %llu of %zu values\n",
                     static_cast<unsigned long long>(warm.count), n);
        return 1;
    }

    double best_gbps = 0.0;
    for (int rep = 0; rep < reps; ++rep) {
        h5b::scan_stats s;
        const auto      t0 = std::chrono::steady_clock::now();
        h5b::accumulate_stats<float>(s, std::span<const float>(v));
        const auto t1 = std::chrono::steady_clock::now();
        if (s.count != n) return 1;
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        best_gbps = std::max(best_gbps,
                             static_cast<double>(n * sizeof(float)) / secs / 1.0e9);
    }
    std::printf("stats kernel f32 throughput: %.2f GB/s (best of %d; reference ~4.4)\n",
                best_gbps, reps);
    return 0;
}
