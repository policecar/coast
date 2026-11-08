//
// ensemble_code.h — pure, GUI-free synthetic "ensemble code" generator for the
// Metric Explorer (Layer 5).
//
// Produces vectors shaped to look like neuron-group population outputs: each code
// is sparse (k high-amplitude "winners" near 1), the rest suppressed by inhibition
// toward a bounded low-amplitude background. Codes are drawn from M distinct
// "inputs" (prototypes), so a similarity matrix over the result shows M separable
// blocks — letting us watch how a metric behaves on inputs whose ground-truth
// cluster structure we control.
//
// Header-only `inline` so the GUI tool and the unit test just include it. Cold
// path: precondition violations throw (project convention).
//
// Geometry: a code has D real entries, zero-padded to S*S with S = ceil(sqrt(D)),
// so it can be displayed either as an S*S square or an S^2 x 1 strip. The padding
// is metric-neutral (shared trailing zeros add nothing to cosine or Minkowski).
//

#ifndef COAST_ENSEMBLE_CODE_H
#define COAST_ENSEMBLE_CODE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace coast {

struct ensemble_code_params {
    int D = 256;             // neurons per code (vector length before padding)
    int N = 64;              // number of codes generated
    int M = 4;               // distinct inputs / prototypes (clusters)
    int k = 8;               // winners per code (sparsity)
    float swap_frac  = 0.25f; // fraction of winners re-rolled off the prototype (per-code variation)
    float amp_var    = 0.15f; // per-winner amplitude jitter: winner = recognition * (1 - U[0, amp_var])
    float inhibition = 0.8f;  // non-winner suppression in [0,1]; 1 -> others ~0
    float noise      = 0.05f; // background noise std-dev (bounded to 3*noise)
    float recognition= 1.0f;  // r in [0,1]; 0 -> no winners, background-only "noise code"
    std::uint32_t seed = 0;
};

struct ensemble_codes {
    std::vector<float> data;  // N * side*side, row-major, zero-padded past D
    int side  = 0;            // S = ceil(sqrt(D))
    int count = 0;            // N
};

/// Generate N synthetic ensemble codes. pre: D>=1, 0<=k<=D, N>=0, M>=1, and
/// swap_frac/amp_var/inhibition/recognition in [0,1], noise>=0 (throws otherwise).
/// Output values are in [0,1]; entries with index >= D inside each padded code
/// are zero; same params + seed produce an identical buffer.
inline ensemble_codes generate_ensemble_codes(const ensemble_code_params& p)
{
    if (p.D < 1)                                   throw std::invalid_argument("generate_ensemble_codes: D must be >= 1");
    if (p.N < 0)                                   throw std::invalid_argument("generate_ensemble_codes: N must be >= 0");
    if (p.M < 1)                                   throw std::invalid_argument("generate_ensemble_codes: M must be >= 1");
    if (p.k < 0 || p.k > p.D)                      throw std::invalid_argument("generate_ensemble_codes: k must be in [0,D]");
    if (!(p.swap_frac  >= 0.0f && p.swap_frac  <= 1.0f)) throw std::invalid_argument("generate_ensemble_codes: swap_frac must be in [0,1]");
    if (!(p.amp_var    >= 0.0f && p.amp_var    <= 1.0f)) throw std::invalid_argument("generate_ensemble_codes: amp_var must be in [0,1]");
    if (!(p.inhibition >= 0.0f && p.inhibition <= 1.0f)) throw std::invalid_argument("generate_ensemble_codes: inhibition must be in [0,1]");
    if (!(p.recognition>= 0.0f && p.recognition<= 1.0f)) throw std::invalid_argument("generate_ensemble_codes: recognition must be in [0,1]");
    if (!(p.noise >= 0.0f))                        throw std::invalid_argument("generate_ensemble_codes: noise must be >= 0");

    const int S      = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(p.D))));
    const int stride = S * S;

    ensemble_codes out;
    out.side  = S;
    out.count = p.N;
    out.data.assign(static_cast<std::size_t>(p.N) * stride, 0.0f);

    std::mt19937 rng(p.seed);

    // --- M prototypes: each is k distinct winner indices in [0, D) ---
    std::vector<std::vector<int>> proto(static_cast<std::size_t>(p.M));
    {
        std::vector<int> all(p.D);
        std::iota(all.begin(), all.end(), 0);
        for (int m = 0; m < p.M; ++m) {
            for (int i = 0; i < p.k; ++i) {     // partial Fisher-Yates, first k
                std::uniform_int_distribution<int> d(i, p.D - 1);
                std::swap(all[i], all[d(rng)]);
            }
            proto[static_cast<std::size_t>(m)].assign(all.begin(), all.begin() + p.k);
        }
    }

    std::normal_distribution<float>      nd(0.0f, p.noise);
    std::uniform_real_distribution<float> ud_amp(0.0f, std::max(p.amp_var, 1e-7f));
    const float bg_cap = 3.0f * p.noise;

    std::vector<char> is_winner(static_cast<std::size_t>(p.D), 0);
    std::vector<int>  nonwinner;                 // [0,D) \ prototype, reused per code
    nonwinner.reserve(static_cast<std::size_t>(p.D));

    for (int c = 0; c < p.N; ++c) {
        const auto& pm = proto[static_cast<std::size_t>(c % p.M)];

        // Winners = prototype with `n_swap` of them re-rolled to fresh non-winner
        // indices, giving per-code variation around the prototype.
        const int max_swap = p.D - p.k;          // indices available outside the prototype
        int n_swap = static_cast<int>(std::lround(static_cast<double>(p.swap_frac) * p.k));
        n_swap = std::clamp(n_swap, 0, std::min(p.k, max_swap));

        std::fill(is_winner.begin(), is_winner.end(), char{0});
        for (int w : pm) is_winner[static_cast<std::size_t>(w)] = 1;   // mark prototype

        if (n_swap > 0) {
            // Drop the last n_swap prototype winners ...
            for (int s = 0; s < n_swap; ++s)
                is_winner[static_cast<std::size_t>(pm[static_cast<std::size_t>(p.k - 1 - s)])] = 0;
            // ... and add n_swap distinct indices from outside the prototype.
            nonwinner.clear();
            for (int i = 0; i < p.D; ++i)
                if (!is_winner[static_cast<std::size_t>(i)]) nonwinner.push_back(i);
            for (int s = 0; s < n_swap; ++s) {
                std::uniform_int_distribution<int> d(s, static_cast<int>(nonwinner.size()) - 1);
                std::swap(nonwinner[static_cast<std::size_t>(s)],
                          nonwinner[static_cast<std::size_t>(d(rng))]);
                is_winner[static_cast<std::size_t>(nonwinner[static_cast<std::size_t>(s)])] = 1;
            }
        }

        const std::size_t base = static_cast<std::size_t>(c) * stride;
        for (int i = 0; i < p.D; ++i) {
            if (p.recognition > 0.0f && is_winner[static_cast<std::size_t>(i)]) {
                const float u = (p.amp_var > 0.0f) ? ud_amp(rng) : 0.0f;
                out.data[base + i] = std::clamp(p.recognition * (1.0f - u), 0.0f, 1.0f);
                continue;
            }
            float bg = 0.0f;
            if (p.noise > 0.0f) bg = std::min(std::fabs(nd(rng)), bg_cap);
            out.data[base + i] = (p.recognition > 0.0f) ? (1.0f - p.inhibition) * bg : bg;
        }
        // indices [D, stride) stay zero (padding)
    }

    return out;
}

} // namespace coast

#endif // COAST_ENSEMBLE_CODE_H
