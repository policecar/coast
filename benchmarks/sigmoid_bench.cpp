// sigmoid_bench.cpp
// Benchmark comparing various sigmoid implementations for hd_ngm2_tools.h
//
// Compile: g++ -std=c++23 -O3 -march=native -o sigmoid_bench sigmoid_bench.cpp
// Run:     ./sigmoid_bench
//
// Tests:
//   v0  Original (shape members are double → implicit double-precision path)
//   v1  Float-clean (shape members are float, eliminating double promotions)
//   v2  Precomputed params + expf  (pow() moved out of hot path)
//   v3  Precomputed params + fast polynomial exp  (~1e-6 relative error)
//   v4  Precomputed params + Schraudolph bit-hack exp (~5% relative error, maximally fast)
//
// Each variant tested with:
//   A. Fixed shape  (compiler has best chance to constant-fold pow())
//   B. 4 rotating shapes  (forces recomputation of shape-derived constants for v0/v1)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <chrono>
#include <algorithm>

// ─── v0: Original ─────────────────────────────────────────────────────────────
// Shape members are double → std::pow(double, float) → result is double →
// the entire inv_step_size / starting_point computation runs in double precision.

struct sigmoid_shape_d {
    double steepness        = 0.5;
    double transition_point = 0.5;
};

float sigmoid_v0(float x, sigmoid_shape_d shape) {
    const float inv_step_size  = 1.0f / (1.0f - std::pow(shape.steepness, 0.1f));
    const float starting_point = static_cast<float>(-shape.transition_point) * inv_step_size;
    return 1.0f / (1.0f + std::exp(-(x * inv_step_size + starting_point)));
}

// ─── v1: Float-clean ──────────────────────────────────────────────────────────
// Change shape members to float → powf(float,float) → all single-precision.
// Same algorithm, no double promotions.

struct sigmoid_shape_f {
    float steepness        = 0.5f;
    float transition_point = 0.5f;
};

float sigmoid_v1(float x, sigmoid_shape_f shape) {
    const float inv_step_size  = 1.0f / (1.0f - std::powf(shape.steepness, 0.1f));
    const float starting_point = -shape.transition_point * inv_step_size;
    return 1.0f / (1.0f + std::expf(-(x * inv_step_size + starting_point)));
}

// ─── v2: Precomputed params + expf ────────────────────────────────────────────
// The pow() and two divisions for the shape-derived constants are amortized:
// compute once when the shape is set, not on every sigmoid evaluation.

struct sigmoid_params_t {
    float inv_step_size;
    float starting_point;

    sigmoid_params_t(float steepness = 0.5f, float tp = 0.5f) {
        inv_step_size  = 1.0f / (1.0f - std::powf(steepness, 0.1f));
        starting_point = -tp * inv_step_size;
    }
};

float sigmoid_v2(float x, const sigmoid_params_t& p) {
    return 1.0f / (1.0f + std::expf(-(x * p.inv_step_size + p.starting_point)));
}

// ─── v3: Precomputed params + fast polynomial exp ─────────────────────────────
// Replaces libm expf with a range-reduced 5-term minimax polynomial.
// Relative error < 2^-20 (~1e-6) for x in [-87.3, 88.7].
// The bit-twiddling to scale by 2^k costs 1 int add.

static float fast_exp_poly(float x) {
    constexpr float log2e = 1.44269504088896f;   // 1/ln(2)
    constexpr float ln2   = 0.69314718055995f;

    // k = round(x / ln2),  r = x - k*ln2   →  r in [-0.5*ln2, 0.5*ln2]
    float z = x * log2e;
    int k = static_cast<int>(z + (z >= 0.0f ? 0.5f : -0.5f));
    float r = x - static_cast<float>(k) * ln2;

    // Horner form of 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f
              + r * (0.041666668f + r * 0.008333334f))));

    // Multiply by 2^k using the float exponent field
    int32_t pi;
    std::memcpy(&pi, &p, sizeof pi);
    pi += k << 23;
    std::memcpy(&p, &pi, sizeof pi);
    return p;
}

float sigmoid_v3(float x, const sigmoid_params_t& p) {
    return 1.0f / (1.0f + fast_exp_poly(-(x * p.inv_step_size + p.starting_point)));
}

// ─── v4: Precomputed params + Schraudolph bit-hack exp ────────────────────────
// Legendary 1-line approximation (Schraudolph 1999).
// Maps the float's bitfield linearly to approximate exp.
// Max relative error ~5%, average ~1.7%. Useful when accuracy is secondary.
// 3 machine instructions (mul, add, mov).

static float schraudolph_exp(float x) {
    // 12102203 ≈ 2^23 / ln(2);  constant 1064866805 minimises mean absolute error
    union { float f; int32_t i; } u;
    u.i = static_cast<int32_t>(12102203.0f * x) + 1064866805;
    return u.f;
}

float sigmoid_v4(float x, const sigmoid_params_t& p) {
    return 1.0f / (1.0f + schraudolph_exp(-(x * p.inv_step_size + p.starting_point)));
}

// ─── Benchmark harness ────────────────────────────────────────────────────────

static constexpr int N = 100'000'000;

volatile float sink = 0.0f;  // prevent dead-code elimination

template<typename Fn>
double bench(const char* name, Fn fn) {
    // Warm-up pass (L1/branch predictor)
    float acc = 0.0f;
    for (int i = 0; i < 1000; ++i) acc += fn(i);
    sink = acc;

    // Timed pass
    auto t0 = std::chrono::high_resolution_clock::now();
    acc = 0.0f;
    for (int i = 0; i < N; ++i) acc += fn(i);
    sink = acc;
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_per_call = ns / N;
    printf("  %-40s  %6.2f ns/call   (acc=%.4f)\n", name, ns_per_call, (double)acc);
    return ns_per_call;
}

// ─── Accuracy check ───────────────────────────────────────────────────────────

void accuracy_check() {
    printf("=== Accuracy vs v0 (reference) ===\n");
    sigmoid_shape_d shape_d {0.5, 0.5};
    sigmoid_shape_f shape_f {0.5f, 0.5f};
    sigmoid_params_t params {0.5f, 0.5f};

    const int M = 1000;
    double max_err_v1 = 0, max_err_v2 = 0, max_err_v3 = 0, max_err_v4 = 0;

    for (int i = 0; i <= M; ++i) {
        float x = -5.0f + 10.0f * i / M;
        float ref = sigmoid_v0(x, shape_d);
        max_err_v1 = std::max(max_err_v1, std::abs(sigmoid_v1(x, shape_f) - ref) / (std::abs(ref) + 1e-10));
        max_err_v2 = std::max(max_err_v2, std::abs(sigmoid_v2(x, params) - ref) / (std::abs(ref) + 1e-10));
        max_err_v3 = std::max(max_err_v3, std::abs(sigmoid_v3(x, params) - ref) / (std::abs(ref) + 1e-10));
        max_err_v4 = std::max(max_err_v4, std::abs(sigmoid_v4(x, params) - ref) / (std::abs(ref) + 1e-10));
    }

    printf("  v1 float-clean:          max relative error = %.2e\n", max_err_v1);
    printf("  v2 precomputed + expf:   max relative error = %.2e\n", max_err_v2);
    printf("  v3 precomputed + poly:   max relative error = %.2e\n", max_err_v3);
    printf("  v4 precomputed + schaud: max relative error = %.2e\n", max_err_v4);
    printf("\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("Sigmoid Benchmark  (N = %d per variant)\n\n", N);

    accuracy_check();

    // x values derived from loop index in [-1, 1] — avoids precomputing a big array
    const float x_scale = 2.0f / N;

    // ── A: Fixed default shape ─────────────────────────────────────────────────
    // Compiler may constant-fold pow(0.5, 0.1) for v0/v1 when inlined.
    printf("=== A. Fixed default shape {steepness=0.5, tp=0.5} ===\n");

    sigmoid_shape_d shape_d   {0.5, 0.5};
    sigmoid_shape_f shape_f   {0.5f, 0.5f};
    sigmoid_params_t params   {0.5f, 0.5f};

    bench("v0 original (double shape)",       [&](int i){ return sigmoid_v0(i*x_scale-1.0f, shape_d); });
    bench("v1 float-clean",                   [&](int i){ return sigmoid_v1(i*x_scale-1.0f, shape_f); });
    bench("v2 precomputed + expf",            [&](int i){ return sigmoid_v2(i*x_scale-1.0f, params); });
    bench("v3 precomputed + fast_exp_poly",   [&](int i){ return sigmoid_v3(i*x_scale-1.0f, params); });
    bench("v4 precomputed + schraudolph",     [&](int i){ return sigmoid_v4(i*x_scale-1.0f, params); });

    // ── B: 4 rotating shapes ──────────────────────────────────────────────────
    // Forces v0/v1 to recompute pow() on every call.
    printf("\n=== B. 4 rotating shapes (forces pow() recomputation for v0/v1) ===\n");

    std::array<sigmoid_shape_d,4> shapes_d {{
        {0.5,  0.5},  {0.3,  0.4},  {0.7,  0.6},  {0.2,  0.3}
    }};
    std::array<sigmoid_shape_f,4> shapes_f {{
        {0.5f, 0.5f}, {0.3f, 0.4f}, {0.7f, 0.6f}, {0.2f, 0.3f}
    }};
    std::array<sigmoid_params_t,4> params4 {{
        {0.5f, 0.5f}, {0.3f, 0.4f}, {0.7f, 0.6f}, {0.2f, 0.3f}
    }};

    bench("v0 original (double shape, 4-rotate)",     [&](int i){ return sigmoid_v0(i*x_scale-1.0f, shapes_d[i&3]); });
    bench("v1 float-clean (4-rotate)",                [&](int i){ return sigmoid_v1(i*x_scale-1.0f, shapes_f[i&3]); });
    bench("v2 precomputed + expf (4-rotate)",         [&](int i){ return sigmoid_v2(i*x_scale-1.0f, params4[i&3]); });
    bench("v3 precomputed + fast_exp_poly (4-rotate)",[&](int i){ return sigmoid_v3(i*x_scale-1.0f, params4[i&3]); });
    bench("v4 precomputed + schraudolph (4-rotate)",  [&](int i){ return sigmoid_v4(i*x_scale-1.0f, params4[i&3]); });

    return 0;
}
