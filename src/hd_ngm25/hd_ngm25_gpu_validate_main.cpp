// GI2 validation for the ngm2.5 GPU response+combine kernels (design/ngm2_5_gpu.md
// §7/§8). Three checks:
//   [rng]   host ngm25::pcg_hash(syn_key(..)) == a WGSL kernel computing the same,
//           over a grid of (seed,step,dend,seg,input_idx) — the §7 parity guard.
//   [parity] CPU ngm25 vs GPU ngm25 forward output agree CLOSELY on a deterministic
//           config (noise 0), i.e. no gross transcription error. NOT bit-equality.
//   [determ] the GPU forward path is run-to-run bit-identical (fixed-order
//           reductions, no atomics).
//
// Run UNSANDBOXED (needs /dev/dri). The null backend cannot run [rng]/[parity]
// (compute dropped); this exe targets the webgpu backend only.

#include "core/gpu/gpu_device.h"
#include "core/gpu/shared_prng.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm25/hd_ngm25_prng.h"
#include "hd_ngm25/hd_ngm25_cfg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace g = coast::gpu;

// ---------------------------------------------------------------- [rng] ----
bool check_rng_parity() {
    const char* wgsl = R"WGSL(
fn syn_key(seed: u32, step: u32, dend: u32, seg: u32, input_idx: u32) -> u32 {
    return seed ^ (step*3266489917u) ^ (dend*2246822519u) ^ (seg*668265263u) ^ (input_idx*374761393u);
}
@group(0) @binding(0) var<storage, read>       ks: array<vec4<u32>>;   // step,dend,seg,iidx
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@group(0) @binding(2) var<uniform>             seed: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&outp)) { return; }
    let k = ks[i];
    outp[i] = pcg_hash(syn_key(seed.x, k.x, k.y, k.z, k.w));
}
)WGSL";

    constexpr std::uint32_t kSeed = 0xC0FFEEu;
    std::vector<std::array<std::uint32_t,4>> keys;   // step,dend,seg,iidx
    for (std::uint32_t step = 0; step < 8; ++step)
        for (std::uint32_t dend = 0; dend < 16; ++dend)
            for (std::uint32_t seg = 0; seg < 16; ++seg)
                for (std::uint32_t ii = 0; ii < 8; ++ii)
                    keys.push_back({step, dend, seg, ii});
    const std::size_t n = keys.size();

    auto dev = g::make_gpu_device();
    const g::buffer_id b_keys = dev->register_buffer(g::topology::gpu_resident, n * 16, false);
    const g::buffer_id b_out  = dev->register_buffer(g::topology::gpu_resident, n * 4, false);
    const g::buffer_id b_seed = dev->register_buffer(g::topology::uniform, 16, false);
    dev->upload(b_keys, std::as_bytes(std::span<const std::array<std::uint32_t,4>>(keys)));
    const std::array<std::uint32_t,4> seedv{ kSeed, 0, 0, 0 };
    dev->upload(b_seed, std::as_bytes(std::span<const std::array<std::uint32_t,4>>(&seedv, 1)));

    const std::string src = std::string(g::shared_prng_wgsl()) + wgsl;
    const g::kernel_id k = dev->register_kernel(src, "main");
    g::step_plan sp;
    g::plan_op op{}; op.kind = g::op_kind::compute; op.kernel = k;
    g::plan_dispatch d{};
    d.bindings = { {b_keys, g::access::read, 0}, {b_out, g::access::write, 1}, {b_seed, g::access::read, 2} };
    d.dispatch = { static_cast<std::uint32_t>((n + 63) / 64), 1, 1 };
    op.dispatches = { std::move(d) };
    sp.ops = { std::move(op) };
    const g::plan_id pid = dev->register_plan();
    dev->build_plan(pid, sp);
    dev->begin_step(); dev->enqueue(pid); dev->submit_step(); dev->rendezvous_step();

    std::vector<std::uint32_t> gpu(n, 0);
    (void)dev->readback(b_out, std::as_writable_bytes(std::span<std::uint32_t>(gpu)));

    std::size_t mismatch = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& kk = keys[i];
        const std::uint32_t host = ngm25::pcg_hash(ngm25::syn_key(kSeed, kk[0], kk[1], kk[2], kk[3]));
        if (host != gpu[i]) ++mismatch;
    }
    std::printf("  [rng]    %s: %zu keys, %zu mismatches (host pcg_hash(syn_key) == WGSL)\n",
                mismatch == 0 ? "PASS" : "FAIL", n, mismatch);
    return mismatch == 0;
}

// -------------------------------------------------------- env plumbing -----
constexpr sim::output_id_t kIdSrc  = 0;
constexpr sim::output_id_t kIdNgm  = 1;
constexpr std::size_t      kWidth  = 400;
constexpr std::size_t      kNeurons = 32;

struct synth_source final : sim::io_entity {
    sim::io_buffer* out_ = nullptr;
    std::vector<float> pat_;
    // UNIFORM input (impl-plan §9 fixture): with every position equal, half_max =
    // max/2 < inp, so the D1 penalty never fires and CPU/GPU differ ONLY by seg
    // reduction order (~1e-6). This isolates gross transcription errors. The
    // penalty and its per-synapse floor are covered step by step, against the CPU
    // implementations, by integration.impl_equivalence_gpu.
    explicit synth_source(std::size_t w) : pat_(w, 0.5f) {}
    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override {
        auto o = out_->cur_write_buffer(); std::copy(pat_.begin(), pat_.end(), o.begin()); return {};
    }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return kIdSrc; }
    [[nodiscard]] std::size_t get_outp_size() const override { return pat_.size(); }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};
struct cpu_sink final : sim::io_entity {
    std::array<std::size_t,1> in_{ kIdNgm };
    void set_output_buffer(sim::io_buffer&) override {}
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return 2; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return in_; }
};

ngm25::cfg_store det_cfg() {
    ngm25::cfg_store cs;
    ngm25::neuron_group_params_t gp;
    gp.neuron_cnt = kNeurons; gp.apical_cnt = 0; gp.proximal_cnt = 4;
    gp.max_branch_level = 3;
    gp.neuron_noise_range = { 0.0f, 0.0f };   // deterministic fixture: no D2 noise
    gp.inputs = { { "in", ngm25::bank_t::proximal } };
    gp.random_seed = seed_t{ 0 };
    cs.neuron_groups["grp"] = gp;
    cs.frozen = true;                         // forward-only: isolate response+combine
    return cs;
}

// Run a group of type T for `steps` and return its final output.
template <class T>
std::vector<float> run_group(const ngm25::cfg_store& cs, bool gpu, int steps) {
    sim::env env;
    env.emplace_back<synth_source>(kWidth);
    env.emplace_back<T>(cs, "grp", kIdNgm);
    env.emplace_back<cpu_sink>();
    if (gpu) env.enable_gpu();
    { auto ng = env.template get_entities<T>(); (*ng.value())[0].init_io({ { "in", { kIdSrc, kWidth } } }); }
    { auto s = env.get_entities<synth_source>(); (*s.value())[0].init_structure(); }
    env.init_io_buffers();
    for (int s = 0; s < steps; ++s) {
        sim::throw_on_step_fault(env.process(), "process");
        sim::throw_on_step_fault(env.swap_io(), "swap_io");
    }
    auto out = (*env.get_io_buffer(kIdNgm))->cur_read_buffer();
    return { out.begin(), out.end() };
}

// The learned per-synapse arrays, in arena slot order (identical on CPU and GPU).
struct learned { std::vector<float> perm, adapt_hist, mismatch; };

// Run a group with LEARNING ON for `steps` and snapshot its learned arrays.
template <class T>
learned run_learn(const ngm25::cfg_store& cs, bool gpu, int steps) {
    sim::env env;
    env.emplace_back<synth_source>(kWidth);
    env.emplace_back<T>(cs, "grp", kIdNgm);
    env.emplace_back<cpu_sink>();
    if (gpu) env.enable_gpu();
    { auto ng = env.template get_entities<T>(); (*ng.value())[0].init_io({ { "in", { kIdSrc, kWidth } } }); }
    { auto s = env.get_entities<synth_source>(); (*s.value())[0].init_structure(); }
    env.init_io_buffers();
    for (int s = 0; s < steps; ++s) {
        sim::throw_on_step_fault(env.process(), "process");
        sim::throw_on_step_fault(env.swap_io(), "swap_io");
    }
    auto& g = (*env.template get_entities<T>().value())[0];
    if constexpr (std::is_same_v<T, ngm25::neuron_group_gpu_t>)
        return { g.readback_perm(), g.readback_adapt_hist(), g.readback_mismatch() };
    else
        return { g.perm_snapshot(), g.adapt_hist_snapshot(), g.mismatch_snapshot() };
}

struct dist { float mean_abs_d, max_abs_d, mean_cpu, mean_gpu; };
dist compare(const std::vector<float>& a, const std::vector<float>& b) {
    dist r{ 1.0f, 1.0f, 0.0f, 0.0f };
    if (a.size() != b.size() || a.empty()) return r;
    float sd = 0.0f, mx = 0.0f, sa = 0.0f, sb = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // A non-finite value on either side is an infinite difference:
        // std::max would silently drop a NaN.
        const float d = (std::isfinite(a[i]) && std::isfinite(b[i])) ? std::abs(a[i] - b[i])
                                                                    : INFINITY;
        sd += d; mx = std::max(mx, d);
        sa += a[i]; sb += b[i];
    }
    const float n = static_cast<float>(a.size());
    return { sd / n, mx, sa / n, sb / n };
}

} // namespace

int main() {
    try {
        const ngm25::cfg_store cs = det_cfg();
        bool ok = true;

        ok &= check_rng_parity();

        // [parity] CPU vs GPU forward output, deterministic config.
        const auto cpu = run_group<ngm25::neuron_group_t>(cs, /*gpu=*/false, 5);
        const auto gpu = run_group<ngm25::neuron_group_gpu_t>(cs, /*gpu=*/true, 5);
        float max_abs = 0.0f, sum_abs = 0.0f;
        const bool same_size = cpu.size() == gpu.size() && cpu.size() == kNeurons;
        if (same_size)
            for (std::size_t i = 0; i < cpu.size(); ++i) {
                const float dd = std::abs(cpu[i] - gpu[i]);
                max_abs = std::max(max_abs, dd); sum_abs += dd;
            }
        const float mean_abs = same_size ? sum_abs / cpu.size() : 1.0f;
        // Close, not bit-exact: parallel tree reduction vs CPU serial order only
        // (penalty inert on uniform input). 1e-4 catches gross transcription errors
        // while tolerating reduction-order FP over a 400-wide summed response.
        const bool parity_ok = same_size && max_abs < 1e-4f;
        std::printf("  [parity] %s: CPU vs GPU forward, mean|d|=%.2e max|d|=%.2e (tol 1e-4)\n",
                    parity_ok ? "PASS" : "FAIL", mean_abs, max_abs);
        ok = ok && parity_ok;

        // [determ] GPU run-to-run bit-identical.
        const auto gpu2 = run_group<ngm25::neuron_group_gpu_t>(cs, /*gpu=*/true, 5);
        bool determ = gpu.size() == gpu2.size();
        for (std::size_t i = 0; determ && i < gpu.size(); ++i)
            if (gpu[i] != gpu2[i]) determ = false;
        std::printf("  [determ] %s: GPU run-to-run bit-identical over %zu outputs\n",
                    determ ? "PASS" : "FAIL", gpu.size());
        ok = ok && determ;

        // [adapt] CPU vs GPU learned arrays (R5/R6/R7): permanence, adapt history
        // AND mismatch, compared PER ELEMENT. Mismatch matters most: its update does
        // not scale with the learning weight, so an extra or missing adaptation
        // (e.g. the winner adapted twice, M.4.14) shows there first, while the
        // permanence change can be below float resolution. The tolerance sits
        // well above the measured CPU/GPU agreement (reduction order only) and
        // far below any real model difference.
        ngm25::cfg_store lc = det_cfg();
        lc.frozen = false;
        constexpr float kAdaptTol = 1e-5f;
        for (const int steps : { 1, 100 }) {
            const learned c = run_learn<ngm25::neuron_group_t>(lc, false, steps);
            const learned g = run_learn<ngm25::neuron_group_gpu_t>(lc, true, steps);
            const dist p = compare(c.perm, g.perm);
            const dist h = compare(c.adapt_hist, g.adapt_hist);
            const dist m = compare(c.mismatch, g.mismatch);
            const bool step_ok = c.perm.size() == g.perm.size() && p.max_abs_d <= kAdaptTol &&
                                 h.max_abs_d <= kAdaptTol && m.max_abs_d <= kAdaptTol;
            std::printf("  [adapt@%d] %s: max|d| perm=%.2e adapt_hist=%.2e mismatch=%.2e "
                        "(%zu syn, per-element tol %.0e)\n",
                        steps, step_ok ? "PASS" : "FAIL", p.max_abs_d, h.max_abs_d, m.max_abs_d,
                        c.perm.size(), kAdaptTol);
            ok = ok && step_ok;
        }

        std::printf("\n[coast-ngm25-gpu-validate] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-ngm25-gpu-validate] FAILED: %s\n", e.what());
        return 1;
    }
}
