// GI1 isolation probe for the ngm2.5 GPU neuron group (design/ngm2_5_gpu.md).
//
// Self-contained (no dataset files): a synthetic CPU source feeds a proximal
// channel into ngm25_gpu; a cpu_sink downstream forces the ngm25 output io_buffer
// to gpu_to_cpu so the host can read it. Drives env::process()/swap_io() through
// the real facade and checks the GI1 gate:
//   [sizes]   the device arena matches a CPU ngm25 group built from the same cfg
//   [forward] on a real adapter: outputs finite, in [0,1], non-degenerate
//   [ckpt]    save_state -> load_state round-trip does not corrupt the run
// On the null backend (adapter "null") compute passes are dropped, so only the
// plumbing (register -> build -> step -> swap -> checkpoint) is exercised and the
// value checks are skipped.
//
// Run UNSANDBOXED for the webgpu backend (needs /dev/dri); the null backend build
// runs anywhere.

#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "logging/mem_state_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <span>
#include <vector>

namespace {

constexpr sim::output_id_t kIdSrc  = 0;
constexpr sim::output_id_t kIdNgm  = 1;
constexpr sim::output_id_t kIdSink = 2;
constexpr std::size_t      kWidth  = 784;
constexpr std::size_t      kNeurons = 64;

// Synthetic CPU input source: a fixed structured pattern each step.
struct synth_source final : sim::io_entity {
    sim::io_buffer* out_ = nullptr;
    std::vector<float> pattern_;
    explicit synth_source(std::size_t w) : pattern_(w) {
        for (std::size_t i = 0; i < w; ++i)
            pattern_[i] = ((i * 37u) % 11u == 0u) ? 0.85f : 0.03f;
    }
    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override {
        auto o = out_->cur_write_buffer();
        std::copy(pattern_.begin(), pattern_.end(), o.begin());
        return {};
    }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdSrc; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return pattern_.size(); }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// Trivial CPU consumer of the ngm25 output — forces io_buffer_gpu_to_cpu on buf 1.
struct cpu_sink final : sim::io_entity {
    std::array<std::size_t,1> inp_ids_{ kIdNgm };
    void set_output_buffer(sim::io_buffer&) override {}
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdSink; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return inp_ids_; }
};

ngm25::neuron_group_params_t base_params() {
    ngm25::neuron_group_params_t gp;
    gp.neuron_cnt   = kNeurons;
    gp.apical_cnt   = 0;
    gp.proximal_cnt = 4;
    gp.max_branch_level = 3;
    gp.inputs = { { "in", ngm25::bank_t::proximal } };
    gp.random_seed = seed_t{ 0 };
    return gp;
}
ngm25::cfg_store make_cfg() {
    ngm25::cfg_store cs;
    cs.neuron_groups["grp"] = base_params();
    return cs;
}
// Config tuned to branch quickly (GI4): short interval + low thresholds so the
// per-neuron gate fires within a few hundred steps.
ngm25::cfg_store make_growth_cfg() {
    ngm25::cfg_store cs;
    auto gp = base_params();
    gp.branch_interval         = 25;
    gp.accumulated_theta_thres = 0.05f;
    gp.min_mismatch_deviation  = 0.25f;
    gp.min_mismatch_percentage = 0.0005f;
    cs.neuron_groups["grp"] = gp;
    return cs;
}

// Build env: synth_source -> group T -> sink. Returns the group ref via callback.
template <class T, class F>
void with_env(const ngm25::cfg_store& cs, bool gpu, int steps, F&& fn) {
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
    fn((*env.template get_entities<T>().value())[0]);
}

// GI5 checkpoint->resume bit-exactness: run K steps, save; an uninterrupted run
// continues N more -> out_ref. A fresh env loads the step-K checkpoint and runs the
// same N -> out_res. With hash RNG (no stream state) + persisted step/branch_epoch/
// arena/carried-state and a deterministic device, out_res must equal out_ref bit
// for bit. Both envs share the same fixed synth_source, so the input sequence
// matches. Returns {equal, max_abs_diff}.
std::pair<bool,float> check_ckpt_resume(const ngm25::cfg_store& cs, int K, int N) {
    logging::mem_group tree;
    std::vector<float> out_ref;
    {
        sim::env env;
        env.emplace_back<synth_source>(kWidth);
        env.emplace_back<ngm25::neuron_group_gpu_t>(cs, "grp", kIdNgm);
        env.emplace_back<cpu_sink>();
        env.enable_gpu();
        { auto ng = env.get_entities<ngm25::neuron_group_gpu_t>();
          (*ng.value())[0].init_io({ { "in", { kIdSrc, kWidth } } }); }
        { auto s = env.get_entities<synth_source>(); (*s.value())[0].init_structure(); }
        env.init_io_buffers();
        for (int s = 0; s < K; ++s) {
            sim::throw_on_step_fault(env.process(), "A.process");
            sim::throw_on_step_fault(env.swap_io(), "A.swap");
        }
        { logging::mem_state_writer w; (*env.get_entities<ngm25::neuron_group_gpu_t>().value())[0].save_state(w); tree = w.take_root(); }
        for (int s = 0; s < N; ++s) {
            sim::throw_on_step_fault(env.process(), "A.process2");
            sim::throw_on_step_fault(env.swap_io(), "A.swap2");
        }
        auto o = (*env.get_io_buffer(kIdNgm))->cur_read_buffer();
        out_ref.assign(o.begin(), o.end());
    }
    std::vector<float> out_res;
    {
        sim::env env;
        env.emplace_back<synth_source>(kWidth);
        env.emplace_back<ngm25::neuron_group_gpu_t>(cs, "grp", kIdNgm);
        env.emplace_back<cpu_sink>();
        env.enable_gpu();
        { auto ng = env.get_entities<ngm25::neuron_group_gpu_t>();
          (*ng.value())[0].init_io({ { "in", { kIdSrc, kWidth } } }); }
        { auto s = env.get_entities<synth_source>(); (*s.value())[0].init_structure(); }
        env.init_io_buffers();
        // Warm the double-buffered io_buffers with the DC pattern (both pages) BEFORE
        // loading, so the resume env's INPUT sequence matches the uninterrupted run.
        // The 2 warmup steps' entity-state changes are then overwritten by load_state.
        for (int s = 0; s < 2; ++s) {
            sim::throw_on_step_fault(env.process(), "B.warm");
            sim::throw_on_step_fault(env.swap_io(), "B.warm.swap");
        }
        { logging::mem_state_reader r(tree, logging::k_format_version); (*env.get_entities<ngm25::neuron_group_gpu_t>().value())[0].load_state(r); }
        for (int s = 0; s < N; ++s) {
            sim::throw_on_step_fault(env.process(), "B.process");
            sim::throw_on_step_fault(env.swap_io(), "B.swap");
        }
        auto o = (*env.get_io_buffer(kIdNgm))->cur_read_buffer();
        out_res.assign(o.begin(), o.end());
    }
    bool eq = out_ref.size() == out_res.size();
    float mx = 0.0f;
    for (std::size_t i = 0; eq && i < out_ref.size(); ++i) {
        mx = std::max(mx, std::abs(out_ref[i] - out_res[i]));
        if (out_ref[i] != out_res[i]) eq = false;
    }
    return { eq, mx };
}

} // namespace

int main() {
    try {
        const ngm25::cfg_store cs = make_cfg();

        sim::env env;
        env.emplace_back<synth_source>(kWidth);
        env.emplace_back<ngm25::neuron_group_gpu_t>(cs, "grp", kIdNgm);
        env.emplace_back<cpu_sink>();
        env.enable_gpu();

        {
            auto ng = env.get_entities<ngm25::neuron_group_gpu_t>();
            (*ng.value())[0].init_io({ { "in", { kIdSrc, kWidth } } });
        }
        {
            auto s = env.get_entities<synth_source>();
            (*s.value())[0].init_structure();
        }
        env.init_io_buffers();
        const std::string adapter = env.gpu_adapter_name();
        const bool is_null = (adapter == "null");
        std::printf("adapter: %s%s\n\n", adapter.c_str(), is_null ? " (compute dropped)" : "");

        bool ok = true;

        // [sizes] device arena matches a CPU ngm25 group built from the same cfg.
        {
            ngm25::neuron_group_t cpu(cs, "grp", kIdNgm);
            cpu.init_io({ { "in", { kIdSrc, kWidth } } });
            auto ng = env.get_entities<ngm25::neuron_group_gpu_t>();
            const std::size_t gpu_syn = (*ng.value())[0].get_synapse_count();
            const bool sizes_ok = (gpu_syn == cpu.get_synapse_count()) &&
                                  ((*ng.value())[0].get_neuron_count() == cpu.get_neuron_count());
            std::printf("  [sizes]    %s: gpu synapses=%zu == cpu synapses=%zu\n",
                        sizes_ok ? "PASS" : "FAIL", gpu_syn, cpu.get_synapse_count());
            ok = ok && sizes_ok;
        }

        // Drive steps.
        constexpr int kSteps = 200;
        for (int s = 0; s < kSteps; ++s) {
            sim::throw_on_step_fault(env.process(),  "env::process");
            sim::throw_on_step_fault(env.swap_io(),  "env::swap_io");
        }

        auto out = (*env.get_io_buffer(kIdNgm))->cur_read_buffer();

        // [forward] real-adapter value checks (skipped on null: no compute ran).
        if (!is_null) {
            bool range_ok = (out.size() == kNeurons);
            float omax = 0.0f, omean = 0.0f;
            for (float x : out) {
                if (!std::isfinite(x) || x < -1e-6f || x > 1.0f + 1e-6f) range_ok = false;
                omax = std::max(omax, x); omean += x;
            }
            omean = out.empty() ? 0.0f : omean / static_cast<float>(out.size());
            const bool nondeg = omax > 1e-6f;
            std::printf("  [forward]  %s: %zu outputs finite/in[0,1], mean=%.5f max=%.5f\n",
                        (range_ok && nondeg) ? "PASS" : "FAIL", out.size(), omean, omax);
            ok = ok && range_ok && nondeg;
        } else {
            std::printf("  [forward]  SKIP: null backend (no compute)\n");
        }

        // [ckpt] save_state -> load_state round-trip into the same entity; then a
        // few more steps must still produce a valid output (no corruption).
        {
            auto ng = env.get_entities<ngm25::neuron_group_gpu_t>();
            auto& g = (*ng.value())[0];
            logging::mem_state_writer wtr;
            g.save_state(wtr);
            const logging::mem_group tree = wtr.take_root();
            { logging::mem_state_reader rdr(tree, logging::k_format_version); g.load_state(rdr); }

            for (int s = 0; s < 20; ++s) {
                sim::throw_on_step_fault(env.process(), "env::process (post-ckpt)");
                sim::throw_on_step_fault(env.swap_io(), "env::swap_io (post-ckpt)");
            }
            auto out2 = (*env.get_io_buffer(kIdNgm))->cur_read_buffer();
            bool ckpt_ok = (out2.size() == kNeurons);
            if (!is_null)
                for (float x : out2)
                    if (!std::isfinite(x) || x < -1e-6f || x > 1.0f + 1e-6f) ckpt_ok = false;
            std::printf("  [ckpt]     %s: save/load round-trip, %d post-ckpt steps valid\n",
                        ckpt_ok ? "PASS" : "FAIL", 20);
            ok = ok && ckpt_ok;
        }

        // [grow] GI4: branching + host repack + re-upload/capacity-rebuild. Only
        // meaningful on a real adapter (null drops the adapt compute -> no learning
        // -> no mismatch -> no branch).
        if (!is_null) {
            const std::size_t init_syn = kNeurons * 4 * kWidth;   // depth-0 init
            const ngm25::cfg_store gc = make_growth_cfg();
            std::size_t gpu_syn = 0; std::uint32_t gpu_epoch = 0; bool consistent = false;
            std::string err;
            with_env<ngm25::neuron_group_gpu_t>(gc, /*gpu=*/true, 400, [&](auto& g) {
                gpu_syn = g.get_synapse_count();
                gpu_epoch = g.get_branch_epoch();
                consistent = g.check_consistency(&err);
            });
            std::size_t cpu_syn = 0;
            with_env<ngm25::neuron_group_t>(gc, /*gpu=*/false, 400, [&](auto& g) {
                cpu_syn = g.get_synapse_count();
            });
            const bool grew = gpu_syn > init_syn && gpu_epoch > 0;
            const double ratio = cpu_syn > init_syn
                ? double(gpu_syn - init_syn) / double(cpu_syn - init_syn) : 0.0;
            // Growth is loose CPU-vs-GPU (branch decisions ride adapt_hist/mismatch
            // that diverge on penalty-firing input) — gate on GPU growing +
            // consistent + in the same ballpark as CPU.
            const bool grow_ok = grew && consistent && ratio > 0.3 && ratio < 3.0;
            const std::string cstat = consistent ? "yes" : ("NO:" + err);
            std::printf("  [grow]     %s: gpu %zu syn (epoch %u, consistent=%s) vs cpu %zu "
                        "(init %zu, growth ratio %.2f)\n",
                        grow_ok ? "PASS" : "FAIL", gpu_syn, gpu_epoch,
                        cstat.c_str(), cpu_syn, init_syn, ratio);
            ok = ok && grow_ok;
        }

        // [vis] refresh_vis_mirror() (GPU->CPU readback) + the flat/tree accessor
        // surface: per-rep spans are input_size-wide, finite, in range, seg ids in
        // [1,max_seg_id], the absent apical bank is empty, and the proximal bank is
        // non-degenerate. (Perm correctness itself is proven in GI2/GI3; the mirror
        // just reshapes the same arena via the CPU-identical materialize_bank.)
        if (!is_null) {
            with_env<ngm25::neuron_group_gpu_t>(make_cfg(), /*gpu=*/true, 100, [&](auto& g) {
                g.refresh_vis_mirror();
                const std::size_t in = kWidth;
                const std::uint8_t max_seg = g.get_max_segment_id();
                bool vis_ok = true; float pmax = 0.0f;
                const std::array<std::array<std::size_t,3>,3> samples{{ {{0,0,0}}, {{7,2,3}}, {{63,3,7}} }};
                for (const auto& s : samples) {
                    auto rep = g.get_proximal_representation(s[0], s[1], s[2]);
                    auto sid = g.get_proximal_segment_ids(s[0], s[1], s[2]);
                    if (rep.size() != in || sid.size() != in) { vis_ok = false; break; }
                    for (std::size_t i = 0; i < in; ++i) {
                        pmax = std::max(pmax, rep[i]);
                        if (!std::isfinite(rep[i]) || rep[i] < -1e-6f || rep[i] > 1.0f + 1e-6f) vis_ok = false;
                        if (sid[i] < 1 || sid[i] > max_seg) vis_ok = false;
                    }
                }
                const bool apical_empty = g.get_apical_representation(0,0,0).empty() &&
                                          g.get_apical_segment_ids(0,0,0).empty();
                vis_ok = vis_ok && apical_empty && pmax > 1e-6f;
                std::printf("  [vis]      %s: refresh_vis_mirror + rep/seg accessors "
                            "(proximal max perm=%.3f, apical bank empty=%s)\n",
                            vis_ok ? "PASS" : "FAIL", pmax, apical_empty ? "yes" : "no");
                ok = ok && vis_ok;
            });
        }

        // [resume] GI5: checkpoint->resume bit-exact vs an uninterrupted run.
        if (!is_null) {
            const auto [eq, mx] = check_ckpt_resume(make_cfg(), /*K=*/40, /*N=*/30);
            std::printf("  [resume]   %s: checkpoint@40 -> resume 30 steps bit-exact "
                        "vs uninterrupted (max|d|=%.3g)\n", eq ? "PASS" : "FAIL", mx);
            ok = ok && eq;
        }

        // [resume-grown] H1 regression guard: a checkpoint taken AFTER growth,
        // resumed into a FRESH group. capacity_syn_ has no headroom, so the first
        // branch event (interval 25) crosses capacity -> the K=200 checkpoint's
        // arena exceeds a fresh group's initial capacity. load_state must grow
        // capacity + rebuild the plan before upload (mirroring grow_if_branched)
        // instead of throwing "arena exceeds device capacity". Must be bit-exact.
        if (!is_null) {
            const auto [eq, mx] = check_ckpt_resume(make_growth_cfg(), /*K=*/200, /*N=*/30);
            std::printf("  [resume-grown] %s: ckpt@200 (post-growth) -> fresh resume "
                        "30 steps bit-exact (max|d|=%.3g)\n", eq ? "PASS" : "FAIL", mx);
            ok = ok && eq;
        }

        std::printf("\n[coast-ngm25-gpu-probe] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-ngm25-gpu-probe] FAILED: %s\n", e.what());
        return 1;
    }
}
