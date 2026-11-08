// Isolation probe for the ngm3 GPU neuron group (S6 / develop.s6).
//
// Following the S1-S5 pattern: a standalone unsandboxed exe that exercises the
// new GPU entity in isolation from JSON / sim_builder. It hand-builds a real
// sim::env with a 3-entity chain:
//
//   mnist_io_gpu (id 0)  --[in_proximal]-->  ngm3_gpu (id 1)  -->  cpu_sink (id 2)
//
// The mnist->ngm3 seam is GPU->GPU (io_buffer_gpu); the ngm3->sink seam is
// GPU->CPU (io_buffer_gpu_to_cpu), so the post-inhibition group_outp the ngm3
// kernel writes is read back to the host and we can assert on it.
//
// S6 is SANITY-ONLY (no adapt -> frozen permanences, so no CPU-vs-GPU numerical
// parity here — that is pinned at S12). Checks after N steps:
//   1. FINITE + IN-RANGE — every group_outp value is finite and in [0,1].
//   2. NON-DEGENERATE   — not all-zero; at least one neuron responds.
//   3. AGGREGATE BANDS  — mean/max printed and bounded to plausible bands.
//
// Run UNSANDBOXED (sandbox blocks /dev/dri) from the repo root (loads ./data/mnist).

#include "core/gpu/gpu_device.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "io/mnist_io_gpu.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "io/mnist_cfg.h"
#include "probe/probe_instance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <span>

namespace {

// 5000 = the default branch_interval, so branch_main fires once on the final
// step (input_cnt reaches 5000 -> % branch_interval == 0). The extra 200 steps
// let adapt run on the freshly-branched segments and confirm the post-branch
// state stays valid. The S8 [branch] check asserts segment_ids evolve from the
// uniform {1} init; the S7 [learning] check still holds over the longer run.
constexpr int kSteps = 5200;
constexpr sim::output_id_t kIdMnist = 0;
constexpr sim::output_id_t kIdNgm3  = 1;
constexpr sim::output_id_t kIdSink  = 2;
constexpr std::size_t kImageSize = 784;

// Trivial CPU consumer of the ngm3 output — forces io_buffer_gpu_to_cpu on
// buffer 1 so the kernel output is CPU-visible.
struct cpu_sink final : sim::io_entity {
    sim::io_buffer* in_  = nullptr;
    sim::io_buffer* out_ = nullptr;
    std::array<std::size_t,1> inp_ids_{ kIdNgm3 };

    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer& b) override { in_ = &b; }
    sim::step_result process() override { /* the seam itself is the point */ return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdSink; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return inp_ids_; }
};

// Count distinct segment ids in a flat readback (seg_ids are u8-valued, <= 31).
std::size_t distinct_seg_ids(std::span<const std::uint32_t> v) {
    std::array<bool, 256> seen{};
    std::size_t cnt = 0;
    for (std::uint32_t s : v) { const std::uint32_t k = s & 0xffu; if (!seen[k]) { seen[k] = true; ++cnt; } }
    return cnt;
}

} // namespace

int main() {
    try {
        // --- mnist GPU dataset params ---
        mnist::mnist_param_t mp;
        mp.image_file      = "./data/mnist/train-images-idx3-ubyte";
        mp.label_file      = "./data/mnist/train-labels-idx1-ubyte";
        mp.change_interval = 1;
        mp.noise_range     = {0.0, 0.05};
        mp.random_seed     = seed_t{ 0 };

        // --- ngm3 GPU group config (one proximal input, apical absent) ---
        ngm3::cfg_store cs;
        ngm3::neuron_group_params_t gp;
        gp.neuron_cnt            = 64;
        gp.apical_cnt            = 0;
        gp.proximal_cnt          = 4;
        gp.max_branch_level = 3;
        gp.inputs                = { { "in_proximal", ngm3::bank_t::proximal } };
        gp.random_seed           = seed_t{ 0 };
        // The single branch pass (step 5000) splits only synapses whose
        // accumulated theta reached this threshold. Under the noise gate many
        // sit just below the default 2.0 at that step: 2.0 reassigned 112 seg
        // cells on RADV and 0 on Metal. Reassigned cells vs threshold on RADV:
        // 2.0 -> 112, 1.5 -> 8568, 1.0 -> 15784, 0.5 -> 23736. 1.0 is past
        // that cliff, so the [branch] check no longer hinges on FP rounding.
        gp.accumulated_theta_thres = 1.0f;
        cs.neuron_groups["grp"]  = gp;

        sim::env env;
        env.emplace_back<mnist::mnist_io_gpu>(mp, kIdMnist);
        env.emplace_back<ngm3::neuron_group_gpu_t>(cs, "grp", kIdNgm3);
        env.emplace_back<cpu_sink>();
        env.enable_gpu();

        // Resolve ngm3 routing + allocate/upload bank storage BEFORE init_io_buffers
        // (the topology pre-pass reads get_inp_ids(), which init_io populates).
        {
            auto ng = env.get_entities<ngm3::neuron_group_gpu_t>();
            (*ng.value())[0].init_io({ { "in_proximal", { kIdMnist, kImageSize } } });
        }

        // ngm3's structure/probe set is built inside init_io() above; mnist_io_gpu
        // has no such init hook here (sim_builder's init func does it in real runs),
        // so build it explicitly now that placement is final — required for the S9
        // [probes] check below to have a live fan-out target (root_entity_).
        {
            auto mn = env.get_entities<mnist::mnist_io_gpu>();
            (*mn.value())[0].init_structure();
        }

        env.init_io_buffers();   // buffer 0 -> io_buffer_gpu, buffer 1 -> gpu_to_cpu
        std::printf("adapter: %s\n\n", env.gpu_adapter_name().c_str());

        // Snapshot the frozen Poisson-init proximal permanences before any step,
        // so S7 can prove that adapt actually moves them (learning happens).
        std::vector<float>         perm_before;
        std::vector<std::uint32_t> seg_before;
        {
            auto ng = env.get_entities<ngm3::neuron_group_gpu_t>();
            perm_before = (*ng.value())[0].readback_permanences(ngm3::bank_t::proximal);
            seg_before  = (*ng.value())[0].readback_segment_ids(ngm3::bank_t::proximal);
        }

        // --- S9 probe fan-out setup (before stepping) ---------------------------
        // Activate one probe on the ngm3 GPU group: read_probes() (called by the
        // dispatcher after each phase-1 wait_idle) should then read the GPU-
        // computed value back and push it through the existing probe SPSC queue.
        // Also DISCOVER (without activating) a second ngm3 probe as a negative
        // control: even though "max activity" being active drives every PROBE()
        // statement down the slow path each step, entity_probe_set::record gates
        // on the per-instance active flag, so the inactive probe must stay empty.
        probe::probe_instance* p_ngm3_act  = nullptr;   // activated
        probe::probe_instance* p_ngm3_idle = nullptr;   // discovered, inactive
        {
            auto ng = env.get_entities<ngm3::neuron_group_gpu_t>();
            auto& g = (*ng.value())[0];
            p_ngm3_act  = g.probes().activate(&g, "max activity");
            p_ngm3_idle = g.probes().discover(&g, "winning thres");
        }

        for (int s = 0; s < kSteps; ++s) {
            sim::throw_on_step_fault(env.process(), "env::process");
            sim::throw_on_step_fault(env.swap_io(), "env::swap_io");
        }

        std::vector<float>         perm_after;
        std::vector<std::uint32_t> seg_after;
        {
            auto ng = env.get_entities<ngm3::neuron_group_gpu_t>();
            perm_after = (*ng.value())[0].readback_permanences(ngm3::bank_t::proximal);
            seg_after  = (*ng.value())[0].readback_segment_ids(ngm3::bank_t::proximal);
        }

        auto out = (*env.get_io_buffer(kIdNgm3))->cur_read_buffer();

        bool ok = true;

        // 1. finite + in [0,1]
        bool range_ok = (out.size() == gp.neuron_cnt);
        for (float x : out) {
            if (!std::isfinite(x) || x < -1e-6f || x > 1.0f + 1e-6f) { range_ok = false; break; }
        }
        std::printf("  [range]      %s: all %zu outputs finite and in [0,1]\n",
                    range_ok ? "PASS" : "FAIL", out.size());
        ok = ok && range_ok;

        // 2. non-degenerate (not all zero)
        const float omax = out.empty() ? 0.0f
                         : *std::max_element(out.begin(), out.end());
        const bool nondeg_ok = omax > 1e-6f;
        std::printf("  [non-degen]  %s: max output = %.6f (> 0 => neurons respond)\n",
                    nondeg_ok ? "PASS" : "FAIL", omax);
        ok = ok && nondeg_ok;

        // 3. aggregate bands (informational + loose sanity)
        const float omean = out.empty() ? 0.0f
                          : std::reduce(out.begin(), out.end(), 0.0f) /
                            static_cast<float>(out.size());
        const bool agg_ok = std::isfinite(omean) && omean >= 0.0f && omean <= 1.0f;
        std::printf("  [aggregate]  %s: mean=%.5f max=%.5f\n",
                    agg_ok ? "PASS" : "FAIL", omean, omax);
        ok = ok && agg_ok;

        // 4. learning happened — permanences drifted from the frozen init, and
        //    the drifted state is still valid (finite, in [0,1]).
        bool learn_ok = (!perm_before.empty()) &&
                        (perm_after.size() == perm_before.size());
        std::size_t changed = 0;
        float max_abs_drift = 0.0f, sum_abs_drift = 0.0f;
        bool after_valid = true;
        if (learn_ok) {
            for (std::size_t i = 0; i < perm_after.size(); ++i) {
                const float d = std::abs(perm_after[i] - perm_before[i]);
                if (d > 1e-7f) ++changed;
                max_abs_drift = std::max(max_abs_drift, d);
                sum_abs_drift += d;
                if (!std::isfinite(perm_after[i]) ||
                    perm_after[i] < -1e-6f || perm_after[i] > 1.0f + 1e-6f) {
                    after_valid = false;
                }
            }
            learn_ok = (changed > 0) && after_valid;
        }
        const float mean_abs_drift = perm_after.empty() ? 0.0f
            : sum_abs_drift / static_cast<float>(perm_after.size());
        std::printf("  [learning]   %s: %zu/%zu permanences moved "
                    "(mean|Δ|=%.6g max|Δ|=%.6g) after %d steps\n",
                    learn_ok ? "PASS" : "FAIL", changed, perm_after.size(),
                    mean_abs_drift, max_abs_drift, kSteps);
        ok = ok && learn_ok;

        // 5. branching happened — segment_ids evolved from the uniform {1} init
        //    (total_representation_count grew) and the post-branch state is valid
        //    (seg_ids never exceed max_seg_id = 2^(depth+1)-1).
        const std::size_t distinct_before = distinct_seg_ids(seg_before);
        const std::size_t distinct_after  = distinct_seg_ids(seg_after);
        const std::uint32_t max_seg_id = (1u << (gp.max_branch_level + 1)) - 1u;
        std::uint32_t seg_max = 0;
        std::size_t   seg_changed = 0;
        bool seg_valid = (seg_after.size() == seg_before.size()) && !seg_after.empty();
        if (seg_valid) {
            for (std::size_t i = 0; i < seg_after.size(); ++i) {
                seg_max = std::max(seg_max, seg_after[i]);
                if (seg_after[i] != seg_before[i]) ++seg_changed;
                if (seg_after[i] < 1u || seg_after[i] > max_seg_id) seg_valid = false;
            }
        }
        const bool branch_ok = seg_valid &&
                               (distinct_before == 1) &&       // init is all roots {1}
                               (distinct_after  > 1) &&        // segments split
                               (seg_changed     > 0);
        std::printf("  [branch]     %s: distinct seg_ids %zu -> %zu, max_seg_id=%u, "
                    "%zu/%zu seg cells reassigned after %d steps\n",
                    branch_ok ? "PASS" : "FAIL", distinct_before, distinct_after,
                    seg_max, seg_changed, seg_after.size(), kSteps);
        ok = ok && branch_ok;

        // 6. probe fan-out (S9) — the activated probe received finite values
        //    through the SPSC queue, in-range; the discovered-but-inactive probe
        //    stayed empty (per-instance active gate). Exercises the full path:
        //    register -> activate -> per-step GPU readback -> record -> drain.
        auto drain_count = [](probe::probe_instance* p, std::size_t& n,
                              double& last, bool& finite_ok) {
            n = 0; last = 0.0; finite_ok = true;
            if (p == nullptr) { finite_ok = false; return; }
            p->samples.drain([&](const probe::probe_sample& smp) {
                ++n; last = smp.value;
                if (!std::isfinite(smp.value)) finite_ok = false;
            });
        };
        std::size_t n_ng = 0, n_idle = 0;
        double last_ng = 0.0, last_idle = 0.0;
        bool fin_ng = true, fin_idle = true;
        drain_count(p_ngm3_act,  n_ng,   last_ng,   fin_ng);
        drain_count(p_ngm3_idle, n_idle, last_idle, fin_idle);

        const bool probes_ok =
            (n_ng > 0)  && fin_ng && (last_ng  >= -1e-6) &&                 // ngm3 max activity >= 0
            (n_idle == 0);                                                  // negative control: inactive => empty
        std::printf("  [probes]     %s: ngm3 'max activity' %zu samples (last=%.5f), "
                    "inactive 'winning thres' %zu samples\n",
                    probes_ok ? "PASS" : "FAIL",
                    n_ng, last_ng, n_idle);
        ok = ok && probes_ok;

        // 7. vis mirror (S10) — refresh_vis_mirror() does a bulk GPU->CPU readback
        //    into the CPU mirrors; the per-(n,d,r) accessors index that mirror with
        //    the same flat-SoA offset as the CPU type. The device is idle here (a
        //    step boundary), the exact precondition the GUI vis relies on via the
        //    post-swap barrier. Cross-check the accessor output against the direct
        //    readback (perm_after / seg_after — identical source, no steps between,
        //    so exact equality holds), confirm finite/in-range, the mirror-step
        //    stamp advanced, and the absent apical bank yields empty spans.
        bool vis_ok = true;
        unsigned long long vstep = 0;
        {
            auto ng = env.get_entities<ngm3::neuron_group_gpu_t>();
            auto& g = (*ng.value())[0];
            g.refresh_vis_mirror();
            vstep = static_cast<unsigned long long>(g.get_vis_mirror_step());

            const std::size_t rep_cnt = std::size_t{1} << gp.max_branch_level; // 8
            const std::size_t in_size = kImageSize;                                 // proximal width
            auto offset = [&](std::size_t n, std::size_t d, std::size_t r) {
                return ((n * gp.proximal_cnt + d) * rep_cnt + r) * in_size;
            };
            const std::array<std::array<std::size_t, 3>, 4> samples{{
                {{0, 0, 0}}, {{1, 2, 3}}, {{63, 3, 7}}, {{31, 1, 5}}
            }};
            vis_ok = (vstep > 0) && (perm_after.size() == seg_after.size());
            for (const auto& s : samples) {
                if (!vis_ok) break;
                auto rep = g.get_proximal_representation(s[0], s[1], s[2]);
                auto sid = g.get_proximal_segment_ids(s[0], s[1], s[2]);
                if (rep.size() != in_size || sid.size() != in_size) { vis_ok = false; break; }
                const std::size_t off = offset(s[0], s[1], s[2]);
                if (off + in_size > perm_after.size()) { vis_ok = false; break; }
                for (std::size_t i = 0; i < in_size; ++i) {
                    if (rep[i] != perm_after[off + i] ||
                        !std::isfinite(rep[i]) || rep[i] < -1e-6f || rep[i] > 1.0f + 1e-6f ||
                        sid[i] != static_cast<std::uint8_t>(seg_after[off + i] & 0xffu)) {
                        vis_ok = false; break;
                    }
                }
            }
            // apical bank is absent in this config -> accessors must return empty.
            if (!g.get_apical_representation(0, 0, 0).empty() ||
                !g.get_apical_segment_ids(0, 0, 0).empty()) {
                vis_ok = false;
            }
        }
        std::printf("  [vis-mirror] %s: refresh_vis_mirror() + accessors match direct "
                    "readback (mirror_step=%llu), absent apical bank empty\n",
                    vis_ok ? "PASS" : "FAIL", vstep);
        ok = ok && vis_ok;

        std::printf("\n[coast-ngm3-gpu-probe] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[coast-ngm3-gpu-probe] FAILED: %s\n", ex.what());
        return 1;
    }
}
