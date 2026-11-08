//
// Created by jk on 21.03.26.
// Rewritten 25.05.26: two-bank apical/proximal flat-SoA compute.
//

#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hd_ngm3/hd_ngm3_cold_init.h"

#include "core/require.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "hd_ngm3/hd_ngm3_layout.h"
#include "ngm/common_config_record.h"
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ngm3 {

// ---------------------------------------------------------------------------
// M1.3 — constructor + storage allocation
// ---------------------------------------------------------------------------

neuron_group_t::neuron_group_t(const cfg_store     &cs,
                               const std::string   &type,
                               sim::output_id_t     output_id,
                               seed_offset_t        seed_offset)
  : params(cs.group_or_default(type)),
    local_inhibition_strength(params.local_inhibition_strength),
    common_learning_rate     (params.common_learning_rate),
    weight_filter            (params.weight_filter),
    stochastic_win_thres     (params.stochastic_win_thres),
    rgen                     (std::to_underlying(params.random_seed + seed_offset)),
    id                       (output_id)
{
    // Record the effective seed so the run manifest can surface it without
    // re-deriving coast's seed-composition rule (config seed + seed_offset).
    effective_seed_ = params.random_seed + seed_offset;

    validate_params(params, k_cpu_caps);   // hd_ngm3_host.h, shared with the GPU group

    rep_cnt    = std::size_t{1} << params.max_branch_level;
    max_seg_id = static_cast<uint8_t>((1u << (params.max_branch_level + 1)) - 1);

    // Group output and per-neuron scratch can be sized immediately; bank
    // storage waits for init_storage() once concatenated input widths are
    // known (in M2 these come from channel routing).
    group_outp         .assign(params.neuron_cnt, 0.0f);
    neuron_act         .assign(params.neuron_cnt, 0.0f);
    neuron_mx_act      .assign(params.neuron_cnt, 0.0f);
    neuron_apical_max  .assign(params.neuron_cnt, params.apical_cnt == 0 ? 1.0f : 0.0f);
    neuron_proximal_max.assign(params.neuron_cnt, 0.0f);

    // The master rgen and the per-neuron streams (one per neuron, so the
    // parallel response/adapt loops below each draw from their own) are seeded
    // by cold_init() in init_storage, together with the permanences: their draw
    // order is the contract the GPU group shares (hd_ngm3_cold_init.h).

    // Construct this group's inhibition. An empty inhibition_cfg synthesizes a
    // max_softmax with beta = local_inhibition_strength.
    inhibitor_ = inhibition::make_inhibitor(cs.inhibitions, params.inhibition_cfg,
                                            local_inhibition_strength, seed_offset);
}

void neuron_group_t::init_storage(std::size_t apical_in_size, std::size_t proximal_in_size)
{
    // Cold setup contract (no proximal input routed is a config error). Throw, not
    // assert: a no-op under NDEBUG would let a 0-width proximal bank through.
    if (proximal_in_size < 1)
        throw std::runtime_error("ngm3: proximal input size must be >= 1");
    apical_input_size   = (params.apical_cnt == 0) ? 0 : apical_in_size;
    proximal_input_size = proximal_in_size;

    // Permanences and RNG streams from the contract shared with the GPU group
    // (hd_ngm3_cold_init.h); the companion arrays start at their cold values.
    cold_state_t cold = cold_init(effective_seed_, params.neuron_cnt, rep_cnt,
                                  params.permanence_threshold,
                                  { params.apical_cnt,   apical_input_size   },
                                  { params.proximal_cnt, proximal_input_size });
    rgen         = cold.master;
    neuron_rgens = std::move(cold.neuron_rgens);

    auto cold_bank = [&](std::vector<float>   &&cold_perm,
                         std::vector<float>   &perms,
                         std::vector<uint8_t> &seg_ids,
                         std::vector<float>   &adapt,
                         std::vector<float>   &mm,
                         std::vector<float>   &acts,
                         std::size_t           d_cnt)
    {
        const std::size_t total = cold_perm.size();   // 0 for an absent bank
        perms = std::move(cold_perm);
        seg_ids.assign(total, uint8_t{1});            // single base segment
        adapt  .assign(total, 0.0f);
        mm     .assign(total, 0.0f);
        acts   .assign(total == 0 ? 0 : params.neuron_cnt * d_cnt * rep_cnt, 0.0f);
    };
    cold_bank(std::move(cold.apical_perm),   apical_permanences,   apical_segment_ids,
              apical_adapt_history,   apical_mismatch,   apical_rep_act,   params.apical_cnt);
    cold_bank(std::move(cold.proximal_perm), proximal_permanences, proximal_segment_ids,
              proximal_adapt_history, proximal_mismatch, proximal_rep_act, params.proximal_cnt);

    // Response scratch: one slice per neuron, shared by its apical and
    // proximal pass, so compute_step allocates nothing.
    penalised_buf.assign(params.neuron_cnt * std::max(apical_input_size, proximal_input_size),
                         uint8_t{0});

    input_cnt = 0;
}

// ---------------------------------------------------------------------------
// M2.5 — io_entity wiring: init_io, set_input_buffer, set_output_buffer
// ---------------------------------------------------------------------------

void neuron_group_t::init_io(
    const std::unordered_map<std::string, sim::output_channel_t> &input_map)
{
    // Routing and its guards are shared with the GPU group (hd_ngm3_host.h).
    // It builds fresh routing state, so init_io may run more than once during
    // the entity's lifetime (e.g. after a config change in the GUI).
    auto routing    = route_inputs(params, input_map, k_cpu_caps, id);
    apical_inputs   = std::move(routing.apical);
    proximal_inputs = std::move(routing.proximal);
    inp_ids         = std::move(routing.inp_ids);
    const std::size_t a_total = apical_inputs.width();
    const std::size_t p_total = proximal_inputs.width();

    apical_input_buf  .assign(a_total, 0.0f);
    proximal_input_buf.assign(p_total, 0.0f);

    init_storage(a_total, p_total);

    // Build the entity-element structure registry for the probe system.
    init_structure();
}

void neuron_group_t::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

void neuron_group_t::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    apical_inputs  .bind(id, buf);
    proximal_inputs.bind(id, buf);
}

// ---------------------------------------------------------------------------
// bank-view construction (private)
// ---------------------------------------------------------------------------

neuron_group_t::bank_view_t
neuron_group_t::apical_view(std::span<const float> input_buf)
{
    return bank_view_t{
        .permanences   = apical_permanences,
        .segment_ids   = apical_segment_ids,
        .adapt_history = apical_adapt_history,
        .mismatch      = apical_mismatch,
        .rep_act       = apical_rep_act,
        .input_buf     = input_buf,
        .bank_max      = neuron_apical_max,
        .d_cnt         = params.apical_cnt,
        .input_size    = apical_input_size,
    };
}

neuron_group_t::bank_view_t
neuron_group_t::proximal_view(std::span<const float> input_buf)
{
    return bank_view_t{
        .permanences   = proximal_permanences,
        .segment_ids   = proximal_segment_ids,
        .adapt_history = proximal_adapt_history,
        .mismatch      = proximal_mismatch,
        .rep_act       = proximal_rep_act,
        .input_buf     = input_buf,
        .bank_max      = neuron_proximal_max,
        .d_cnt         = params.proximal_cnt,
        .input_size    = proximal_input_size,
    };
}

// ---------------------------------------------------------------------------
// M1.4 — per-bank input statistics
// ---------------------------------------------------------------------------

neuron_group_t::input_stats_t
neuron_group_t::compute_input_stats(std::span<const float> input_buf)
{
    // Delegate to the single shared ngm2 definition (true data max, reduced
    // from 0.0f — no 1.0 floor). For an empty bank (apical_cnt == 0) this
    // returns all-zero stats; those values are unused because the response
    // loop early-returns when input_size == 0.
    return ngm2::compute_input_stats(input_buf);
}

void neuron_group_t::build_bank_ctx(bank_ctx_t &ctx, const bank_inputs_t &routes,
                                    std::span<const float> input_buf)
{
    const std::size_t n = input_buf.size();
    for (auto *v : { &ctx.inv_sum, &ctx.half_max, &ctx.adapt_atten,
                     &ctx.high_thres, &ctx.inv_high_range, &ctx.inv_high_thres })
        v->resize(n);
    ctx.total_sum = 0.0f;
    ctx.max_val   = 0.0f;
    if (n == 0) { ctx.resp_atten = 1.0f; return; }

    // Channel boundaries from the routing; a span the routing does not cover
    // (e.g. compute_step driven directly in a unit test) is one channel.
    std::vector<std::pair<std::size_t, std::size_t>> chans;   // [lo, hi)
    std::size_t covered = 0;
    for (std::size_t c = 0; c < routes.offsets.size(); ++c) {
        chans.emplace_back(routes.offsets[c], routes.offsets[c] + routes.sizes[c]);
        covered += routes.sizes[c];
    }
    if (chans.empty() || covered != n) chans.assign(1, { 0, n });

    ngm2::noise_gate_min_t gate_min;
    for (const auto &[lo, hi] : chans) {
        const input_stats_t st = compute_input_stats(input_buf.subspan(lo, hi - lo));
        ctx.total_sum += st.sum;
        ctx.max_val    = std::max(ctx.max_val, st.max_val);
        gate_min.add(st);
        // The same per-channel constants as ngm25's step context (one definition).
        const ngm2::channel_constants_t c = ngm2::channel_constants(st);
        for (std::size_t p = lo; p < hi; ++p) {
            ctx.inv_sum[p]        = c.inv_sum;
            ctx.half_max[p]       = c.half_max;
            ctx.adapt_atten[p]    = c.adapt_atten;
            ctx.high_thres[p]     = c.high_thres;
            ctx.inv_high_range[p] = c.inv_high_range;
            ctx.inv_high_thres[p] = c.inv_high_thres;
        }
    }
    // Response attenuation from the minimum noise (1 - Hoyer) and average over
    // the channels that carry shape evidence (ngm2::noise_gate_min_t, as ngm2).
    ctx.resp_atten = gate_min.attenuation(ngm2::response_noise_window);
}

// ---------------------------------------------------------------------------
// M2.6 — io_entity process() override: gather partial inputs into concat
// buffers, dispatch to compute_step, push group_outp to output_mem().
// ---------------------------------------------------------------------------

sim::step_result neuron_group_t::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);

    auto gather = [](const bank_inputs_t &b, std::vector<float> &buf) -> sim::step_result
    {
        for (std::size_t i = 0; i < b.bufs.size(); ++i) {
            if (!b.bufs[i]) continue;
            const auto src_span = b.bufs[i]->cur_read_buffer();
            const std::size_t off = b.offsets[i];
            COAST_REQUIRE(src_span.size() == b.sizes[i],
                          core::error_kind::contract_violation);
            std::copy(src_span.begin(), src_span.end(), buf.begin() + off);
        }
        return {};
    };

    if (auto r = gather(apical_inputs,   apical_input_buf);   !r) return r;
    if (auto r = gather(proximal_inputs, proximal_input_buf); !r) return r;

    // The input the group received this step is observed in perform_observations
    // (post-process pass): proximal_input_buf / apical_input_buf are members
    // populated here by gather() and left untouched for the rest of the step, so
    // they are stable when the pass reads them back as population/in + in_apical.

    if (auto r = compute_step(std::span<const float>(apical_input_buf),
                              std::span<const float>(proximal_input_buf)); !r)
        return r;

    std::span<float> out = output_buf->cur_write_buffer();
    COAST_REQUIRE(out.size() == group_outp.size(), core::error_kind::contract_violation);
    std::copy(group_outp.begin(), group_outp.end(), out.begin());
    return {};
}

// ---------------------------------------------------------------------------
// M1.5 — top-level compute_step() orchestration: compute, combine, inhibit, adapt, branch
// ---------------------------------------------------------------------------

namespace {

// Reduce across the [d × rep] block of one neuron, taking the max.
float reduce_neuron_max(std::span<const float> rep_act_block)
{
    if (rep_act_block.empty()) return 0.0f;
    float m = 0.0f;
    for (float v : rep_act_block) if (v > m) m = v;
    return m;
}

} // namespace

void neuron_group_t::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("population")) {
        // proximal_input_buf / apical_input_buf hold this step's input (populated
        // in process() by gather()); group_outp holds this step's post-inhibition
        // output (the adapt section reads but does not modify it). All stable at
        // end-of-step. ngm3 keeps the proximal and apical banks separate:
        //   population/in        — proximal feedforward = the presented sample
        //   population/in_apical — top-down/context input, only when configured
        //   population/out       — post-inhibition output (mirrors ngm2 / ngm3-GPU)
        acc.observe(*this, "population", "in", std::span<const float>(proximal_input_buf));
        if (!apical_input_buf.empty())
            acc.observe(*this, "population", "in_apical", std::span<const float>(apical_input_buf));
        acc.observe(*this, "population", "out", std::span<const float>(group_outp));
        // Passive CPU observation, valid in frozen mode as well as training.
        acc.observe(*this, "population", "out_pre_inhibition", std::span<const float>(neuron_act));
    }
    // Its own group: this scans every segment id, so only a run that asks pays for it.
    if (acc.group_active("population_structure")) {
        acc.observe(*this, "population_structure", "representation_count",
                    static_cast<std::uint64_t>(get_total_representation_count()));
    }
}

sim::step_result neuron_group_t::compute_step(std::span<const float> apical_in,
                                              std::span<const float> proximal_in)
{
    // Bank-input precondition. apical may be empty iff apical_cnt == 0.
    COAST_REQUIRE(proximal_in.size() == proximal_input_size,
                  core::error_kind::contract_violation);
    COAST_REQUIRE((params.apical_cnt == 0 && apical_in.empty()) ||
                  (params.apical_cnt  > 0 && apical_in.size() == apical_input_size),
                  core::error_kind::contract_violation);

    // ---- per-bank, per-channel input statistics (serial; SPEC.md M.3.2) ----
    build_bank_ctx(apical_ctx,   apical_inputs,   apical_in);
    build_bank_ctx(proximal_ctx, proximal_inputs, proximal_in);

    bank_view_t a_view = apical_view(apical_in);
    bank_view_t p_view = proximal_view(proximal_in);

    // Per-bank constants hoisted out of the per-neuron loop.
    const float inv_perm_range = 1.0f / (1.0f - params.permanence_threshold);
    const float perm_thres     = params.permanence_threshold;

    const std::size_t a_block = params.apical_cnt   * rep_cnt;
    const std::size_t p_block = params.proximal_cnt * rep_cnt;
    const std::size_t pen_stride = std::max(apical_input_size, proximal_input_size);   // penalised_buf slice

    // ---- FUSED parallel_for: both banks' response + per-neuron combine ----
    // One barrier instead of three (was: apical response + proximal response
    // + combine, each its own parallel_for). Each neuron's combine reads
    // rep_act values that were just written by the same iteration.
    auto response_one_bank = [&](const bank_view_t &v, std::size_t n,
                                 const bank_ctx_t &ctx, fast_rng_t &nrng)
    {
        if (v.d_cnt == 0 || v.input_size == 0) return;
        // The penalty's draw range and 1/sum are the synapse's own channel's
        // (ctx.half_max / ctx.inv_sum per input position); the attenuation and
        // the leaf normaliser are bank-wide (min over channels / total sum).
        std::uniform_real_distribution<float> low_dis;
        using low_range = std::uniform_real_distribution<float>::param_type;
        const float inv_total = ctx.total_sum > 0.0f ? 1.0f / ctx.total_sum : 0.0f;
        const std::size_t cells_per_n = v.d_cnt * rep_cnt;
        // One penalty draw per logical synapse per step (SPEC.md M.0.6 b).
        // A dendrite's rows are visited in order; an input whose
        // segment id in row r equals row r-1's is the same logical synapse
        // (a replicated copy), so it reuses that draw's outcome instead of
        // drawing again. Drawing per row made an unbranched dendrite's activity
        // the max of 2^L noisy evaluations, biased upward. The outcomes live in
        // this neuron's slice of penalised_buf; an entry is only read for a
        // reused synapse, after the previous row wrote it, so it needs no reset.
        uint8_t *penalised = penalised_buf.data() + n * pen_stride;
        for (std::size_t local = 0; local < cells_per_n; ++local) {
            const std::size_t cell = n * cells_per_n + local;
            std::span<float> dendrite_rep {
                v.permanences.data() + cell * v.input_size, v.input_size };
            const uint8_t *row_seg = v.segment_ids.data() + cell * v.input_size;
            const uint8_t *prev_seg = (local % rep_cnt) ? row_seg - v.input_size : nullptr;
            // Per-SEGMENT sums, matching ngm2 (hd_ngm2_dendrite.cpp get_response):
            // each segment accumulates its own synapses and is clamped to >= 0 after
            // each penalised synapse; a leaf's activity is then the sum of the
            // segment sums along its path. A row holds exactly one path (root to
            // its leaf), so this row's segment ids are that path's segments. (One
            // running sum over the whole path, clamped as a whole, let a negative
            // contribution in one segment cancel positive activity in another;
            // SPEC.md M.7.31.)
            std::array<float, MAX_SEG_CNT> seg_acc{};
            uint8_t deepest = 1;
            for (std::size_t s = 0; s < v.input_size; ++s) {
                const float p_val = dendrite_rep[s];
                if (p_val <= perm_thres) continue;   // connected iff p > tau, as in ngm2/ngm25 and this group's mismatch test
                const uint8_t sid = row_seg[s];
                deepest = std::max(deepest, sid);
                const float inp_val = v.input_buf[s];
                float &acc = seg_acc[sid];
                acc += inp_val;
                // Replicas are coherent, so a reused synapse is connected in the
                // previous row too and penalised[s] holds its outcome.
                const bool same_synapse = prev_seg && prev_seg[s] == sid;
                if (!same_synapse) penalised[s] = low_dis(nrng, low_range{ 0.0f, ctx.half_max[s] }) > inp_val;
                if (penalised[s]) {
                    const float perm_strength =
                        (p_val - perm_thres) * inv_perm_range;
                    acc -= perm_strength * (1.0f - inp_val * ctx.inv_sum[s]);
                    if (acc < 0.0f) acc = 0.0f;   // per-segment floor
                }
            }
            // Leaf = sum of the path's segment sums, accumulated from the root
            // down in the same order ngm2 pushes activity root -> leaf
            // (child += parent), so the float result is identical.
            float leaf = 0.0f;
            for (int shift = std::bit_width(static_cast<unsigned>(deepest)) - 1; shift >= 0; --shift)
                leaf = seg_acc[deepest >> shift] + leaf;
            v.rep_act[cell] = std::clamp(leaf * ctx.resp_atten * inv_total, 0.0f, 1.0f);
        }
    };

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, params.neuron_cnt),
        [&](const tbb::blocked_range<std::size_t> &r) {
            std::uniform_real_distribution<float> noise_dis(
                params.neuron_noise_range.first, params.neuron_noise_range.second);
            for (std::size_t n = r.begin(); n != r.end(); ++n) {
                auto &nrng = neuron_rgens[n];

                // Response — apical (optional) then proximal.
                response_one_bank(a_view, n, apical_ctx,   nrng);
                response_one_bank(p_view, n, proximal_ctx, nrng);

                // Combine — both banks' rep_act for this neuron are now fresh.
                if (params.apical_cnt > 0) {
                    neuron_apical_max[n] = reduce_neuron_max(
                        std::span<const float>{ a_view.rep_act.data() + n * a_block, a_block });
                }
                neuron_proximal_max[n] = reduce_neuron_max(
                    std::span<const float>{ p_view.rep_act.data() + n * p_block, p_block });

                neuron_act[n] = std::clamp(
                    neuron_apical_max[n] * neuron_proximal_max[n] + noise_dis(nrng),
                    0.0f, 1.0f);
            }
        },
        tbb::auto_partitioner{}
    );

    // ---- neuron_act -> group_outp via swappable inhibition ----
    // The inhibitor copies neuron_act -> group_outp, then transforms it in place.
    inhibitor_->inhibit(neuron_act, group_outp);

    // ---- frozen (inference-only) mode ----
    // Everything below mutates learning state (winner draw, adapt pass, branch
    // section). A frozen group is a pure forward pass: stop here.
    if (params.frozen) {
        return {};
    }


    // ---- zero group output (SPEC.md M.4.15) ----
    // When every output is 0 (they lie in [0,1], so the sum is 0 exactly then)
    // no neuron produced output to attribute learning to: the step is not a
    // learning step and, like a frozen one, changes no learning state.
    const float act_sum =
        std::reduce(group_outp.begin(), group_outp.end(), 0.0f);
    if (act_sum == 0.0f) {
        return {};
    }

    // Decaying running max — read only by the adapt loop's activity-learning-
    // window sigmoid. Learning state, so it is updated here, on learning steps
    // only, rather than in the response pass.
    for (std::size_t n = 0; n < params.neuron_cnt; ++n) {
        neuron_mx_act[n] *= 0.99999f;
        neuron_mx_act[n] = std::max(neuron_mx_act[n], neuron_act[n]);
    }

    // ---- adapt section (M1.6) ----
    const float mx_act = *std::max_element(group_outp.begin(), group_outp.end());
    std::uniform_real_distribution<float> win_dis(mx_act * stochastic_win_thres, mx_act);
    const float win_act = win_dis(rgen);
    const float win_pick = std::uniform_real_distribution<float>(0.0f, 1.0f)(rgen);
    const ngm2::sigmoid_params_t wf_params(weight_filter);

    // Pick the winner serially before the parallel adapt loop: uniform among the
    // neurons clearing win_act (SPEC.md M.4.4); neuron_cnt if none does, which
    // matches no neuron below.
    const std::size_t winner_idx = ngm2::pick_winner(group_outp, win_act, win_pick);

    // The tracked activity learning windows the edges reached at the end of the
    // previous learning step (ngm/window_tracking.h).
    const learning_window_t window_winner = ngm::window_tracking::winner_window(window_edges);
    const learning_window_t window_other  = ngm::window_tracking::other_window(window_edges);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, params.neuron_cnt),
        [&](const tbb::blocked_range<std::size_t> &r) {
            for (std::size_t n = r.begin(); n != r.end(); ++n) {

                // Skip neurons whose combined activity is degenerate — no
                // meaningful direction for adaptation.
                if (!std::isnormal(neuron_proximal_max[n]) &&
                    !(params.apical_cnt > 0 && std::isnormal(neuron_apical_max[n])))
                    continue;

                float adapt_weight;
                if (n == winner_idx) {
                    adapt_weight = ngm2::sigmoid(1.0f - group_outp[n], wf_params);
                } else {
                    const float ratio = act_sum > 0.0f ? group_outp[n] / act_sum : 0.0f;
                    adapt_weight = ngm2::sigmoid(1.0f - ratio, wf_params) * common_learning_rate;
                }

                // Activity-learning-window envelope: scale adapt_weight by the
                // current combined activity relative to the decaying max — same
                // formula as the sketch / ngm2, with the winner's or the others'
                // tracked window.
                const learning_window_t &window = (n == winner_idx) ? window_winner : window_other;
                const float synapse_weight = adapt_weight * std::min(
                    std::max(
                        1.0f - ngm2::sigmoid(neuron_mx_act[n]),
                        ngm2::sigmoid(neuron_act[n], window.first)
                    ),
                    1.0f - ngm2::sigmoid(neuron_act[n], window.second)
                );

                if (params.apical_cnt > 0 && std::isnormal(neuron_apical_max[n]))
                    adapt_bank_neuron(a_view, apical_ctx, n, synapse_weight);
                if (std::isnormal(neuron_proximal_max[n]))
                    adapt_bank_neuron(p_view, proximal_ctx, n, synapse_weight);
            }
        },
        tbb::auto_partitioner{}
    );

    // Advance the window edges with this step's raw (pre-inhibition) activities.
    ngm::window_tracking::update(window_edges, neuron_act, winner_idx);
    PROBE("window winner lower", window_edges.winner_lower);   // same names as the ngm2 group's
    PROBE("window winner upper", window_edges.winner_upper);
    PROBE("window other lower",  window_edges.other_lower);
    PROBE("window other upper",  window_edges.other_upper);

    // ---- branch section (M1.7) ----
    // Branching fires every params.branch_interval steps. The post-
    // increment means the FIRST firing happens at step branch_interval,
    // matching the sketch.
    if ((++input_cnt) % params.branch_interval != 0)
        return {};

    if (params.apical_cnt > 0) branch_bank(a_view);
    branch_bank(p_view);
    return {};
}

// ---------------------------------------------------------------------------
// M1.6 — adapt_bank_neuron: pre-compute segment weights, fold tree,
//        then sweep all (rep, syn) cells of one neuron's dendrites in
//        one bank.
// ---------------------------------------------------------------------------

void neuron_group_t::adapt_bank_neuron(const bank_view_t   &v,
                                       const bank_ctx_t    &ctx,
                                       std::size_t          n_idx,
                                       float                synapse_weight)
{
    if (v.d_cnt == 0 || v.input_size == 0) return;

    const float max_activity = v.bank_max[n_idx];
    if (!std::isnormal(max_activity)) return;
    const float inv_max_activity = 1.0f / max_activity;

    const uint8_t leaf_begin = static_cast<uint8_t>((max_seg_id + 1) / 2);

    // Attenuation and the high/low threshold are the input's own channel's
    // (per position, as ngm2 adapts per partial input); the mismatch normaliser
    // is the max over all channels.
    const float inv_max_inp = ctx.max_val > 0.0f ? 1.0f / ctx.max_val : 0.0f;

    const std::size_t dendrite_block = rep_cnt * v.input_size;

    for (std::size_t d = 0; d < v.d_cnt; ++d) {

        // Slices over this single dendrite's reps.
        std::span<float> dendrite_act {
            v.rep_act.data() + n_idx * v.d_cnt * rep_cnt + d * rep_cnt,
            rep_cnt };

        // ---- pre-compute segment_weights + segment_activity ----
        std::array<float, MAX_SEG_CNT> segment_weights  {};
        std::array<float, MAX_SEG_CNT> segment_activity {};
        bool max_response_seen = false;
        for (uint8_t si = leaf_begin; si <= max_seg_id; ++si) {
            segment_activity[si] = dendrite_act[si - leaf_begin];
            if (!max_response_seen &&
                segment_activity[si] + std::numeric_limits<float>::epsilon() >= max_activity)
            {
                max_response_seen   = true;
                segment_weights[si] = synapse_weight * params.primary_learning_rate;
            } else {
                segment_weights[si] = segment_activity[si] * synapse_weight
                                    * params.secondary_learning_rate * inv_max_activity;
            }
        }
        // Fold pairs up the tree.
        for (uint8_t level_start = leaf_begin; level_start > 1; level_start = static_cast<uint8_t>(level_start / 2)) {
            for (uint8_t si = level_start; si < level_start * 2; si = static_cast<uint8_t>(si + 2)) {
                const uint8_t parent = static_cast<uint8_t>(si / 2);
                segment_weights [parent] = std::max(segment_weights [si], segment_weights [si + 1]);
                segment_activity[parent] = std::max(segment_activity[si], segment_activity[si + 1]);
            }
        }

        // ---- sweep all (rep, syn) cells of this dendrite ----
        std::span<float>   dendrite_perm {
            v.permanences  .data() + n_idx * v.d_cnt * dendrite_block + d * dendrite_block,
            dendrite_block };
        std::span<uint8_t> dendrite_seg_ids {
            v.segment_ids  .data() + n_idx * v.d_cnt * dendrite_block + d * dendrite_block,
            dendrite_block };
        std::span<float>   dendrite_adapt_history {
            v.adapt_history.data() + n_idx * v.d_cnt * dendrite_block + d * dendrite_block,
            dendrite_block };
        std::span<float>   dendrite_mismatch {
            v.mismatch     .data() + n_idx * v.d_cnt * dendrite_block + d * dendrite_block,
            dendrite_block };

        // Decomposed (r, i): the [rep][synapse] layout means syn = r*input_size + i,
        // so cur_inp = input_buf[i] is invariant in the inner loop. Removes the
        // runtime integer modulo and lets the compiler hoist comparisons.
        for (std::size_t r = 0; r < rep_cnt; ++r) {
            const std::size_t row_base = r * v.input_size;
            for (std::size_t i = 0; i < v.input_size; ++i) {
                const std::size_t syn = row_base + i;
                const float cur_inp = v.input_buf[i];
                const uint8_t sid   = dendrite_seg_ids[syn];

                const float high_thres = ctx.high_thres[i];
                const float theta = std::clamp(
                    segment_weights[sid] * ( cur_inp > high_thres ?
                                             (cur_inp - high_thres) * ctx.inv_high_range[i] :
                                             (high_thres - cur_inp) * ctx.inv_high_thres[i]
                                           ) * ctx.adapt_atten[i],
                    0.0f, 1.0f);

                // adapt permanence
                dendrite_perm[syn] = std::clamp(
                    dendrite_perm[syn] * (1.0f - theta)
                      + (cur_inp > high_thres ? theta : 0.0f),
                    0.0f, 1.0f);

                dendrite_adapt_history[syn] += theta;

                // smoothed mismatch (only when segment is highly active)
                const float act_ratio = segment_activity[sid] * inv_max_activity;
                if (act_ratio >= params.mismatch_act_thres) {
                    const float inp_ratio = cur_inp * inv_max_inp;
                    float mm = dendrite_perm[syn] > params.permanence_threshold
                               ? 1.0f - inp_ratio
                               : inp_ratio;
                    mm *= act_ratio;
                    dendrite_mismatch[syn] =
                        dendrite_mismatch[syn] * (1.0f - params.mismatch_smoothing)
                        + mm * params.mismatch_smoothing;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M1.7 — branch_bank: split mature, high-mismatch synapses into two
//        deeper segments. Iterates (neuron, dendrite) of one bank.
// ---------------------------------------------------------------------------

namespace {

// Calls f(idx) once per synapse of a dendrite block, i.e. once per run of
// equal segment ids down each input column (hd_ngm3_layout.h): row 0 of every
// input, then every later row whose id differs from the row above. Visits in
// input-major order, so sums taken in f are the same in every caller.
template <class F>
void for_each_synapse(std::span<const uint8_t> seg_ids, std::size_t input_size,
                      std::size_t rep_cnt, F &&f)
{
    for (std::size_t inp = 0; inp < input_size; ++inp) {
        f(inp);
        for (std::size_t r = 1; r < rep_cnt; ++r) {
            const std::size_t prev = inp + (r - 1) * input_size;
            const std::size_t cur  = prev + input_size;
            if (seg_ids[prev] != seg_ids[cur]) f(cur);
        }
    }
}

} // namespace

void neuron_group_t::branch_bank(const bank_view_t &v)
{
    if (v.d_cnt == 0 || v.input_size == 0) return;

    const std::size_t dendrite_block = rep_cnt * v.input_size;
    std::uniform_real_distribution<float> rdis(-0.1f, 0.1f);

    for (std::size_t n = 0; n < params.neuron_cnt; ++n) {
        for (std::size_t d = 0; d < v.d_cnt; ++d) {

            auto block = [&](auto &bank) {
                return std::span{ bank.data() + (n * v.d_cnt + d) * dendrite_block, dendrite_block };
            };
            const std::span<float>   dendrite_perm          = block(v.permanences);
            const std::span<uint8_t> dendrite_seg_ids       = block(v.segment_ids);
            const std::span<float>   dendrite_adapt_history = block(v.adapt_history);
            const std::span<float>   dendrite_mismatch      = block(v.mismatch);

            // ---- mismatch mean and spread over the dendrite's synapses ----
            float mm_avg           = 0.0f;
            float real_synapse_cnt = 0.0f;
            for_each_synapse(dendrite_seg_ids, v.input_size, rep_cnt, [&](std::size_t i) {
                mm_avg += dendrite_mismatch[i];
                real_synapse_cnt += 1.0f;
            });
            if (real_synapse_cnt <= 0.0f) continue;
            mm_avg /= real_synapse_cnt;

            float mm_std = 0.0f;
            for_each_synapse(dendrite_seg_ids, v.input_size, rep_cnt, [&](std::size_t i) {
                const float dv = mm_avg - dendrite_mismatch[i];
                mm_std += dv * dv;
            });
            mm_std /= real_synapse_cnt;

            const float mm_thres = mm_avg
                                 + mm_std * params.min_mismatch_deviation
                                 + 1.0f / static_cast<float>(v.input_size);

            // ---- count qualifying synapses (predicate triplet) ----
            auto qualifies = [&](std::size_t idx) {
                return  dendrite_adapt_history[idx] >= params.accumulated_theta_thres
                     && dendrite_mismatch[idx]      >= mm_thres
                     && (dendrite_seg_ids[idx] * 2 + 1) <= max_seg_id;
            };
            float mm_cnt = 0.0f;
            for_each_synapse(dendrite_seg_ids, v.input_size, rep_cnt, [&](std::size_t i) {
                if (qualifies(i)) mm_cnt += 1.0f;
            });

            if (mm_cnt < static_cast<float>(v.input_size) * params.min_mismatch_percentage)
                continue;

            // ---- split-write ----
            for (std::size_t inp = 0; inp < v.input_size; ++inp) {
                for (std::size_t r = 0; r < rep_cnt; ) {
                    const std::size_t idx = inp + r * v.input_size;
                    if (qualifies(idx)) {
                        const uint8_t old_sid       = dendrite_seg_ids[idx];
                        const int     split_size_i  = 1 << (params.max_branch_level - std::bit_width(old_sid));
                        const std::size_t split_size = static_cast<std::size_t>(split_size_i);
                        const uint8_t new_sid_a     = static_cast<uint8_t>(old_sid * 2);
                        const uint8_t new_sid_b     = static_cast<uint8_t>(new_sid_a + 1);
                        const float   old_perm      = dendrite_perm[idx];
                        const float   new_perm_a    = std::clamp(old_perm + rdis(rgen), 0.0f, 1.0f);
                        const float   new_perm_b    = std::clamp(old_perm + rdis(rgen), 0.0f, 1.0f);

                        for (std::size_t s = r; s < r + split_size; ++s) {
                            const std::size_t a_idx = inp + s             * v.input_size;
                            const std::size_t b_idx = inp + (s+split_size)* v.input_size;
                            dendrite_seg_ids      [a_idx] = new_sid_a;
                            dendrite_seg_ids      [b_idx] = new_sid_b;
                            dendrite_adapt_history[a_idx] = 0.0f;
                            dendrite_adapt_history[b_idx] = 0.0f;
                            dendrite_mismatch     [a_idx] = 0.0f;
                            dendrite_mismatch     [b_idx] = 0.0f;
                            dendrite_perm         [a_idx] = new_perm_a;
                            dendrite_perm         [b_idx] = new_perm_b;
                        }
                        r += split_size * 2;
                    } else {
                        ++r;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M1.8 — helper accessors
// ---------------------------------------------------------------------------

namespace {

// Compute the start offset of the (n, d, r) representation slice within a
// bank's flat array. Shape is [n × d × rep × input_size].
inline std::size_t rep_offset(std::size_t n, std::size_t d_cnt,
                              std::size_t d, std::size_t rep_cnt,
                              std::size_t r, std::size_t input_size)
{
    return n * d_cnt * rep_cnt * input_size
         + d         * rep_cnt * input_size
         + r                  * input_size;
}

// Count distinct seg_ids appearing in a dendrite block.
std::size_t count_distinct_seg_ids(std::span<const uint8_t> block)
{
    std::array<bool, 256> seen {};
    std::size_t cnt = 0;
    for (uint8_t sid : block) {
        if (!seen[sid]) { seen[sid] = true; ++cnt; }
    }
    return cnt;
}

} // namespace

std::span<const float>
neuron_group_t::get_apical_representation(std::size_t n, std::size_t d, std::size_t r) const
{
    if (params.apical_cnt == 0) return {};
    const std::size_t off = rep_offset(n, params.apical_cnt, d, rep_cnt, r, apical_input_size);
    return { apical_permanences.data() + off, apical_input_size };
}

std::span<const float>
neuron_group_t::get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const
{
    const std::size_t off = rep_offset(n, params.proximal_cnt, d, rep_cnt, r, proximal_input_size);
    return { proximal_permanences.data() + off, proximal_input_size };
}

std::span<const uint8_t>
neuron_group_t::get_apical_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{
    if (params.apical_cnt == 0) return {};
    const std::size_t off = rep_offset(n, params.apical_cnt, d, rep_cnt, r, apical_input_size);
    return { apical_segment_ids.data() + off, apical_input_size };
}

std::span<const uint8_t>
neuron_group_t::get_proximal_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{
    const std::size_t off = rep_offset(n, params.proximal_cnt, d, rep_cnt, r, proximal_input_size);
    return { proximal_segment_ids.data() + off, proximal_input_size };
}

std::size_t neuron_group_t::get_total_representation_count() const
{
    std::size_t total = 0;

    auto count_bank = [&](const std::vector<uint8_t> &seg_ids,
                          std::size_t d_cnt,
                          std::size_t input_size)
    {
        if (d_cnt == 0 || input_size == 0) return;
        const std::size_t dendrite_block = rep_cnt * input_size;
        for (std::size_t d_block = 0; d_block < params.neuron_cnt * d_cnt; ++d_block) {
            std::span<const uint8_t> block { seg_ids.data() + d_block * dendrite_block, dendrite_block };
            total += count_distinct_seg_ids(block);
        }
    };
    count_bank(apical_segment_ids,   params.apical_cnt,   apical_input_size);
    count_bank(proximal_segment_ids, params.proximal_cnt, proximal_input_size);
    return total;
}

float neuron_group_t::get_max_mismatch() const
{
    float m = 0.0f;
    if (!apical_mismatch.empty())
        m = std::max(m, *std::max_element(apical_mismatch.begin(), apical_mismatch.end()));
    if (!proximal_mismatch.empty())
        m = std::max(m, *std::max_element(proximal_mismatch.begin(), proximal_mismatch.end()));
    return m;
}

float neuron_group_t::get_avg_mismatch() const
{
    const double sum = std::reduce(apical_mismatch.begin(),   apical_mismatch.end(),   0.0)
                     + std::reduce(proximal_mismatch.begin(), proximal_mismatch.end(), 0.0);
    const double cnt = static_cast<double>(apical_mismatch.size() + proximal_mismatch.size());
    return cnt > 0.0 ? static_cast<float>(sum / cnt) : 0.0f;
}

// ---------------------------------------------------------------------------
// M2.7 — io_entity metadata: get_outp_id / get_outp_size / get_inp_ids / status_str
// ---------------------------------------------------------------------------

sim::output_id_t neuron_group_t::get_outp_id() const
{
    return id;
}

std::size_t neuron_group_t::get_outp_size() const
{
    return params.neuron_cnt;
}

std::span<const std::size_t> neuron_group_t::get_inp_ids() const
{
    return { inp_ids.data(), inp_ids.size() };
}

std::string neuron_group_t::status_str() const
{
    std::string s { "NGM3 Group" };
    s += " | id: "             + std::to_string(get_outp_id());
    s += "\n | neurons: "      + std::to_string(get_neuron_count());
    s += " | apical/prox: "   + std::to_string(params.apical_cnt)
       + "/"                  + std::to_string(params.proximal_cnt);
    s += " | depth: "          + std::to_string(static_cast<int>(params.max_branch_level));
    s += " | reps: "           + std::to_string(get_total_representation_count());
    s += " | max mm: "         + std::to_string(get_max_mismatch());
    s += " | avg mm: "         + std::to_string(get_avg_mismatch());
    return s;
}

// ---------------------------------------------------------------------------
// M2.8 — save_state / load_state (HDF5 persistence)
// ---------------------------------------------------------------------------

logging::config_record config_record(const neuron_group_params_t&           p,
                                     seed_t                                 effective_seed,
                                     const inhibition::inhibition_params_t& inh)
{
    return ngm::common_config_record(p, effective_seed, inh);
}

logging::config_record neuron_group_t::current_config() const
{
    // The runtime-tunable copies, not the params they started from.
    auto p = params;
    p.local_inhibition_strength = local_inhibition_strength;
    p.common_learning_rate      = common_learning_rate;
    p.weight_filter             = weight_filter;
    p.stochastic_win_thres      = stochastic_win_thres;
    return config_record(p, effective_seed_, inhibitor_->params());
}

void neuron_group_t::save_state(logging::state_writer &w) const
{
    // Header. Settings are not state (M.7.8): they are in the configuration
    // record written last. effective_seed is for the GPU group, whose draws are
    // keyed on it — a CPU checkpoint resumed on GPU keeps the CPU run's seed.
    w.write("input_cnt", static_cast<std::uint64_t>(input_cnt));
    w.write("effective_seed", std::to_underlying(effective_seed_));

    write_checkpoint_shape(w, { params, apical_input_size, proximal_input_size,
                                apical_inputs, proximal_inputs });

    auto save_bank = [&](std::string_view bank_name,
                         const std::vector<float>   &perm,
                         const std::vector<uint8_t> &seg,
                         const std::vector<float>   &adapt,
                         const std::vector<float>   &mm)
    {
        if (perm.empty()) return;
        w.begin_group(bank_name);
        w.write("permanences"  , std::span<const float>  (perm));
        w.write("segment_ids"  , std::span<const uint8_t>(seg));
        w.write("adapt_history", std::span<const float>  (adapt));
        w.write("mismatch"     , std::span<const float>  (mm));
        w.end_group();
    };
    save_bank("apical",   apical_permanences,   apical_segment_ids,
              apical_adapt_history, apical_mismatch);
    save_bank("proximal", proximal_permanences, proximal_segment_ids,
              proximal_adapt_history, proximal_mismatch);

    // Decaying per-neuron running max (carried across steps, read in adapt).
    w.write("neuron_mx_act", std::span<const float>(neuron_mx_act));
    ngm::window_tracking::save(w, window_edges);

    // PRNG state: the group master plus every per-neuron stream. Required for a
    // bit-identical continuation of the stochastic paths after a resume.
    logging::write_rng(w, "rng", rgen);
    w.begin_group("neuron_rgens");
    w.write("count", static_cast<std::uint64_t>(neuron_rgens.size()));
    for (std::size_t i = 0; i < neuron_rgens.size(); ++i)
        logging::write_rng(w, std::to_string(i), neuron_rgens[i]);
    w.end_group();

    // Inhibition state (scoped). max_softmax is stateless so this is then an
    // empty group, but stateful kinds round-trip here — string-keyed, like rng.
    w.begin_group("inhibition");
    inhibitor_->save_state(w);
    w.end_group();

    logging::write_config_record(w, current_config());
}

void neuron_group_t::load_state(logging::state_reader &r)
{
    // A checkpoint from a differently shaped group cannot be loaded into this one.
    const auto who = std::format("ngm3 neuron group {}", id);
    check_checkpoint_shape(r, { params, apical_input_size, proximal_input_size,
                                apical_inputs, proximal_inputs }, who);

    // The configuration wins (M.7.8): settings keep the values this group was
    // built with, whatever the checkpoint recorded or `frozen` says; only the
    // differences are reported. This runs before the payload checks below, so a
    // corrupt checkpoint saved under another configuration reports both.
    logging::check_config_record(r, current_config(), who);

    // The payload is read and checked before anything is assigned: a corrupt
    // checkpoint throws and leaves this group as it was.
    auto apical   = stage_bank(r, "apical",   params, params.apical_cnt,   apical_input_size,   who);
    auto proximal = stage_bank(r, "proximal", params, params.proximal_cnt, proximal_input_size, who);
    auto mx       = read_neuron_mx_act(r, params.neuron_cnt, who);

    // Window edges, checked here and assigned below with the rest (a format-1
    // checkpoint has none: the trackers start over, see read()).
    const auto edges = ngm::window_tracking::read(r, who);

    // PRNG state, staged in copies like the rest so a corrupt stream throws
    // before anything is assigned. An absent group leaves a seeded stream (a
    // pre-field checkpoint); the per-neuron streams are written all together, so
    // one missing inside a present neuron_rgens is corrupt. A GPU checkpoint has
    // no streams: the fresh ones stay, seeded from the configuration (the two
    // derive their draws differently, M.7.16).
    auto staged_rgen         = rgen;
    auto staged_neuron_rgens = neuron_rgens;   // sized to the validated neuron_cnt
    const bool has_rng = logging::read_rng(r, "rng", staged_rgen, who);
    if (r.enter_group("neuron_rgens")) {
        for (std::size_t i = 0; i < staged_neuron_rgens.size(); ++i)
            if (!logging::read_rng(r, std::to_string(i), staged_neuron_rgens[i],
                                   std::format("{}: neuron {}", who, i)))
                throw corrupt_checkpoint(who, std::format("rng of neuron {} missing", i));
        r.leave_group();
    }
    const auto staged_input_cnt = static_cast<std::size_t>(r.read_uint64("input_cnt", 0));
    // The streams are the checkpoint's, so is the seed they came from.
    const seed_t staged_seed = has_rng
        ? seed_t{ r.read_uint32("effective_seed", std::to_underlying(effective_seed_)) }
        : effective_seed_;

    // The inhibitor is all or nothing too (inhibitor_t::load_state), and is the
    // last part that can throw: after it, this group commits by moves only.
    if (r.enter_group("inhibition")) {
        inhibitor_->load_state(r, params.neuron_cnt);
        r.leave_group();
    }

    auto commit_bank = [](staged_bank_t &b, std::vector<float> &perm, std::vector<uint8_t> &seg,
                          std::vector<float> &adapt, std::vector<float> &mm)
    {
        if (!b.present) return;
        perm  = std::move(b.perm);
        seg   = std::move(b.seg);
        adapt = std::move(b.adapt);
        mm    = std::move(b.mm);
    };
    commit_bank(apical,   apical_permanences,   apical_segment_ids,
                apical_adapt_history,   apical_mismatch);
    commit_bank(proximal, proximal_permanences, proximal_segment_ids,
                proximal_adapt_history, proximal_mismatch);

    if (mx)   // a pre-field checkpoint keeps the constructed running max
        neuron_mx_act = std::move(*mx);
    window_edges    = edges;
    input_cnt       = staged_input_cnt;
    effective_seed_ = staged_seed;
    rgen            = staged_rgen;
    neuron_rgens    = std::move(staged_neuron_rgens);
}

} // namespace ngm3
