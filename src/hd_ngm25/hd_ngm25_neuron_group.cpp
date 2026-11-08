//
// Created 10.07.26 — ngm2.5 increment I6: the io_entity group. process()
// assembles the map §0 step skeleton (R1..R8); the kernels live in
// hd_ngm25_{response,adapt,branch} and this class is the thin caller (the
// dispatch on penalty policy is the only per-step runtime branch). Semantic
// reference: hd_ngm2_neuron_group.cpp process() order.
//

#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm25/hd_ngm25_host.h"

#include <algorithm>
#include <numeric>
#include <array>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/require.h"
#include "hd_ngm2/hd_ngm2_tools.h"          // ngm2::compute_input_stats
#include "hd_ngm25/hd_ngm25_prng.h"          // penalty policy types (response_all<P>)
#include "ngm/common_config_record.h"
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "observe/observe_accessor.h"

namespace ngm25 {

neuron_group_t::neuron_group_t(const cfg_store   &cs,
                               const std::string &type,
                               sim::output_id_t   output_id,
                               seed_offset_t      seed_offset)
  : params(cs.group_or_default(type)),
    id_(output_id)
{
    effective_seed_ = params.random_seed + seed_offset;

    validate_params(params, k_cpu_caps);

    shape_         = base_shape(params);           // input ranges: init_io
    adapt_params_  = make_adapt_params(params);    // map R5/R6/R7
    branch_params_ = make_branch_params(params);   // map R8

    group_outp_.assign(params.neuron_cnt, 0.0f);
    neuron_act_.assign(params.neuron_cnt, 0.0f);
    neuron_mx_act_.assign(params.neuron_cnt, 0.0f);
    input_count_.assign(params.neuron_cnt, 0);

    inhibitor_ = inhibition::make_inhibitor(cs.inhibitions, params.inhibition_cfg,
                                            params.local_inhibition_strength, seed_offset);
}

void neuron_group_t::init_io(
    const std::unordered_map<std::string, sim::output_channel_t> &input_map)
{
    auto routing = route_inputs(params, input_map, shape_, k_cpu_caps, id_);
    routes_  = std::move(routing.routes);
    inp_ids_ = std::move(routing.inp_ids);
    input_concat_.assign(routing.width, 0.0f);

    // Build the arena (D5 Poisson init) and its per-step scratch.
    arena_ = build_arena(shape_, params.permanence_threshold, std::to_underlying(effective_seed_));
    dendrite_act_.assign(arena_.dend_count(), 0.0f);
    const std::size_t seg_cnt = static_cast<std::size_t>(shape_.max_seg_id) + 1;
    seg_activity_.assign(arena_.dend_count() * seg_cnt, 0.0f);
    seg_weights_.assign(arena_.dend_count() * seg_cnt, 0.0f);

    init_structure();
}

void neuron_group_t::set_output_buffer(sim::io_buffer& buf) { output_buf_ = &buf; }

void neuron_group_t::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    for (auto &r : routes_)
        if (r.id == id) r.buf = &buf;
}

void neuron_group_t::response_dispatch()
{
    // One runtime branch per step on the penalty policy (map §8); the hot loops
    // carry zero mode branches (each policy is its own template instantiation).
    switch (params.penalty_policy) {
    case penalty_policy_t::stochastic:
        detail::response_all<stochastic_penalty>(
            arena_, shape_, input_concat_, ctx_, params.permanence_threshold,
            params.neuron_noise_range, std::to_underlying(effective_seed_), step_,
            seg_activity_, dendrite_act_, neuron_act_);
        break;
    case penalty_policy_t::expected_value:
        detail::response_all<expected_penalty>(
            arena_, shape_, input_concat_, ctx_, params.permanence_threshold,
            params.neuron_noise_range, std::to_underlying(effective_seed_), step_,
            seg_activity_, dendrite_act_, neuron_act_);
        break;
    case penalty_policy_t::none:
        detail::response_all<no_penalty>(
            arena_, shape_, input_concat_, ctx_, params.permanence_threshold,
            params.neuron_noise_range, std::to_underlying(effective_seed_), step_,
            seg_activity_, dendrite_act_, neuron_act_);
        break;
    }
}

sim::step_result neuron_group_t::process()
{
    COAST_REQUIRE(output_buf_ != nullptr, core::error_kind::contract_violation);

    // --- gather inputs into the concatenated buffer (map §0.0) ---
    for (const auto &r : routes_) {
        if (r.buf == nullptr) continue;
        const auto src = r.buf->cur_read_buffer();
        COAST_REQUIRE(src.size() == r.pos_hi - r.pos_lo,
                      core::error_kind::contract_violation);
        std::copy(src.begin(), src.end(), input_concat_.begin() + r.pos_lo);
    }

    // --- R1: per-channel stats -> per-position + per-type ctx ---
    std::vector<channel_stats_t> channels;
    channels.reserve(routes_.size());
    for (const auto &r : routes_) {
        const std::span<const float> slice(input_concat_.data() + r.pos_lo,
                                           r.pos_hi - r.pos_lo);
        channels.push_back({r.bank, r.pos_lo, r.pos_hi, ngm2::compute_input_stats(slice)});
    }
    detail::build_step_ctx(shape_, channels, ctx_);

    // --- R2/R3: response + combine (parallel over dendrites) ---
    response_dispatch();

    // --- R4: inhibition, in place (serial) ---
    std::copy(neuron_act_.begin(), neuron_act_.end(), group_outp_.begin());
    inhibitor_->inhibit(group_outp_, group_outp_);

    // --- not a learning step: frozen, or zero group output (SPEC.md M.4.15) ---
    // A frozen group is a pure forward pass. A step whose output is all 0 (the
    // outputs lie in [0,1], so the sum is 0 exactly then) has no neuron to
    // attribute learning to. Both write the output, advance the step key (it
    // keys the stateless draws) and change no learning state.
    const float act_sum = std::reduce(group_outp_.begin(), group_outp_.end(), 0.0f);
    if (params.frozen || act_sum == 0.0f) {
        std::span<float> out = output_buf_->cur_write_buffer();
        COAST_REQUIRE(out.size() == group_outp_.size(), core::error_kind::contract_violation);
        std::copy(group_outp_.begin(), group_outp_.end(), out.begin());
        ++step_;
        return {};
    }

    // --- running max (R3 tail; ngm2 neuron_t::update_max_activity) ---
    // Learning state: updated on learning steps only, before adapt reads it.
    for (std::size_t n = 0; n < params.neuron_cnt; ++n)
        neuron_mx_act_[n] = std::max(neuron_mx_act_[n] * 0.99999f, neuron_act_[n]);

    // --- R5/R6/R7: winner primary (serial) + secondary adapt (parallel) ---
    // The windows are the ones the edges reached at the end of the previous
    // learning step; the edges then advance with this step's raw activities.
    const std::vector<std::uint64_t> count_before = input_count_;
    const std::size_t winner = detail::adapt_all(
        arena_, shape_, input_concat_, ctx_, dendrite_act_, neuron_act_,
        neuron_mx_act_, group_outp_, adapt_params_,
        ngm::window_tracking::winner_window(window_edges_),
        ngm::window_tracking::other_window(window_edges_),
        seg_activity_, seg_weights_, input_count_,
        std::to_underlying(effective_seed_), step_);
    ngm::window_tracking::update(window_edges_, neuron_act_, winner);
    PROBE("window winner lower", window_edges_.winner_lower);   // same names as the ngm2 group's
    PROBE("window winner upper", window_edges_.winner_upper);
    PROBE("window other lower",  window_edges_.other_lower);
    PROBE("window other upper",  window_edges_.other_upper);

    // --- R8: per-neuron branch gate -> decide + re-pack (map §0 step 8) ---
    const bool grew = detail::maybe_branch_group(arena_, shape_, branch_params_,
                                                 count_before, input_count_,
                                                 std::to_underlying(effective_seed_), branch_epoch_);

    // Post-repack safety net (design §9): a re-pack that violates an arena invariant
    // corrupts silently and surfaces only as "learning got worse" much later — the
    // most expensive failure mode to localize. Run the full consistency checker on a
    // branch event (~1/5000 steps, so the O(syn) walk is negligible) and halt via the
    // value channel on a violation instead of degrading in the field. The checker is
    // a strict superset of the NDEBUG-elided splittability asserts inside
    // repack_with_splits, so it also covers those. The invariant string goes to
    // stderr (cold path only) because the value channel can't carry it.
    if (grew) {
        std::string why;
        if (!check_arena_consistency(arena_, shape_, &why)) {
            std::fprintf(stderr, "ngm25: arena inconsistent after repack: %s\n", why.c_str());
            COAST_REQUIRE(false, core::error_kind::contract_violation);
        }
    }

    // --- scatter the post-inhibition output ---
    std::span<float> out = output_buf_->cur_write_buffer();
    COAST_REQUIRE(out.size() == group_outp_.size(), core::error_kind::contract_violation);
    std::copy(group_outp_.begin(), group_outp_.end(), out.begin());

    ++step_;
    return {};
}

sim::output_id_t neuron_group_t::get_outp_id()  const { return id_; }
std::size_t      neuron_group_t::get_outp_size() const { return params.neuron_cnt; }

std::span<const std::size_t> neuron_group_t::get_inp_ids() const
{ return { inp_ids_.data(), inp_ids_.size() }; }

std::string neuron_group_t::status_str() const
{
    return "NGM2.5 Group " + std::to_string(id_)
         + " | neurons "   + std::to_string(params.neuron_cnt)
         + " | apical "    + std::to_string(params.apical_cnt)
         + " | proximal "  + std::to_string(params.proximal_cnt)
         + " | depth "     + std::to_string(static_cast<int>(params.max_branch_level))
         + " | synapses "  + std::to_string(arena_.syn_count())
         + " | step "      + std::to_string(step_);
}

// ---------------------------------------------------------------------------
// Vis / introspection surface (pitfall P5): the dense mirror (hd_ngm25_host.h).
// ---------------------------------------------------------------------------
void neuron_group_t::refresh_vis_mirror() const { mirror_.refresh(arena_, shape_); }

std::span<const float>
neuron_group_t::get_apical_representation(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.perm(bank_t::apical, n, d, r); }

std::span<const float>
neuron_group_t::get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.perm(bank_t::proximal, n, d, r); }

std::span<const std::uint8_t>
neuron_group_t::get_apical_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.seg(bank_t::apical, n, d, r); }

std::span<const std::uint8_t>
neuron_group_t::get_proximal_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.seg(bank_t::proximal, n, d, r); }

std::size_t neuron_group_t::get_total_representation_count() const
{
    std::size_t total = 0;
    for (std::size_t d = 0; d < arena_.dend_count(); ++d) {
        std::array<bool, 256> seen{};
        const auto rng = arena_.dendrite(d);
        for (std::uint32_t s = rng.begin; s < rng.end; ++s)
            if (!seen[arena_.seg[s]]) { seen[arena_.seg[s]] = true; ++total; }
    }
    return total;
}

float neuron_group_t::get_max_mismatch() const
{
    float m = 0.0f;
    for (float v : arena_.mismatch) m = std::max(m, v);
    return m;
}

float neuron_group_t::get_avg_mismatch() const
{
    if (arena_.mismatch.empty()) return 0.0f;
    float s = 0.0f;
    for (float v : arena_.mismatch) s += v;
    return s / static_cast<float>(arena_.mismatch.size());
}

void neuron_group_t::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("population")) {
        acc.observe(*this, "population", "in",  std::span<const float>(input_concat_));
        acc.observe(*this, "population", "out", std::span<const float>(group_outp_));
        // Passive CPU observation, valid in frozen mode as well as training.
        acc.observe(*this, "population", "out_pre_inhibition", std::span<const float>(neuron_act_));
    }
    // Its own group: this scans every segment id, so only a run that asks pays for it.
    if (acc.group_active("population_structure")) {
        acc.observe(*this, "population_structure", "representation_count",
                    static_cast<std::uint64_t>(get_total_representation_count()));
    }
    entity_element::perform_observations(acc);
}

// Persistence (map R9). The hash-RNG payoff: there is NO RNG stream state to
// checkpoint — only step_ and branch_epoch_ (which feed the structural hash
// keys) plus the arena, directory, and per-neuron carried state. Reproducing a
// draw after resume needs the key, not a stream position, so a whole class of
// resume bugs disappears (map §2 note).
logging::config_record config_record(const neuron_group_params_t&           p,
                                     seed_t                                 effective_seed,
                                     const inhibition::inhibition_params_t& inh)
{
    auto policy = [](penalty_policy_t pp) -> std::string {
        switch (pp) {
            case penalty_policy_t::stochastic:     return "stochastic";
            case penalty_policy_t::expected_value: return "expected_value";
            case penalty_policy_t::none:           return "none";
        }
        throw std::runtime_error("ngm25: unhandled penalty_policy_t");
    };
    auto r = ngm::common_config_record(p, effective_seed, inh);
    r.push_back({ "penalty_policy", policy(p.penalty_policy) });
    return r;
}

void neuron_group_t::save_state(logging::state_writer &w) const
{
    const auto who = std::format("ngm25 neuron group {}", id_);
    const auto cfg = config_record(params, effective_seed_, inhibitor_->params());
    write_checkpoint(w, { who, params, shape_, routes_, cfg },
                     step_, branch_epoch_, effective_seed_, arena_,
                     input_count_, neuron_mx_act_, window_edges_);

    // Inhibition (delegated).
    w.begin_group("inhibition");
    inhibitor_->save_state(w);
    w.end_group();
}

void neuron_group_t::load_state(logging::state_reader &r)
{
    // Read and check the whole checkpoint, then the inhibitor's part, before
    // anything of this group is assigned: a misfit or corrupt checkpoint throws
    // and leaves the group as it was (M.7.18).
    const auto who = std::format("ngm25 neuron group {}", id_);
    const auto cfg = config_record(params, effective_seed_, inhibitor_->params());
    checkpoint_state_t s = stage_checkpoint(r, { who, params, shape_, routes_, cfg },
                                            step_, branch_epoch_, effective_seed_);

    if (r.enter_group("inhibition")) { inhibitor_->load_state(r, params.neuron_cnt); r.leave_group(); }

    // Commit (moves only; nothing below throws).
    step_           = s.step;
    branch_epoch_   = s.branch_epoch;
    effective_seed_ = s.effective_seed;
    arena_          = std::move(s.arena);
    input_count_    = std::move(s.input_count);
    neuron_mx_act_  = std::move(s.neuron_mx_act);
    window_edges_   = s.window_edges;
}

} // namespace ngm25
