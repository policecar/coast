//
// Created 09.07.26 — ngm2.5 scaffold (design/ngm2_5_flat_sparse.md).
// This header is a CONTRACT: signatures, invariants, and semantic anchors for
// the implementing agent. Implementation must follow
// design/ngm2_5_transcription_map.md, not a fresh reading of the ngm2 source.
//

#ifndef HD_NGM25_CFG_H
#define HD_NGM25_CFG_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/seed.h"
#include "hd_ngm2/hd_ngm2_tools.h"              // sigmoid_shape_t, input_stats_t (shared)
#include "inhibition/inhibition_cfg.h"
#include "ngm/bank.h"

namespace ngm25 {

using sigmoid_shape_t   = ngm2::sigmoid_shape_t;
using learning_window_t = std::pair<sigmoid_shape_t, sigmoid_shape_t>;

// Dendrite bank identity (ngm/bank.h, shared with ngm3). One arena stores both
// types (§13.1); dend_type metadata separates them wherever the algorithm
// distinguishes (combine R3, adapt weighting R6, per-type stats R1).
using ngm::bank_t;
using ngm::input_route_t;

// Penalty-draw policy (§8 research seam). Selected at runtime from config;
// all variants are compiled (policy-template dispatch, one branch per group
// per step — never per synapse).
enum class penalty_policy_t : uint8_t {
    stochastic     = 0,   // ngm2 semantics via structural-key hash draw (D1)
    expected_value = 1,   // deterministic E[penalty]; untested middle ground
    none           = 2,   // ablation: no penalty (known-negative in earlier
                          // experiments when removed outright — research only)
};

// New flat schema (decision §13.3, user's call). NOTE the obligation that
// came with that decision: a generator script translating an ngm2 group
// config into this schema is part of v1 tooling, so ngm2-vs-ngm2.5
// comparison configs are derived, never hand-maintained.
struct neuron_group_params_t {

    // structural
    std::size_t neuron_cnt   = 16;
    std::size_t apical_cnt   = 0;   // dendrites per neuron, may be 0
    std::size_t proximal_cnt = 4;   // dendrites per neuron, must be >= 1
    // Max split depth (SPEC.md L; the name is ngm2's and ngm3's too). Decision
    // §13.2: u8 seg ids, depth <= 6
    // (seg_cnt = 2^(depth+1) <= 128, per-dendrite scratch stays L1-resident).
    // Unlike ngm3, memory does NOT scale with this cap — only realized
    // structure is stored.
    uint8_t     max_branch_level = 3;

    // algorithmic parameters — the defaults are ngm2's (hd_ngm2_cfg.h), the
    // names ngm3's flat ones (the two share one config-record field list,
    // ngm/common_config_record.h). ngm2 nests the same settings per dendrite /
    // neuron with a default_ prefix (default_primary_learning_rate, ...), so a
    // generator from an ngm2 config maps names, not just values.
    float             permanence_threshold      = 0.3f;
    float             local_inhibition_strength = 10.0f;
    float             stochastic_win_thres      = 0.8f;
    sigmoid_shape_t   weight_filter             = { 0.5, 0.33 };
    float             common_learning_rate      = 0.0001f;
    float             primary_learning_rate     = 0.01f;
    float             secondary_learning_rate   = 0.0001f;
    float             mismatch_act_thres        = 0.8f;
    float             mismatch_smoothing        = 0.001f;
    std::size_t       branch_interval           = 5000;
    float             min_mismatch_deviation    = 1.0f;
    float             accumulated_theta_thres   = 2.0f;
    float             min_mismatch_percentage   = 0.002f;
    std::pair<float,float> neuron_noise_range   = { 0.01f, 0.05f };

    penalty_policy_t  penalty_policy = penalty_policy_t::stochastic;

    // Frozen (inference-only) mode: process() stops after inhibition; no
    // learning state mutates. With hash draws there is no RNG stream to
    // keep advancing — one class of resume bugs gone (map §2 note). Not
    // configurable: cfg_store::group_or_default sets it from the run's `frozen`.
    bool frozen = false;

    // io + reproducibility
    std::vector<input_route_t> inputs;
    seed_t                     random_seed = {};

    // Named inhibition config; empty synthesizes max_softmax with beta =
    // local_inhibition_strength (same convention as ngm2/ngm3).
    std::string inhibition_cfg;
};

// The configuration record (SPEC.md M.7.8) of a group built from `p`: every
// setting it runs with, its effective seed (random state, not applied on load) and
// its inhibitor's settings. Shared by the CPU and the GPU group (one checkpoint
// schema).
logging::config_record config_record(const neuron_group_params_t&           p,
                                     seed_t                                 effective_seed,
                                     const inhibition::inhibition_params_t& inh);

struct cfg_store {
    std::unordered_map<std::string, neuron_group_params_t>           neuron_groups;
    std::unordered_map<std::string, inhibition::inhibition_params_t> inhibitions;
    cfg_store() { neuron_groups["default"]; inhibitions["default"]; }
    // The named group's parameters; the defaults for a name not configured.
    // The run's top-level `frozen` (loader::global_cfg_t::frozen), set by
    // sim::register_entity_handlers. It is never configured per group: a frozen run
    // keeps all learning off, so every group gets it through group_or_default.
    bool frozen = false;

    // The named group's parameters (the defaults for a name not configured), with the
    // run's `frozen`. An entry that sets `frozen` itself is refused rather than
    // silently overwritten: per-group freezing no longer exists.
    [[nodiscard]] neuron_group_params_t group_or_default(const std::string& name) const {
        const auto it = neuron_groups.find(name);
        neuron_group_params_t p = it != neuron_groups.end() ? it->second : neuron_group_params_t{};
        if (p.frozen)
            throw std::logic_error("ngm25 group '" + name + "': frozen is set per group; "
                                   "set cfg_store::frozen, the run's top-level frozen");
        p.frozen = frozen;
        return p;
    }
};

} // namespace ngm25

#endif // HD_NGM25_CFG_H
