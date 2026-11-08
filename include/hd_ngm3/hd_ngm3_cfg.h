//
// Created by jk on 25.05.26.
//

#ifndef HD_NGM3_CFG_H
#define HD_NGM3_CFG_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/seed.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "inhibition/inhibition_cfg.h"
#include "ngm/bank.h"

namespace ngm3 {

using sigmoid_shape_t   = ngm2::sigmoid_shape_t;
using learning_window_t = std::pair<sigmoid_shape_t, sigmoid_shape_t>;

using ngm::bank_t;          // ngm/bank.h, shared with ngm2.5
using ngm::input_route_t;

struct neuron_group_params_t {

    // structural
    std::size_t neuron_cnt             = 16;
    std::size_t apical_cnt             = 0;        // dendrites per neuron, may be 0
    std::size_t proximal_cnt           = 4;        // dendrites per neuron, must be >= 1
    uint8_t     max_branch_level       = 3;        // rep_cnt = 2^depth, max_seg_id = 2^(depth+1)-1

    // algorithmic floats
    float            permanence_threshold      = 0.3f;
    float            local_inhibition_strength = 10.0f;
    float            stochastic_win_thres      = 0.8f;
    sigmoid_shape_t  weight_filter             = { 0.5, 0.33 };
    float            common_learning_rate      = 0.0001f;
    float            primary_learning_rate     = 0.01f;
    float            secondary_learning_rate   = 0.0001f;
    float            mismatch_act_thres        = 0.8f;
    float            mismatch_smoothing        = 0.001f;
    std::size_t      branch_interval           = 5000;
    float            min_mismatch_deviation    = 1.0f;
    float            accumulated_theta_thres   = 2.0f;
    float            min_mismatch_percentage   = 0.002f;
    // Inference-time combine noise added to every neuron's activity
    // (neuron_apical_max * neuron_proximal_max + noise). Default preserves the
    // original 1%-5%; set to [0,0] for a deterministic, learning-frozen eval pass.
    // Mirrors ngm2 neuron_params_t::default_neuron_noise_range (a78e0d6).
    std::pair<float,float> neuron_noise_range  = { 0.01f, 0.05f };

    // Frozen (inference-only) mode: compute_step skips winner selection,
    // adaptation, branching, and the neuron_mx_act running max — no learning
    // state mutates, only the noise RNG advances. Mirrors ngm2. Not configurable:
    // cfg_store::group_or_default sets it from the run's top-level `frozen`.
    bool frozen = false;

    // io
    std::vector<input_route_t> inputs;
    seed_t                     random_seed   = {};

    // Name of an entry in cfg_store::inhibitions. Empty => synthesize a
    // max_softmax with beta = local_inhibition_strength.
    std::string                inhibition_cfg;
};

// The configuration record (SPEC.md M.7.8) of a group built from `p`: every
// setting it runs with, its effective seed (random state, not applied on load) and
// its inhibitor's settings. One function for the CPU and the GPU group, so a
// checkpoint is compared the same way whichever of the two wrote or loads it.
logging::config_record config_record(const neuron_group_params_t&         p,
                                     seed_t                               effective_seed,
                                     const inhibition::inhibition_params_t& inh);

struct cfg_store {

    std::unordered_map<std::string, neuron_group_params_t> neuron_groups;
    std::unordered_map<std::string, inhibition::inhibition_params_t> inhibitions;

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
            throw std::logic_error("ngm3 group '" + name + "': frozen is set per group; "
                                   "set cfg_store::frozen, the run's top-level frozen");
        p.frozen = frozen;
        return p;
    }

    cfg_store()
    {
        neuron_groups["default"];
        inhibitions["default"];
    }
};

} // namespace ngm3

#endif // HD_NGM3_CFG_H
