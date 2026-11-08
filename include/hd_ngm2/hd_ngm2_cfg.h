//
// Created by jk on 04.09.25.
//

#ifndef HD_NGM2_CFG_H
#define HD_NGM2_CFG_H

#include <string>
#include <stdexcept>
#include <unordered_map>
#include <set>

#include "core/seed.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "inhibition/inhibition_cfg.h"
/*
 * utility function to generate a default parameterization of a cortical neuron group without apical dendrites
 */

namespace ngm2 {

enum class dendrite_type_t {
    apical,          // this type of dendrite modulates the neurons activity (if present)
    proximal,        // main type of dendrite that determines the neuron activity
    TYPE_COUNT
};

// usable default values given here (input_size and input_channels have no "usable" default)
struct dendrite_params_t {
    dendrite_type_t type                        = dendrite_type_t::proximal;
    std::vector<std::string> input_channels     = {};
    double      permanence_threshold            = 0.3;
    uint8_t     max_branch_level                = 3;
    seed_t      rnd_seed                        = {};
    double      default_primary_learning_rate   = 0.01;
    double      default_secondary_learning_rate = 0.0001;
    double      default_mismatch_act_thres      = 0.8;
    double      default_mismatch_smoothing      = 0.001;
    double      default_accumulated_theta_thres = 2.0;
    double      default_min_mismatch_deviation  = 1.0;
    double      default_min_mismatch_percentage = 0.002;
};

using learning_window_t = std::pair<sigmoid_shape_t,sigmoid_shape_t>;

struct dendrite_set_t
{
    std::string name;
    std::string dendrite_cfg;
    std::size_t dendrite_count;
};

struct channel_mapping_t
{
    std::string from_channel;
    std::string to_name;
    std::string to_channel;
};

// usable default values, apart from dendrite params
struct neuron_params_t {
    std::vector<std::string>       input_channels                   = {};
    std::vector<dendrite_set_t>    dendrites                        = {};
    std::vector<channel_mapping_t> channel_mappings                 = {};
    std::size_t                    default_branch_interval          = 5000;
    seed_t                         random_seed                      = {};
    // Inference-time multiplicative-modulation noise added to every neuron's
    // activity. Defaults preserve the original 1%-5% behavior; set to [0,0] in a
    // config for a deterministic, learning-frozen evaluation pass.
    std::pair<double,double>       default_neuron_noise_range       = { 0.01, 0.05 };
};

struct neuron_set_t
{
    std::string name;
    std::string neuron_cfg;
    std::size_t neuron_count;
};

struct neuron_group_params_t {
    std::vector<std::string>       input_channels                    = {};
    std::vector<neuron_set_t>      neurons                           = {};
    std::vector<channel_mapping_t> channel_mappings                  = {};
    double                         default_local_inhibition_strength = 10.0;
    double                         default_common_learning_rate      = 0.0001;
    sigmoid_shape_t                default_weight_filter             = { 0.5, 0.33 };
    double                         default_stochastic_win_thres      = 0.8;
    // Frozen (inference-only) mode: process() skips winner selection,
    // adaptation, and branching — no learning state mutates, only the noise
    // RNG advances. Mirrors ngm3. Not configurable: cfg_store::group_or_default
    // sets it from the run's top-level `frozen`.
    bool                           frozen                            = false;
    seed_t                         random_seed                       = {};
    // Name of an entry in cfg_store::inhibitions. Empty => synthesize a
    // max_softmax with beta = default_local_inhibition_strength.
    std::string                    inhibition_cfg;
};

struct cfg_store {

    std::unordered_map<std::string,dendrite_params_t>     dendrites;
    std::unordered_map<std::string,neuron_params_t>       neurons;
    std::unordered_map<std::string,neuron_group_params_t> neuron_groups;
    std::unordered_map<std::string,inhibition::inhibition_params_t> inhibitions;

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
            throw std::logic_error("ngm2 group '" + name + "': frozen is set per group; "
                                   "set cfg_store::frozen, the run's top-level frozen");
        p.frozen = frozen;
        return p;
    }

    cfg_store()
    {
        dendrites["default"];
        neurons["default"];
        neuron_groups["default"];
        inhibitions["default"];
    }
};

}


#endif //HD_NGM2_CFG_H
