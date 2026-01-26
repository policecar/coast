//
// Created by jk on 22.01.26.
//

#ifndef NGM2_JSON_H
#define NGM2_JSON_H

#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include "hd_ngm2_cfg.h"

using json = nlohmann::json;

namespace ngm2 {

// json adaptors for dendrite parameters
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sigmoid_shape_t,
    steepness,
    transition_point
)

NLOHMANN_JSON_SERIALIZE_ENUM( dendrite_type_t, {
    {dendrite_type_t::apical,    "apical"  },
    {dendrite_type_t::proximal,  "proximal"},
    {dendrite_type_t::TYPE_COUNT, nullptr  },
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dendrite_params_t,
    type,
    input_channels,
    permanence_threshold,
    max_branch_level,
    rnd_seed,
    default_primary_learning_rate,
    default_secondary_learning_rate,
    default_mismatch_act_thres,
    default_mismatch_smoothing,
    default_accumulated_theta_thres,
    default_min_mismatch_deviation,
    default_min_mismatch_percentage
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(dendrite_set_t,
    name,
    dendrite_cfg,
    dendrite_count
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(channel_mapping_t,
    from_channel,
    to_name,
    to_channel
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(neuron_params_t,
    input_channels,
    dendrites,
    channel_mappings,
    default_branch_interval,
    default_activity_learning_window,
    random_seed
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(neuron_set_t,
    name,
    neuron_cfg,
    neuron_count
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(neuron_group_params_t,
    input_channels,
    neurons,
    channel_mappings,
    default_local_inhibition_strength,
    default_common_learning_rate,
    default_weight_filter,
    default_stochastic_win_thres,
    random_seed
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    dendrites,
    neurons,
    neuron_groups
)


}

#endif //NGM2_JSON_H
