//
// Created by jk on 22.01.26.
//

#ifndef NGM2_JSON_H
#define NGM2_JSON_H

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "json_cfg/inhibition_json.h"

using json = nlohmann::ordered_json;

namespace ngm2 {

// sigmoid_shape_t is shared by ngm2, ngm25 and ngm3 (CPU and GPU), so this one
// adapter is where a shape is checked. sigmoid() takes inv_step = 1/(1 - s^0.1):
// s == 1 gives inf and turns every output into NaN, s > 1 inverts the curve,
// s < 0 is NaN. s == 0 is well-defined (inv_step = 1). The formula runs in float,
// where pow(s, 0.1f) already rounds to 1 for s >= 0.9999999, so the check is on the
// computed inv_step, not on s < 1 alone.
template <typename BasicJsonType>
void to_json(BasicJsonType& j, const sigmoid_shape_t& s)
{
    j = BasicJsonType{ { "steepness", s.steepness }, { "transition_point", s.transition_point } };
}

template <typename BasicJsonType>
void from_json(const BasicJsonType& j, sigmoid_shape_t& s)
{
    const sigmoid_shape_t def{};
    if (j.is_null()) { s = def; return; }   // as the _WITH_DEFAULT macros: null means defaults
    const double steep = j.value("steepness",        def.steepness);
    const double trans = j.value("transition_point", def.transition_point);
    if (!(steep >= 0.0 && steep < 1.0 && std::isfinite(sigmoid_params_t(sigmoid_shape_t{ steep, 0.0 }).inv_step_size)))
        throw std::invalid_argument(std::format("sigmoid steepness must be in [0, 1) and give a finite "
                                                "1/(1 - steepness^0.1) in float (s up to about 0.9999997), got {}", steep));
    if (!std::isfinite(trans))
        throw std::invalid_argument(std::format("sigmoid transition_point must be finite, got {}", trans));
    s = { steep, trans };
}

COAST_JSON_STRICT_ENUM( dendrite_type_t, {
    {dendrite_type_t::apical,    "apical"  },
    {dendrite_type_t::proximal,  "proximal"},
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
    random_seed,
    default_neuron_noise_range
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
    random_seed,
    inhibition_cfg
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    dendrites,
    neurons,
    neuron_groups,
    inhibitions
)


}

#endif //NGM2_JSON_H
