//
// Created by jk on 25.05.26.
//

#ifndef NGM3_JSON_H
#define NGM3_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"
#include "json_cfg/ngm_bank_json.h"   // bank_t, input_route_t (shared)
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "json_cfg/ngm2_json.h" // brings sigmoid_shape_t adapter (shared with ngm2)

using json = nlohmann::ordered_json;

namespace ngm3 {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(neuron_group_params_t,
    neuron_cnt,
    apical_cnt,
    proximal_cnt,
    max_branch_level,
    permanence_threshold,
    local_inhibition_strength,
    stochastic_win_thres,
    weight_filter,
    common_learning_rate,
    primary_learning_rate,
    secondary_learning_rate,
    mismatch_act_thres,
    mismatch_smoothing,
    branch_interval,
    min_mismatch_deviation,
    accumulated_theta_thres,
    min_mismatch_percentage,
    neuron_noise_range,
    inputs,
    random_seed,
    inhibition_cfg
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    neuron_groups,
    inhibitions
)

} // namespace ngm3

#endif // NGM3_JSON_H
