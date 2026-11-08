//
// Created 10.07.26 — ngm2.5 increment I6: JSON adapters for the flat schema.
// Pure nlohmann adapter header (mirrors ngm3_json.h). The "ngm25" top-level
// config key binds to ngm25::cfg_store via json_loader.h.
//

#ifndef NGM25_JSON_H
#define NGM25_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"
#include "json_cfg/ngm_bank_json.h"   // bank_t, input_route_t (shared)
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "json_cfg/ngm2_json.h" // brings the shared sigmoid_shape_t adapter

using json = nlohmann::ordered_json;

namespace ngm25 {

// New enum vs ngm3 — the penalty-draw research seam (§8).
COAST_JSON_STRICT_ENUM( penalty_policy_t, {
    {penalty_policy_t::stochastic,     "stochastic"    },
    {penalty_policy_t::expected_value, "expected_value"},
    {penalty_policy_t::none,           "none"          },
})

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
    penalty_policy,
    inputs,
    random_seed,
    inhibition_cfg
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    neuron_groups,
    inhibitions
)

} // namespace ngm25

#endif // NGM25_JSON_H
