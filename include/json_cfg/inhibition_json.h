//
// Created for the COAST swappable-inhibition feature.
//
// JSON adaptors for inhibition config. Mirrors the dendrite_type_t /
// dendrite_params_t block in ngm2_json.h. Included by both ngm2_json.h and
// ngm3_json.h so each cfg_store's `inhibitions` dict (de)serializes uniformly.
//

#ifndef COAST_INHIBITION_JSON_H
#define COAST_INHIBITION_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"

#include "inhibition/inhibition_cfg.h"

namespace inhibition {

COAST_JSON_STRICT_ENUM(inhibition_type_t, {
    {inhibition_type_t::random_index, "random_index"},
    {inhibition_type_t::max_softmax,  "max_softmax" },
})

COAST_JSON_STRICT_ENUM(random_index_variant_t, {
    {random_index_variant_t::threshold_linear, "threshold_linear"},
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(inhibition_params_t,
    type,
    strength,
    ri_variant,
    ri_reach,
    ri_strength,
    ri_seed,
    ri_seed_from_time
)

}

#endif // COAST_INHIBITION_JSON_H
