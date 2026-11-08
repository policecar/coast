//
// Created by jk on 24.01.26.
//

#ifndef SIM_JSON_H
#define SIM_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"
#include "tools/sim_builder_cfg.h"

using json = nlohmann::ordered_json;

namespace sim {

// Names from the one entity-type table (tools/sim_builder_cfg.h).
COAST_JSON_STRICT_ENUM_RANGE(io_entity_type_t, entity_types, type, name)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sim_node_t,
    id,
    label,
    entity_type,
    entity_cfg,
    inputs,
    random_seed_offset
)

COAST_JSON_STRICT_ENUM( sim_node_delta_op, {
    {sim_node_delta_op::add,    "add"   },
    {sim_node_delta_op::remove, "remove"}
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sim_node_set_t,
    base_id,
    base_label,
    nodes,
    node_deltas
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sim_cfg_t,
    nodes,
    node_sets
)


}

#endif //SIM_JSON_H
