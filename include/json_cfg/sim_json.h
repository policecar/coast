//
// Created by jk on 24.01.26.
//

#ifndef SIM_JSON_H
#define SIM_JSON_H

#include <nlohmann/json.hpp>
#include "sim_builder_cfg.h"

using json = nlohmann::ordered_json;

namespace sim {

NLOHMANN_JSON_SERIALIZE_ENUM( io_entity_type_t, {
    {io_entity_type_t::ngm2_neuron_group,  "ngm2_neuron_group" },
    {io_entity_type_t::mnist_input,        "mnist_input"       },
    {io_entity_type_t::dsprites_input,     "dsprites_input"    },
    {io_entity_type_t::emnist_input,       "emnist_input"      },
    {io_entity_type_t::kmnist_input,       "kmnist_input"      },
    {io_entity_type_t::cifar10_input,      "cifar10_input"     },
    {io_entity_type_t::cifar100_input,     "cifar100_input"    },
    {io_entity_type_t::chairs3d_input,     "chairs3d_input"    },
    {io_entity_type_t::color_mnist_input,  "color_mnist_input" },
    {io_entity_type_t::ENTITY_COUNT,        nullptr            },
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sim_node_t,
    id,
    label,
    entity_type,
    entity_cfg,
    inputs,
    random_seed_offset
)

NLOHMANN_JSON_SERIALIZE_ENUM( sim_node_delta_op, {
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
