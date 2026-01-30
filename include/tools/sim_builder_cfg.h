//
// Created by jk on 24.01.26.
//

#ifndef SIM_BUILDER_CFG_H
#define SIM_BUILDER_CFG_H
#include <string>
#include <vector>
#include <tuple>

#include "io_entity.h"

namespace sim {

enum class io_entity_type_t {
    ngm2_neuron_group,
    mnist_input,
    dsprites_input,
    emnist_input,
    kmnist_input,
    cifar10_input,
    cifar100_input,
    chairs3d_input,
    color_mnist_input,
    ENTITY_COUNT
};

using sim_node_id_t = output_id_t;

struct sim_node_t {
    sim_node_id_t                                     id;
    std::string                                       label;
    io_entity_type_t                                  entity_type;
    std::string                                       entity_cfg;
    std::vector<std::pair<sim_node_id_t,std::string>> inputs;
};

enum class sim_node_delta_op
{
    add,
    remove
};

using sim_node_delta_t = std::vector< std::tuple< sim_node_delta_op, sim_node_id_t, std::string >>;

struct sim_node_set_t {
    sim_node_id_t                 base_id;
    std::string                   base_label;
    std::vector<sim_node_id_t>    nodes;
    std::unordered_map<sim_node_id_t,sim_node_delta_t> node_deltas;
};

struct sim_cfg_t {

    std::vector<sim_node_t>     nodes;
    std::vector<sim_node_set_t> node_sets;

};


}


#endif //SIM_BUILDER_CFG_H
