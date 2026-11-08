//
// Created by jk on 27.01.26.
//

#ifndef COAST_VIS_HINTS_JSON_H
#define COAST_VIS_HINTS_JSON_H

#include <nlohmann/json.hpp>

#include "core/vis_hints.h"

using json = nlohmann::ordered_json;

namespace coast {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm_flat_vis_hint_t,
    width,
    height,
    num_channels,
    update_interval
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm_tree_vis_hint_t,
    rep_width,
    rep_height,
    num_channels,
    update_interval,
    bg_r, bg_g, bg_b, bg_a,
    edge_thickness_min,
    edge_thickness_max,
    edge_r, edge_g, edge_b, edge_a,
    apical_edge_r, apical_edge_g, apical_edge_b, apical_edge_a,
    texture_scale_divisor,
    root_node_radius,
    branch_node_radius,
    root_r, root_g, root_b, root_a,
    branch_r, branch_g, branch_b, branch_a
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(buffer_vis_hint_t,
    update_interval,
    ring_length
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(mnist_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dsprites_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(emnist_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(kmnist_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar10_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar100_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(chairs3d_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(color_mnist_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dataset_hints_t,
    mnist,
    dsprites,
    emnist,
    kmnist,
    cifar10,
    cifar100,
    chairs3d,
    color_mnist
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(neuron_group_hints_t,
    ngm_flat_vis,
    ngm_tree_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm2_hints_t,
    neuron_groups
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm3_hints_t,
    neuron_groups
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(vis_hints_t,
    datasets,
    ngm2,
    ngm3,
    buffer_vis
)


}

#endif //COAST_VIS_HINTS_JSON_H
