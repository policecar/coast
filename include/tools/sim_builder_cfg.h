//
// Created by jk on 24.01.26.
//

#ifndef SIM_BUILDER_CFG_H
#define SIM_BUILDER_CFG_H
#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <tuple>

#include "sim_core/io_entity.h"

namespace sim {

enum class io_entity_type_t {
    ngm2_neuron_group,
    ngm3_neuron_group,
    ngm3_neuron_group_gpu,
    ngm25_neuron_group,
    ngm25_neuron_group_gpu,
    mnist_input,
    mnist_input_gpu,
    dsprites_input,
    emnist_input,
    kmnist_input,
    cifar10_input,
    cifar100_input,
    chairs3d_input,
    color_mnist_input,
    retina,
    ENTITY_COUNT
};

// What an entity type is, for the code that treats types alike by kind.
enum class entity_kind_t { neuron_group, dataset, retina };

// Everything COAST knows about an entity type, in one row. Adding a type is: an
// enumerator above, a row here (the static_assert below holds the two in step),
// its registration in sim_builder_interface.h, and — for a dataset — its config
// family in dataset::cfg_store (dataset_cfg.h).
struct entity_type_info_t {
    io_entity_type_t type;
    const char*      name;            // config/JSON name
    const char*      display_name;    // shown in every GUI view
    entity_kind_t    kind;
    // Runs on the GPU, so a gpu_context must exist before the entity is
    // constructed (it uploads in its ctor). sim_builder scans the config for these
    // and calls env::enable_gpu() ahead of entity creation.
    bool             gpu;
    // Datasets: the key of its parameter map in dataset::cfg_store, which is also
    // its name in the run manifest. nullptr for other kinds.
    const char*      dataset_family;
};

inline constexpr std::array<entity_type_info_t,
                            static_cast<std::size_t>(io_entity_type_t::ENTITY_COUNT)>
entity_types{{
    { io_entity_type_t::ngm2_neuron_group,      "ngm2_neuron_group",      "Neuron Group",              entity_kind_t::neuron_group, false, nullptr       },
    { io_entity_type_t::ngm3_neuron_group,      "ngm3_neuron_group",      "NGM3 Neuron Group",         entity_kind_t::neuron_group, false, nullptr       },
    { io_entity_type_t::ngm3_neuron_group_gpu,  "ngm3_neuron_group_gpu",  "NGM3 Neuron Group (GPU)",   entity_kind_t::neuron_group, true,  nullptr       },
    { io_entity_type_t::ngm25_neuron_group,     "ngm25_neuron_group",     "NGM2.5 Neuron Group",       entity_kind_t::neuron_group, false, nullptr       },
    { io_entity_type_t::ngm25_neuron_group_gpu, "ngm25_neuron_group_gpu", "NGM2.5 Neuron Group (GPU)", entity_kind_t::neuron_group, true,  nullptr       },
    { io_entity_type_t::mnist_input,            "mnist_input",            "MNIST Input",               entity_kind_t::dataset,      false, "mnist"       },
    { io_entity_type_t::mnist_input_gpu,        "mnist_input_gpu",        "MNIST Input (GPU)",         entity_kind_t::dataset,      true,  "mnist"       },
    { io_entity_type_t::dsprites_input,         "dsprites_input",         "dSprites Input",            entity_kind_t::dataset,      false, "dsprites"    },
    { io_entity_type_t::emnist_input,           "emnist_input",           "EMNIST Input",              entity_kind_t::dataset,      false, "emnist"      },
    { io_entity_type_t::kmnist_input,           "kmnist_input",           "KMNIST Input",              entity_kind_t::dataset,      false, "kmnist"      },
    { io_entity_type_t::cifar10_input,          "cifar10_input",          "CIFAR-10 Input",            entity_kind_t::dataset,      false, "cifar10"     },
    { io_entity_type_t::cifar100_input,         "cifar100_input",         "CIFAR-100 Input",           entity_kind_t::dataset,      false, "cifar100"    },
    { io_entity_type_t::chairs3d_input,         "chairs3d_input",         "3D Chairs Input",           entity_kind_t::dataset,      false, "chairs3d"    },
    { io_entity_type_t::color_mnist_input,      "color_mnist_input",      "Color MNIST Input",         entity_kind_t::dataset,      false, "color_mnist" },
    { io_entity_type_t::retina,                 "retina",                 "Retina",                    entity_kind_t::retina,       false, nullptr       },
}};

static_assert([] {
    for (std::size_t i = 0; i < entity_types.size(); ++i) {
        const auto& t = entity_types[i];
        if (static_cast<std::size_t>(t.type) != i) return false;      // row i is enumerator i
        if ((t.kind == entity_kind_t::dataset) != (t.dataset_family != nullptr)) return false;
    }
    return true;
}(), "entity_types: one row per io_entity_type_t, in enum order; a family exactly for datasets");

// The row of `t`; nullptr for the ENTITY_COUNT sentinel (or any out-of-range value).
[[nodiscard]] constexpr const entity_type_info_t* find_entity_type(io_entity_type_t t)
{
    const auto i = static_cast<std::size_t>(t);
    return i < entity_types.size() ? &entity_types[i] : nullptr;
}

[[nodiscard]] constexpr bool is_gpu_entity_type(io_entity_type_t t)
{
    const auto* info = find_entity_type(t);
    return info != nullptr && info->gpu;
}

// Dataset inputs (carry samples; no neural state). Used by the dataset-only build
// path (dataset_loader) to keep just these nodes out of a full config.
[[nodiscard]] constexpr bool is_dataset_entity_type(io_entity_type_t t)
{
    const auto* info = find_entity_type(t);
    return info != nullptr && info->kind == entity_kind_t::dataset;
}

[[nodiscard]] constexpr bool is_neuron_group_entity_type(io_entity_type_t t)
{
    const auto* info = find_entity_type(t);
    return info != nullptr && info->kind == entity_kind_t::neuron_group;
}

// The human-readable label of an entity type across every GUI view (config
// editor, network editor, …). "Unknown" for the sentinel.
[[nodiscard]] constexpr const char* entity_type_display_name(io_entity_type_t t)
{
    const auto* info = find_entity_type(t);
    return info != nullptr ? info->display_name : "Unknown";
}

using sim_node_id_t = output_id_t;

struct sim_node_t {
    sim_node_id_t                                     id;
    std::string                                       label;
    io_entity_type_t                                  entity_type;
    std::string                                       entity_cfg;
    std::vector<std::pair<sim_node_id_t,std::string>> inputs;
    seed_offset_t                                     random_seed_offset{};
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
