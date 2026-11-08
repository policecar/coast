//
// Created by jk on 26.01.26.
//

#ifndef SIM_BUILDER_INTERFACE_H
#define SIM_BUILDER_INTERFACE_H

#include "tools/sim_builder.h"
#include "json_cfg/json_loader.h"
#include "json_cfg/dataset_json.h"   // a dataset's params as its identity record

#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "io/mnist_io.h"
#ifdef COAST_WITH_GPU
#include "io/mnist_io_gpu.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#endif
#include "io/dsprites_io.h"
#include "io/cifar10_io.h"
#include "io/cifar100_io.h"
#include "io/chairs3d_io.h"
#include "io/color_mnist_io.h"
#include "retina/retina_io.h"

#include "hdf5/executor.h"

namespace sim {

namespace detail {

using input_map_t = std::unordered_map<std::string, output_channel_t>;

// ---- create: how an entity is built from its node ----

// A neuron group: from its model's whole cfg_store (the group looks its config
// up by name itself, and gets the run's frozen from it: cfg_store::frozen).
template <class Entity, class CfgStore>
void register_group(sim_builder &sb, io_entity_type_t type, CfgStore cfg_store,
                    seed_offset_t seed_offset, bool frozen)
{
    cfg_store.frozen = frozen;
    sb.register_create_func(type,
        [cfg_store = std::move(cfg_store), seed_offset]
        (const std::string &cfg_type, output_id_t out_id, env &sim_env, seed_offset_t so)
        {
            sim_env.emplace_back<Entity>(cfg_store, cfg_type, out_id, seed_offset + so);
        });
}

// A dataset or retina: from its config entry, looked up by name in one family's
// map, its seed folded with the run's and the node's offsets. A dataset also gets
// its resolved params as its identity record (dataset::run_binding_t). `extra` are
// further constructor arguments after (params, out_id). The family's name (for
// the error) is the type's dataset family, or its config name (entity_types).
template <class Entity, class Map, class... Extra>
void register_params(sim_builder &sb, io_entity_type_t type, Map configs,
                     seed_offset_t seed_offset, Extra&... extra)
{
    const auto *info = find_entity_type(type);
    std::string family = info->dataset_family ? info->dataset_family : info->name;
    sb.register_create_func(type,
        [configs = std::move(configs), family = std::move(family), seed_offset, &extra...]
        (const std::string &cfg_type, output_id_t out_id, env &sim_env, seed_offset_t so)
        {
            const auto it = configs.find(cfg_type);
            if (it == configs.end())
                throw std::runtime_error("Unknown " + family + " configuration: " + cfg_type);
            auto params = it->second;
            params.random_seed = params.random_seed + seed_offset + so;
            if constexpr (requires { params.run; }) {
                // The seed is random state, not identity: a checkpoint's RNGs win on
                // restore, so a resume under another seed offset is the same data.
                auto identity = nlohmann::ordered_json(params);
                identity.erase("random_seed");
                params.run = { .identity = identity.dump() };
            }
            sim_env.emplace_back<Entity>(params, out_id, extra...);
        });
}

// ---- init: how a built entity is wired ----

// Routed inputs: the entity's own init_io (which builds its structure registry).
template <class Entity>
void register_init_io(sim_builder &sb, io_entity_type_t type)
{
    sb.register_init_func(type, [](io_entity &entity, output_id_t, const input_map_t &input_map) {
        dynamic_cast<Entity&>(entity).init_io(input_map);
    });
}

// No inputs (a dataset): only the structure registry.
inline void register_init_structure(sim_builder &sb, io_entity_type_t type)
{
    sb.register_init_func(type, [](io_entity &entity, output_id_t, const input_map_t &) {
        entity.init_structure();
    });
}

template <class Entity, class CfgStore>
void register_routed_group(sim_builder &sb, io_entity_type_t type, CfgStore cfg_store,
                           seed_offset_t seed_offset, bool frozen)
{
    register_group<Entity>(sb, type, std::move(cfg_store), seed_offset, frozen);
    register_init_io<Entity>(sb, type);
}

template <class Entity, class Map, class... Extra>
void register_dataset(sim_builder &sb, io_entity_type_t type, Map configs,
                      seed_offset_t seed_offset, Extra&... extra)
{
    register_params<Entity>(sb, type, std::move(configs), seed_offset, extra...);
    register_init_structure(sb, type);
}

} // namespace detail

// Every entity type's create and init function, one line per row of
// entity_types (tools/sim_builder_cfg.h). A type missing here fails only when a
// config uses it (sim_builder::build_local); the unit test
// unit.tools.entity_registration checks that every enumerator is handled.
inline void register_entity_handlers(sim_builder &sb, loader::json_loader &jl,
                                     hdf5::executor &h5_exec, seed_offset_t seed_offset = {})
{
    using namespace detail;
    using T = io_entity_type_t;
    const auto &ds = jl.get_datasets_cfg();
    // The run's frozen reaches every learning entity (SPEC M.7.6).
    const bool frozen = jl.get_config().frozen;

    // Neuron groups. A GPU type reuses its CPU model's config block; the
    // gpu_context was installed by sim_builder::build_local before these run.
    register_routed_group<ngm2::neuron_group_t> (sb, T::ngm2_neuron_group,  jl.get_ngm2_cfg(),  seed_offset, frozen);
    register_routed_group<ngm3::neuron_group_t> (sb, T::ngm3_neuron_group,  jl.get_ngm3_cfg(),  seed_offset, frozen);
    register_routed_group<ngm25::neuron_group_t>(sb, T::ngm25_neuron_group, jl.get_ngm25_cfg(), seed_offset, frozen);
#ifdef COAST_WITH_GPU
    register_routed_group<ngm3::neuron_group_gpu_t> (sb, T::ngm3_neuron_group_gpu,  jl.get_ngm3_cfg(),  seed_offset, frozen);
    register_routed_group<ngm25::neuron_group_gpu_t>(sb, T::ngm25_neuron_group_gpu, jl.get_ngm25_cfg(), seed_offset, frozen);
#endif

    // Retina nodes, one polarity each: init_io binds and size-checks the single
    // input (retina_io.h).
    register_params<retina::retina_io>(sb, T::retina, jl.get_retina_cfg(), seed_offset);
    register_init_io<retina::retina_io>(sb, T::retina);

    // Dataset inputs (a GPU type reuses its CPU family's config block).
    register_dataset<mnist::mnist_io>            (sb, T::mnist_input,       ds.mnist,       seed_offset);
#ifdef COAST_WITH_GPU
    register_dataset<mnist::mnist_io_gpu>        (sb, T::mnist_input_gpu,   ds.mnist,       seed_offset);
#endif
    register_dataset<dsprites::dsprites_io>      (sb, T::dsprites_input,    ds.dsprites,    seed_offset, h5_exec);
    register_dataset<emnist::emnist_io>          (sb, T::emnist_input,      ds.emnist,      seed_offset);
    register_dataset<kmnist::kmnist_io>          (sb, T::kmnist_input,      ds.kmnist,      seed_offset);
    register_dataset<cifar10::cifar10_io>        (sb, T::cifar10_input,     ds.cifar10,     seed_offset);
    register_dataset<cifar100::cifar100_io>      (sb, T::cifar100_input,    ds.cifar100,    seed_offset);
    register_dataset<chairs3d::chairs3d_io>      (sb, T::chairs3d_input,    ds.chairs3d,    seed_offset);
    register_dataset<color_mnist::color_mnist_io>(sb, T::color_mnist_input, ds.color_mnist, seed_offset);
}

}


#endif //SIM_BUILDER_INTERFACE_H
