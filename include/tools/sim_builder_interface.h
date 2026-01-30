//
// Created by jk on 26.01.26.
//

#ifndef SIM_BUILDER_INTERFACE_H
#define SIM_BUILDER_INTERFACE_H

#include "sim_builder.h"
#include "json_loader.h"

#include "hd_ngm2_neuron_group.h"
#include "mnist_io.h"
#include "dsprites_io.h"
#include "emnist_io.h"
#include "kmnist_io.h"
#include "cifar10_io.h"
#include "cifar100_io.h"
#include "chairs3d_io.h"

namespace sim {

inline void register_entity_handlers(sim_builder &sb, loader::json_loader &jl)
{
    // create function for neuron groups
    sb.register_create_func( sim::io_entity_type_t::ngm2_neuron_group,
        [cfg_store = jl.get_ngm2_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            sim_env.emplace_back<ngm2::neuron_group_t>(cfg_store,cfg_type,out_id);
        }
    );
    // init function for neuron groups
    sb.register_init_func( sim::io_entity_type_t::ngm2_neuron_group,
        [](io_entity &entity, sim::output_id_t,   const std::unordered_map<std::string, output_channel_t> &input_map)
        {
            auto &ng = dynamic_cast<ngm2::neuron_group_t&>(entity);
            ng.init_io(input_map);
        }
    );

    // create function for mnist datasets
    sb.register_create_func( sim::io_entity_type_t::mnist_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.mnist.find(cfg_type);
            if (it == cfg_store.mnist.end())
                throw std::runtime_error("Unknown mnist configuration: " + cfg_type);
            mnist::mnist_param_t params = it->second;
            sim_env.emplace_back<mnist::mnist_io>(params,out_id);
        }
    );
    // init function for mnist datasets
    sb.register_init_func( sim::io_entity_type_t::mnist_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for mnist
        }
    );

    // create function for dpsrites datasets
    sb.register_create_func( sim::io_entity_type_t::dsprites_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.dsprites.find(cfg_type);
            if (it == cfg_store.dsprites.end())
                throw std::runtime_error("Unknown dsprites configuration: " + cfg_type);
            dsprites::dsprites_param_t params = it->second;
            sim_env.emplace_back<dsprites::dsprites_io>(params,out_id);
        }
    );
    // init function for dsprites datasets
    sb.register_init_func( sim::io_entity_type_t::dsprites_input,
        [](io_entity&, sim::output_id_t,   const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for dsprites
        }
    );

    // create function for emnist datasets
    sb.register_create_func( sim::io_entity_type_t::emnist_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.emnist.find(cfg_type);
            if (it == cfg_store.emnist.end())
                throw std::runtime_error("Unknown emnist configuration: " + cfg_type);
            emnist::emnist_param_t params = it->second;
            sim_env.emplace_back<emnist::emnist_io>(params,out_id);
        }
    );
    // init function for emnist datasets
    sb.register_init_func( sim::io_entity_type_t::emnist_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for emnist
        }
    );

    // create function for kmnist datasets
    sb.register_create_func( sim::io_entity_type_t::kmnist_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.kmnist.find(cfg_type);
            if (it == cfg_store.kmnist.end())
                throw std::runtime_error("Unknown kmnist configuration: " + cfg_type);
            kmnist::kmnist_param_t params = it->second;
            sim_env.emplace_back<kmnist::kmnist_io>(params,out_id);
        }
    );
    // init function for kmnist datasets
    sb.register_init_func( sim::io_entity_type_t::kmnist_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for kmnist
        }
    );

    // create function for cifar10 datasets
    sb.register_create_func( sim::io_entity_type_t::cifar10_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.cifar10.find(cfg_type);
            if (it == cfg_store.cifar10.end())
                throw std::runtime_error("Unknown cifar10 configuration: " + cfg_type);
            cifar10::cifar10_param_t params = it->second;
            sim_env.emplace_back<cifar10::cifar10_io>(params,out_id);
        }
    );
    // init function for cifar10 datasets
    sb.register_init_func( sim::io_entity_type_t::cifar10_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for cifar10
        }
    );

    // create function for cifar100 datasets
    sb.register_create_func( sim::io_entity_type_t::cifar100_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.cifar100.find(cfg_type);
            if (it == cfg_store.cifar100.end())
                throw std::runtime_error("Unknown cifar100 configuration: " + cfg_type);
            cifar100::cifar100_param_t params = it->second;
            sim_env.emplace_back<cifar100::cifar100_io>(params,out_id);
        }
    );
    // init function for cifar100 datasets
    sb.register_init_func( sim::io_entity_type_t::cifar100_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for cifar100
        }
    );

    // create function for chairs3d datasets
    sb.register_create_func( sim::io_entity_type_t::chairs3d_input,
        [cfg_store = jl.get_datasets_cfg()](const std::string &cfg_type, sim::output_id_t out_id, sim::env &sim_env)
        {
            const auto it = cfg_store.chairs3d.find(cfg_type);
            if (it == cfg_store.chairs3d.end())
                throw std::runtime_error("Unknown chairs3d configuration: " + cfg_type);
            chairs3d::chairs3d_param_t params = it->second;
            sim_env.emplace_back<chairs3d::chairs3d_io>(params,out_id);
        }
    );
    // init function for chairs3d datasets
    sb.register_init_func( sim::io_entity_type_t::chairs3d_input,
        [](io_entity&, sim::output_id_t, const std::unordered_map<std::string, output_channel_t>&)
        {
            // nothing to init for chairs3d
        }
    );

}


}


#endif //SIM_BUILDER_INTERFACE_H
