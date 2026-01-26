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

}


}


#endif //SIM_BUILDER_INTERFACE_H
