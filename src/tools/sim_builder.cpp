//
// Created by jk on 24.01.26.
//

#include <algorithm>
#include "sim_builder.h"

namespace sim {

sim_builder::sim_builder(env &simulation_environment) :
    sim_env(simulation_environment)
{

}

void sim_builder::register_create_func(
    io_entity_type_t etype,
    std::function<void(const std::string&, output_id_t, env&, int)> func
){
    create_func.emplace(etype,func);
}

void sim_builder::register_init_func(
    io_entity_type_t etype,
    std::function<void(io_entity&, output_id_t, const std::unordered_map<std::string, output_channel_t>&)> func
){
    init_func.emplace(etype,func);
}



void sim_builder::build_local(const sim_cfg_t &cfg)
{
    nodes.reserve(cfg.nodes.size());
    // first, create all direct nodes
    for (const auto &node_cfg : cfg.nodes) {
        node_t new_node;
        new_node.id                 = node_cfg.id;
        new_node.label              = node_cfg.label;
        new_node.inputs             = node_cfg.inputs;
        new_node.entity_type        = node_cfg.entity_type;
        new_node.entity_cfg         = node_cfg.entity_cfg;
        new_node.random_seed_offset = node_cfg.random_seed_offset;

        nodes[new_node.id] = new_node;
    }
    // second, build derived nodes based on node_sets
    for (const auto &set_cfg : cfg.node_sets) {
        for (const auto &node_id : set_cfg.nodes) {
            auto node_it = nodes.find(node_id);
            if (node_it == nodes.end())
                throw std::runtime_error("missing node in nodes set. value: " + std::to_string(node_id));
            const auto &template_node = node_it->second;
            node_t new_node;
            new_node.id          = set_cfg.base_id + template_node.id;
            new_node.label       = set_cfg.base_label + "." + template_node.label;
            new_node.inputs      = template_node.inputs;
                                   for (auto &ip : new_node.inputs)
                                       ip.first += set_cfg.base_id;
            new_node.entity_type = template_node.entity_type;
            new_node.entity_cfg  = template_node.entity_cfg;
            new_node.random_seed_offset = template_node.random_seed_offset;
            // apply deltas if present
            const auto delta_it = set_cfg.node_deltas.find(template_node.id);
            if (delta_it != set_cfg.node_deltas.cend()) {
                for (const auto& [op,nid,chan] : delta_it->second)
                    switch (op) {
                        case sim_node_delta_op::add : {
                            new_node.inputs.push_back({nid,chan});
                        } break;
                        case sim_node_delta_op::remove : {
                            std::erase_if(
                                new_node.inputs,
                                [nid,&chan,base_id = set_cfg.base_id](const auto &elem)
                                {
                                    return (elem.first == nid + base_id) && (elem.second == chan);
                                }
                            );
                        } break;
                    }
            }
            nodes[new_node.id] = new_node;
        }
    }
    // third, go through all created nodes and create the actual simulation entities
    // (future work: in a non-local version of this build function we could parition the simulation
    // graph and distribute it over multiple machines)
    for (auto &node : nodes) {
        auto func_it = create_func.find(node.second.entity_type);
        if (func_it == create_func.end())
            throw std::runtime_error("No create function for entity type. cfg:" + node.second.entity_cfg);
        func_it->second(node.second.entity_cfg,node.first,sim_env,node.second.random_seed_offset);
    }
    // fourth, init io of each entity
    for (auto &entity : sim_env.iterate_entities()) {
        // get node
        output_id_t node_output_id = entity.get_outp_id();
        auto node_it = nodes.find(node_output_id);
        if (node_it == nodes.end())
            throw std::runtime_error("simulation environment contains entity with unknown ID: " + std::to_string(node_output_id));
        node_t &cur_node = node_it->second;
        // create input map
        std::unordered_map<std::string, output_channel_t> input_map;
        for (const auto &inp : cur_node.inputs) {
            auto input_entity = sim_env.get_entity(inp.first);
            if (!input_entity)
                throw std::runtime_error("node requires input id that does not exist. id:" + std::to_string(inp.first));
            input_map[inp.second] = { inp.first, input_entity->get_outp_size() };
        }
        auto func_it = init_func.find(cur_node.entity_type);
        if (func_it == init_func.end())
            throw std::runtime_error("No init function for entity type. cfg:" + cur_node.entity_cfg);
        func_it->second(entity,entity.get_outp_id(),input_map);
    }
    // fifth, init io buffers of simulation environment
    sim_env.init_io_buffers();

}

const sim_builder::node_t* sim_builder::get_node(sim_node_id_t id) const
{
    auto it = nodes.find(id);
    if (it == nodes.end())
        return nullptr;
    return &it->second;
}

}


