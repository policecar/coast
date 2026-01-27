//
// Created by jk on 24.01.26.
//

#ifndef SIM_BUILDER_H
#define SIM_BUILDER_H

#include <unordered_map>
#include <string>

#include "sim_env.h"
#include "sim_builder_cfg.h"

namespace sim {

class sim_builder
{

public:
    struct node_t
    {
        sim_node_id_t    id;
        std::string      label;
        io_entity_type_t entity_type;
        std::string      entity_cfg;
        std::vector<std::pair<sim_node_id_t,std::string>> inputs;
    };

private:
    env &sim_env;
    std::unordered_map<sim_node_id_t,node_t> nodes;

    std::unordered_map<
        io_entity_type_t,
        std::function<void(const std::string&, output_id_t, env&)>
    > create_func;

    std::unordered_map<
        io_entity_type_t,
        std::function<void(io_entity&, output_id_t, const std::unordered_map<std::string, output_channel_t>&)>
    > init_func;

public:

    explicit sim_builder(env &simulation_environment);

    void register_create_func(
        io_entity_type_t etype,
        std::function<void(const std::string&, output_id_t, env&)> func
    );

    void register_init_func(
        io_entity_type_t etype,
        std::function<void(io_entity&, output_id_t, const std::unordered_map<std::string, output_channel_t>&)> func
    );

    void build_local(const sim_cfg_t &cfg);

    /**
     * Get node information by ID.
     * @param id The node/output ID to look up
     * @return Pointer to node_t if found, nullptr otherwise
     */
    [[nodiscard]] const node_t* get_node(sim_node_id_t id) const;
};

}


#endif //SIM_BUILDER_H
