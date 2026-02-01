//
// Created by claude on 29.01.26.
//

#include "vis_registry.h"
#include "io_buffer.h"

#include <algorithm>

namespace coast::vis {

void vis_registry::register_entity_vis(const vis_info& info, entity_vis_factory factory)
{
    entity_visualizations_.push_back({info, std::move(factory)});
}

std::vector<const vis_info*>
vis_registry::get_entity_visualizations(sim::io_entity_type_t type) const
{
    std::vector<const vis_info*> result;

    for (const auto& reg : entity_visualizations_) {
        const auto& types = reg.info.supported_entity_types;
        if (std::find(types.begin(), types.end(), type) != types.end()) {
            result.push_back(&reg.info);
        }
    }

    return result;
}

std::unique_ptr<vis_instance>
vis_registry::create_entity_vis(const std::string& vis_id,
                                sim::io_entity& entity,
                                sim::env& sim_env,
                                const coast::vis_hints_t& hints,
                                const std::string& hints_key,
                                const std::string& label)
{
    for (const auto& reg : entity_visualizations_) {
        if (reg.info.id == vis_id) {
            return reg.factory(entity, sim_env, hints, hints_key, label);
        }
    }
    return nullptr;
}

// ============================================================================
// IO Buffer Visualizations
// ============================================================================

void vis_registry::register_buffer_vis(const vis_info& info, buffer_vis_factory factory)
{
    buffer_visualizations_.push_back({info, std::move(factory)});
}

std::vector<const vis_info*>
vis_registry::get_buffer_visualizations() const
{
    std::vector<const vis_info*> result;
    result.reserve(buffer_visualizations_.size());

    for (const auto& reg : buffer_visualizations_) {
        result.push_back(&reg.info);
    }

    return result;
}

std::unique_ptr<vis_instance>
vis_registry::create_buffer_vis(const std::string& vis_id,
                                sim::io_buffer& buffer,
                                sim::env& sim_env,
                                const coast::vis_hints_t& hints,
                                const std::string& label)
{
    for (const auto& reg : buffer_visualizations_) {
        if (reg.info.id == vis_id) {
            return reg.factory(buffer, sim_env, hints, label);
        }
    }
    return nullptr;
}

vis_registry& get_vis_registry()
{
    static vis_registry instance;
    return instance;
}

} // namespace coast::vis
