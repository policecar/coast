#include "sim_instance.h"
#include "sim_builder_interface.h"

#include <stdexcept>

namespace coast {

sim_instance::sim_instance(const std::string& config_path, int seed_offset)
    : config_path(config_path)
{
    // Load configuration
    json_loader = std::make_unique<loader::json_loader>(config_path);

    // Set up simulation environment
    sim_env = std::make_unique<sim::env>();

    // Create simulation builder and register entity handlers
    sim_builder = std::make_unique<sim::sim_builder>(*sim_env);
    sim::register_entity_handlers(*sim_builder, *json_loader, seed_offset);

    // Build the simulation from config
    sim_builder->build_local(json_loader->get_sim_cfg());

    // Get reference to neuron groups
    auto groups_opt = sim_env->get_entities<ngm2::neuron_group_t>();
    if (!groups_opt || groups_opt.value()->empty()) {
        throw std::runtime_error("No neuron groups found in simulation environment");
    }
    groups = groups_opt.value();

    // Create simulation thread controller (starts in paused mode)
    thread_controller = std::make_unique<sim::sim_thread_controller>(*sim_env);

    // Create logging manager
    logging_manager = std::make_unique<logging::logging_manager>(*sim_env);
}

sim_instance::~sim_instance() = default;

sim_instance::sim_instance(sim_instance&& other) noexcept
    : config_path(std::move(other.config_path))
    , json_loader(std::move(other.json_loader))
    , sim_env(std::move(other.sim_env))
    , sim_builder(std::move(other.sim_builder))
    , groups(other.groups)
    , thread_controller(std::move(other.thread_controller))
    , logging_manager(std::move(other.logging_manager))
{
    other.groups = nullptr;
}

sim_instance& sim_instance::operator=(sim_instance&& other) noexcept
{
    if (this != &other) {
        // Stop logging and thread controller first before moving resources
        logging_manager.reset();
        thread_controller.reset();

        config_path = std::move(other.config_path);
        json_loader = std::move(other.json_loader);
        sim_env = std::move(other.sim_env);
        sim_builder = std::move(other.sim_builder);
        groups = other.groups;
        thread_controller = std::move(other.thread_controller);
        logging_manager = std::move(other.logging_manager);

        other.groups = nullptr;
    }
    return *this;
}

} // namespace coast
