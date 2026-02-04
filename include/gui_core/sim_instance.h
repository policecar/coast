#ifndef COAST_SIM_INSTANCE_H
#define COAST_SIM_INSTANCE_H

#include <string>
#include <vector>
#include <memory>

#include "json_loader.h"
#include "sim_env.h"
#include "sim_builder.h"
#include "sim_thread_controller.h"
#include "hd_ngm2_neuron_group.h"
#include "logging/logging_manager.h"

namespace coast {

/**
 * RAII class that encapsulates core simulation state.
 * Manages the lifecycle of json_loader, sim_env, sim_builder,
 * and neuron groups.
 *
 * Construction loads and initializes the simulation from a config file.
 * Destruction properly cleans up all resources.
 */
class sim_instance {
public:
    /**
     * Load and initialize a simulation from a configuration file.
     * @param config_path Path to JSON configuration file
     * @throws std::runtime_error if loading or initialization fails
     */
    explicit sim_instance(const std::string& config_path);

    ~sim_instance();

    // Non-copyable
    sim_instance(const sim_instance&) = delete;
    sim_instance& operator=(const sim_instance&) = delete;

    // Movable
    sim_instance(sim_instance&&) noexcept;
    sim_instance& operator=(sim_instance&&) noexcept;

    // Accessors
    [[nodiscard]] sim::env& get_sim_env() { return *sim_env; }
    [[nodiscard]] const sim::env& get_sim_env() const { return *sim_env; }

    [[nodiscard]] std::vector<ngm2::neuron_group_t>& get_groups() { return *groups; }
    [[nodiscard]] const std::vector<ngm2::neuron_group_t>& get_groups() const { return *groups; }

    [[nodiscard]] loader::global_cfg_t& get_config() { return json_loader->get_config(); }
    [[nodiscard]] const loader::global_cfg_t& get_config() const { return json_loader->get_config(); }

    [[nodiscard]] const sim::sim_builder& get_builder() const { return *sim_builder; }

    [[nodiscard]] const std::string& get_config_path() const { return config_path; }

    /**
     * Get the simulation thread controller.
     * Returns nullptr if simulation is not properly initialized.
     */
    [[nodiscard]] sim::sim_thread_controller* get_thread_controller() { return thread_controller.get(); }
    [[nodiscard]] const sim::sim_thread_controller* get_thread_controller() const { return thread_controller.get(); }

    /**
     * Get the logging manager for this simulation.
     */
    [[nodiscard]] logging::logging_manager& get_logging_manager() { return *logging_manager; }
    [[nodiscard]] const logging::logging_manager& get_logging_manager() const { return *logging_manager; }

private:
    std::string config_path;

    // Order matters for construction/destruction
    std::unique_ptr<loader::json_loader> json_loader;
    std::unique_ptr<sim::env> sim_env;
    std::unique_ptr<sim::sim_builder> sim_builder;

    // Points into sim_env - valid as long as sim_env exists
    std::vector<ngm2::neuron_group_t>* groups = nullptr;

    // Simulation thread controller - created after sim_env is ready
    std::unique_ptr<sim::sim_thread_controller> thread_controller;

    // Logging manager - created after sim_env is ready
    std::unique_ptr<logging::logging_manager> logging_manager;
};

} // namespace coast

#endif // COAST_SIM_INSTANCE_H
