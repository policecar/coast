#ifndef COAST_SIM_INSTANCE_H
#define COAST_SIM_INSTANCE_H

#include <string>
#include <vector>
#include <memory>

#include "json_loader.h"
#include "sim_env.h"
#include "sim_builder.h"
#include "hd_ngm2_neuron_group.h"
#include "vec_ring_buffer.h"
#include "ngm_flat_vis.h"

namespace coast {

/**
 * RAII class that encapsulates all simulation state.
 * Manages the lifecycle of json_loader, sim_env, sim_builder,
 * neuron groups, ring buffers, and visualizations.
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

    [[nodiscard]] std::vector<vec_ring_buffer>& get_ring_buffers() { return ring_buffers; }
    [[nodiscard]] const std::vector<vec_ring_buffer>& get_ring_buffers() const { return ring_buffers; }

    [[nodiscard]] std::vector<ngm_flat_vis>& get_visualizations() { return visualizations; }
    [[nodiscard]] const std::vector<ngm_flat_vis>& get_visualizations() const { return visualizations; }

    [[nodiscard]] loader::global_cfg_t& get_config() { return json_loader->get_config(); }
    [[nodiscard]] const loader::global_cfg_t& get_config() const { return json_loader->get_config(); }

    [[nodiscard]] const sim::sim_builder& get_builder() const { return *sim_builder; }

    [[nodiscard]] const std::string& get_config_path() const { return config_path; }

    /**
     * Free GPU/OpenGL resources for visualizations.
     * Should be called before destruction if OpenGL context is still valid.
     */
    void free_resources();

private:
    std::string config_path;

    // Order matters for construction/destruction
    std::unique_ptr<loader::json_loader> json_loader;
    std::unique_ptr<sim::env> sim_env;
    std::unique_ptr<sim::sim_builder> sim_builder;

    // Points into sim_env - valid as long as sim_env exists
    std::vector<ngm2::neuron_group_t>* groups = nullptr;

    std::vector<vec_ring_buffer> ring_buffers;
    std::vector<ngm_flat_vis> visualizations;

    std::size_t post_process_hook_id = 0;

    void setup_ring_buffers();
    void setup_visualizations();
    void setup_post_process_hook();
};

} // namespace coast

#endif // COAST_SIM_INSTANCE_H
