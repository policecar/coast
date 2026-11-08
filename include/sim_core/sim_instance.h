#ifndef COAST_SIM_INSTANCE_H
#define COAST_SIM_INSTANCE_H

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include "json_cfg/json_loader.h"
#include "sim_core/sim_env.h"
#include "tools/sim_builder.h"
#include "sim_core/sim_thread_controller.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "logging/logging_manager.h"

namespace hdf5 { class executor; }
namespace coast::paths { struct app_config; }

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
    /**
     * @param h5_exec  Process-wide HDF5 executor. Must outlive the sim_instance.
     *                 Forwarded to logging_manager and observe_registry::attach
     *                 so every HDF5 call in the simulation routes through the
     *                 single worker thread libhdf5's non-threadsafe build
     *                 requires.
     * @param app_cfg   Application location config. sim_instance resolves its
     *                 OBSERVE output directory from its own already-parsed config
     *                 (observe.output_dir leaf under app_cfg.paths.config_base, or
     *                 an absolute override) and forwards the result to
     *                 observe_registry::attach. Passing app_cfg (rather than a
     *                 pre-resolved path) avoids re-reading the config file to peek
     *                 the leaf and keeps the leaf's field name in one place
     *                 (observe_cfg_t). app_config lives in core/ (below sim_core),
     *                 so this is a normal downward sim_core -> core edge, not a
     *                 layering inversion.
     */
    explicit sim_instance(const std::string&      config_path,
                          hdf5::executor&         h5_exec,
                          const paths::app_config&  app_cfg,
                          seed_offset_t           seed_offset = {});

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

    // The next load resumes this config's own checkpoint (headless --resume, the GUI's
    // "Resume from snapshot"): a config whose run_id is "auto" takes the checkpoint's,
    // so its sequences continue (SPEC.md S.5.14). An explicit run_id keeps its meaning.
    void prepare_resume();

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
