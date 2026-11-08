#include "sim_core/sim_instance.h"
#include "tools/sim_builder_interface.h"
#include "observe/observe_registry.h"
#include "hdf5/executor.h"
#include "core/app_config.h"
#include "core/path_resolver.h"
#include "core/run_id.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace coast {

sim_instance::sim_instance(const std::string&      config_path,
                           hdf5::executor&         h5_exec,
                           const paths::app_config&  app_cfg,
                           seed_offset_t           seed_offset)
    : config_path(config_path)
{
    // Load configuration
    json_loader = std::make_unique<loader::json_loader>(config_path);

    // Set up simulation environment
    sim_env = std::make_unique<sim::env>();
    sim_env->set_run_id(resolve_run_id(json_loader->get_config().run_id));

    // Create simulation builder and register entity handlers
    sim_builder = std::make_unique<sim::sim_builder>(*sim_env);
    sim::register_entity_handlers(*sim_builder, *json_loader, h5_exec, seed_offset);

    // Build the simulation from config
    sim_builder->build_local(json_loader->get_sim_cfg());

    // Get reference to ngm2 neuron groups (optional — sims may use ngm3 only).
    if (auto groups_opt = sim_env->get_entities<ngm2::neuron_group_t>();
        groups_opt && !groups_opt.value()->empty())
    {
        groups = groups_opt.value();
    }

    // Require at least one neuron group so we know the sim is non-empty.
    // build_local built an entity for every node (or threw), and a node_set only
    // copies declared nodes, so the declared nodes decide it.
    if (std::ranges::none_of(json_loader->get_sim_cfg().nodes, [](const auto& n) {
            return sim::is_neuron_group_entity_type(n.entity_type); }))
        throw std::runtime_error("No neuron groups found in simulation environment");

    // Attach observe registry: install pre/post-process hooks and seed groups
    // from the parsed observe config. Site namespacing is resolved from each
    // observed owner at capture time, so multi-root simulations do not depend
    // on any single group's structure_registry here.
    // Resolve the OBSERVE output directory from the ALREADY-PARSED config: the
    // config's observe.output_dir leaf (empty -> app default paths.observe_dir)
    // placed under config_base (an absolute leaf wins). Reading the leaf from the
    // parsed struct — not by re-peeking the file — keeps the field name in one
    // place (observe_cfg_t) and avoids a second parse.
    const observe::observe_cfg_t& obs_cfg = json_loader->get_config().observe;
    const std::string obs_leaf =
        paths::effective_dir(obs_cfg.output_dir, app_cfg.paths.observe_dir);
    const std::filesystem::path observe_output_dir =
        coast::paths::resolve_config_output(config_path, obs_leaf,
                                            app_cfg.paths.config_base);

    observe::observe_registry::instance().attach(
        *sim_env,
        obs_cfg,
        observe_output_dir,
        h5_exec);

    // Create simulation thread controller (starts in paused mode)
    thread_controller = std::make_unique<sim::sim_thread_controller>(*sim_env);

    // Create logging manager
    logging_manager = std::make_unique<logging::logging_manager>(*sim_env, h5_exec);
}

sim_instance::~sim_instance()
{
    // Join the sim thread FIRST (issue #197): env hook removal is synchronous
    // since #235, but the logging manager's stop_all() and the observe
    // registry's writers must also not race a running step. Then the logging
    // manager (its stop_all() relies on this), then the observe registry, so no
    // entity code can still call OBSERVE while writers are joining.
    thread_controller.reset();
    logging_manager.reset();
    observe::observe_registry::instance().detach();
}

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
        // Same order as ~sim_instance: join the sim thread before the logging
        // manager, then detach observe, then move the resources in.
        thread_controller.reset();
        logging_manager.reset();
        observe::observe_registry::instance().detach();

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

void sim_instance::prepare_resume()
{
    if (json_loader->get_config().run_id == k_auto_run_id)
        sim_env->adopt_checkpoint_run_id();
}

} // namespace coast
