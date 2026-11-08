//
// Created by claude on 29.01.26.
//

#ifndef VIS_REGISTRY_H
#define VIS_REGISTRY_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tools/sim_builder_cfg.h"
#include "core/vis_hints.h"

namespace sim {
struct io_entity;
class io_buffer;
class env;
}

namespace coast::vis {

enum class vis_target : uint8_t {
    sim_entity,    // Visualizes simulation entities
    io_buffer,     // Visualizes IO buffers
};

struct vis_info {
    std::string id;                                           // e.g., "ngm_flat_vis"
    std::string display_name;                                 // e.g., "Neuron Grid"
    std::string description;                                  // Tooltip text
    vis_target target;
    std::vector<sim::io_entity_type_t> supported_entity_types;
};

/**
 * Abstract base class for visualization instances.
 * Visualizations implement this interface to integrate with the registry system.
 */
class vis_instance {
public:
    virtual ~vis_instance() = default;

    /**
     * Update visualization data.
     * Called from simulation hooks when new data is available.
     */
    virtual void update() = 0;

    /**
     * Render the ImGui window.
     * @return false if the window was closed by the user
     */
    virtual bool render_imgui() = 0;

    /**
     * Release GPU/OpenGL resources.
     */
    virtual void free_resources() = 0;

    /**
     * Check if visualization is currently visible.
     */
    [[nodiscard]] virtual bool is_visible() const = 0;

    /**
     * Set visibility state.
     */
    virtual void set_visible(bool v) = 0;
};

/**
 * Factory function type for creating entity visualizations.
 *
 * @param entity The simulation entity to visualize
 * @param sim_env The simulation environment (for hook registration)
 * @param hints Visualization hints from configuration
 * @param hints_key Key to look up entity-specific hints (typically entity_cfg)
 * @param label Display label for the visualization window
 * @return Unique pointer to created visualization instance
 */
using entity_vis_factory = std::function<
    std::unique_ptr<vis_instance>(
        sim::io_entity& entity,
        sim::env& sim_env,
        const coast::vis_hints_t& hints,
        const std::string& hints_key,
        const std::string& label
    )
>;

/**
 * Factory function type for creating IO buffer visualizations.
 *
 * @param buffer The IO buffer to visualize
 * @param sim_env The simulation environment (for hook registration)
 * @param hints Visualization hints from configuration
 * @param label Display label for the visualization window
 * @return Unique pointer to created visualization instance
 */
using buffer_vis_factory = std::function<
    std::unique_ptr<vis_instance>(
        sim::io_buffer& buffer,
        sim::env& sim_env,
        const coast::vis_hints_t& hints,
        const std::string& label
    )
>;

/**
 * Central registry for visualization types.
 *
 * Visualizations register themselves at startup using static initializers.
 * The GUI queries the registry to discover available visualizations
 * for specific entity types.
 */
class vis_registry {
public:
    /**
     * Register an entity visualization.
     *
     * @param info Metadata about the visualization
     * @param factory Function to create visualization instances
     */
    void register_entity_vis(const vis_info& info, entity_vis_factory factory);

    /**
     * Get all visualizations available for a specific entity type.
     *
     * @param type The entity type to query
     * @return Vector of pointers to vis_info for matching visualizations
     */
    [[nodiscard]] std::vector<const vis_info*>
    get_entity_visualizations(sim::io_entity_type_t type) const;

    /**
     * Create a visualization instance.
     *
     * @param vis_id The visualization ID (from vis_info::id)
     * @param entity The entity to visualize
     * @param sim_env The simulation environment (for hook registration)
     * @param hints Visualization hints from configuration
     * @param hints_key Key to look up entity-specific hints (typically entity_cfg)
     * @param label Display label for the window
     * @return Created visualization instance, or nullptr if vis_id not found
     */
    [[nodiscard]] std::unique_ptr<vis_instance>
    create_entity_vis(const std::string& vis_id,
                      sim::io_entity& entity,
                      sim::env& sim_env,
                      const coast::vis_hints_t& hints,
                      const std::string& hints_key,
                      const std::string& label);

    // ========================================================================
    // IO Buffer Visualizations
    // ========================================================================

    /**
     * Register an IO buffer visualization.
     *
     * @param info Metadata about the visualization
     * @param factory Function to create visualization instances
     */
    void register_buffer_vis(const vis_info& info, buffer_vis_factory factory);

    /**
     * Get all visualizations available for IO buffers.
     *
     * @return Vector of pointers to vis_info for buffer visualizations
     */
    [[nodiscard]] std::vector<const vis_info*>
    get_buffer_visualizations() const;

    /**
     * Create a buffer visualization instance.
     *
     * @param vis_id The visualization ID (from vis_info::id)
     * @param buffer The IO buffer to visualize
     * @param sim_env The simulation environment (for hook registration)
     * @param hints Visualization hints from configuration
     * @param label Display label for the window
     * @return Created visualization instance, or nullptr if vis_id not found
     */
    [[nodiscard]] std::unique_ptr<vis_instance>
    create_buffer_vis(const std::string& vis_id,
                      sim::io_buffer& buffer,
                      sim::env& sim_env,
                      const coast::vis_hints_t& hints,
                      const std::string& label);

private:
    struct registered_entity_vis {
        vis_info info;
        entity_vis_factory factory;
    };

    struct registered_buffer_vis {
        vis_info info;
        buffer_vis_factory factory;
    };

    std::vector<registered_entity_vis> entity_visualizations_;
    std::vector<registered_buffer_vis> buffer_visualizations_;
};

/**
 * Get the global visualization registry singleton.
 */
vis_registry& get_vis_registry();

} // namespace coast::vis

#endif //VIS_REGISTRY_H
