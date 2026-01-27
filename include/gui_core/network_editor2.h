#ifndef COAST_NETWORK_EDITOR2_H
#define COAST_NETWORK_EDITOR2_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>

#include "imgui.h"
#include "json_loader.h"

namespace ImFlow {
    class ImNodeFlow;
    class BaseNode;
}

namespace coast {

/**
 * Shadow data for a node - augments config data with visual state.
 */
struct EditorNodeState {
    sim::sim_node_id_t id;
    std::string label;
    sim::io_entity_type_t entity_type;
    std::string entity_cfg;
    ImVec2 position;
    bool position_initialized = false;
};

/**
 * Shadow data for a link - represents a valid connection.
 */
struct EditorLinkState {
    sim::sim_node_id_t src_id;
    sim::sim_node_id_t dst_id;
    std::string channel_name;
};

/**
 * Graphical editor for the simulation graph (Phase 1: Read-only visualization).
 *
 * Key principle: config_editor is the single source of truth.
 * This class is a visualization layer that reads from config.
 *
 * Data flow:
 *   config_editor (authoritative)
 *       -> network_editor2 shadow data (augmented copy with positions)
 *       -> ImNodeFlow visualization
 */
class network_editor2 {
public:
    using dirty_callback_t = std::function<void()>;

    /**
     * Construct network editor with reference to config.
     */
    explicit network_editor2(loader::global_cfg_t& config);

    ~network_editor2();

    // Disable copy (owns ImNodeFlow instance)
    network_editor2(const network_editor2&) = delete;
    network_editor2& operator=(const network_editor2&) = delete;

    /**
     * Render the network editor window.
     * @param p_open Pointer to visibility flag
     */
    void render(bool* p_open);

    /**
     * Set callback for when config is modified.
     * (Reserved for Phase 2 - delegated editing)
     */
    void set_dirty_callback(dirty_callback_t callback);

private:
    loader::global_cfg_t& config;
    dirty_callback_t dirty_callback;

    // ImNodeFlow editor instance
    std::unique_ptr<ImFlow::ImNodeFlow> editor;

    // Shadow data - rebuilt from config, augmented with positions
    std::unordered_map<sim::sim_node_id_t, EditorNodeState> node_states;
    std::vector<EditorLinkState> link_states;

    // Validation errors detected during last rebuild
    std::vector<sim::sim_node_id_t> duplicate_node_ids_;

    // Rebuild nodes and links from config (called on first render and Refresh)
    void rebuild_from_config();

    // Get input channels for a node based on its entity config
    std::vector<std::string> get_input_channels(const sim::sim_node_t& node) const;

    // Check if a link is valid (src exists, dst exists, channel valid)
    bool is_link_valid(sim::sim_node_id_t src_id,
                       sim::sim_node_id_t dst_id,
                       const std::string& channel) const;

    // Find node in config by ID
    const sim::sim_node_t* find_node_by_id(sim::sim_node_id_t id) const;

    // Mark config as dirty (for Phase 2)
    void mark_dirty();
};

} // namespace coast

#endif // COAST_NETWORK_EDITOR2_H
