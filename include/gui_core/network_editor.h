#ifndef COAST_NETWORK_EDITOR_H
#define COAST_NETWORK_EDITOR_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>

#include "imgui.h"
#include "json_cfg/json_loader.h"
#include "gui_layout/layout_graph.h"
#include "gui_layout/force_layout.h"

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
 * Callbacks for delegating edit operations to config_editor.
 * All callbacks return true on success, false if the operation was rejected.
 */
struct edit_callbacks_t {
    // Add a link from src node's output to dst node's input channel
    std::function<bool(sim::sim_node_id_t src, sim::sim_node_id_t dst, const std::string& channel)> add_link;

    // Remove a link
    std::function<bool(sim::sim_node_id_t src, sim::sim_node_id_t dst, const std::string& channel)> remove_link;

    // Remove a node (and all its connections)
    std::function<bool(sim::sim_node_id_t id)> remove_node;

    // Add a new node, returns the new node's ID (0 on failure)
    std::function<sim::sim_node_id_t(sim::io_entity_type_t type, const std::string& label)> add_node;
};

/**
 * Graphical editor for the simulation graph.
 *
 * Key principle: config_editor is the single source of truth.
 * This class is a visualization layer that delegates all edits to config_editor.
 *
 * Data flow:
 *   config_editor (authoritative)
 *       -> network_editor shadow data (augmented copy with positions)
 *       -> ImNodeFlow visualization
 *       -> user action
 *       -> delegate to config_editor -> config_editor validates/applies -> manual refresh
 */
class network_editor {
public:
    using dirty_callback_t = std::function<void()>;

    /**
     * Construct network editor with reference to config.
     */
    explicit network_editor(loader::global_cfg_t& config);

    ~network_editor();

    // Disable copy (owns ImNodeFlow instance)
    network_editor(const network_editor&) = delete;
    network_editor& operator=(const network_editor&) = delete;

    /**
     * Render the network editor window.
     * @param p_open Pointer to visibility flag
     */
    void render(bool* p_open);

    /**
     * Set callback for when config is modified.
     */
    void set_dirty_callback(dirty_callback_t callback);

    /**
     * Set callbacks for delegating edit operations to config_editor.
     */
    void set_edit_callbacks(edit_callbacks_t callbacks);

    /**
     * Mark the editor as needing a refresh from config.
     * Call this when config is modified externally.
     */
    void invalidate() { needs_refresh_ = true; }

private:
    bool needs_refresh_ = false;
    loader::global_cfg_t& config;
    dirty_callback_t dirty_callback;
    edit_callbacks_t edit_callbacks_;

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

    // Mark config as dirty
    void mark_dirty();

    // Handle Delete key for selected nodes/links, returns true if anything was deleted
    bool handle_delete_key();

    // Detect and handle newly created links, returns true if any were added
    bool handle_new_links();

    // Auto-layout state
    bool auto_layout_active_ = false;
    std::unique_ptr<layout::layout_graph> layout_graph_;
    std::unique_ptr<layout::force_layout> force_layout_;
    std::unordered_map<sim::sim_node_id_t, layout::node_id_t> sim_to_layout_id_;

    // Auto-layout helper methods
    void init_auto_layout();
    void step_auto_layout();
    void stop_auto_layout();

    // View reset helper
    void reset_view();
};

} // namespace coast

#endif // COAST_NETWORK_EDITOR_H
