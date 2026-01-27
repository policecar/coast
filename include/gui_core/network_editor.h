#ifndef COAST_NETWORK_EDITOR_H
#define COAST_NETWORK_EDITOR_H

#include <string>
#include <vector>
#include <functional>

#include "imgui.h"
#include "json_loader.h"

namespace coast {

/**
 * Graphical editor for the simulation graph nodes.
 * Provides a node-based view for editing sim_node_t entries
 * with visual link connections.
 */
class network_editor {
public:
    using dirty_callback_t = std::function<void()>;

    /**
     * Construct network editor with reference to config.
     */
    explicit network_editor(loader::global_cfg_t& config);

    /**
     * Render the network editor window.
     * @param p_open Pointer to visibility flag
     */
    void render(bool* p_open);

    /**
     * Set callback for when config is modified.
     */
    void set_dirty_callback(dirty_callback_t callback);

private:
    loader::global_cfg_t& config;
    dirty_callback_t dirty_callback;

    // Get input channels for a node based on its entity config
    std::vector<std::string> get_input_channels(const sim::sim_node_t& node) const;

    // Find node index by its ID (-1 if not found)
    int find_node_index_by_id(sim::sim_node_id_t id) const;

    // Find channel index in input_channels list (-1 if not found)
    int find_channel_index(const std::vector<std::string>& channels, const std::string& name) const;

    // Generate next available node ID
    sim::sim_node_id_t generate_next_id() const;

    // Mark config as dirty
    void mark_dirty();

    // Delete selected nodes/links
    void delete_selected_nodes();
    void delete_selected_links();

    // Handle context menu for adding/removing nodes
    void render_context_menu();

    // Auto-arrange nodes using layered/hierarchical layout
    void auto_layout();

    // State for context menu
    bool show_context_menu = false;
    ImVec2 context_menu_pos;
    int context_menu_node = -1;  // -1 for background, >= 0 for specific node

    // Track if we need to run auto-layout (on first open or after reset)
    bool needs_layout = true;
};

} // namespace coast

#endif // COAST_NETWORK_EDITOR_H
