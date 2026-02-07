//
// ImGui canvas widget for rendering graphs with zoom/pan/interaction
//

#ifndef COAST_GRAPH_CANVAS_H
#define COAST_GRAPH_CANVAS_H

#include "gui_layout/layout_graph.h"
#include "node_renderer.h"

#include "imgui.h"

#include <optional>
#include <functional>

namespace coast::layout::render {

/// Edge style configuration
struct edge_style {
    ImU32 color = IM_COL32(100, 100, 100, 180);
    float thickness = 1.5f;
    bool show_direction = false;  // Draw arrows on directed edges
    float arrow_size = 8.0f;
};

/// Callback for custom edge rendering
using edge_render_callback = std::function<void(
    ImDrawList* draw_list,
    const edge_t& edge,
    ImVec2 from_screen,
    ImVec2 to_screen,
    float zoom
)>;

/// ImGui canvas for rendering and interacting with force-directed graphs
class graph_canvas {
public:
    graph_canvas();

    /// Set the graph to render
    void set_graph(layout_graph* graph);

    /// Set the node renderer
    void set_node_renderer(node_renderer* renderer);

    /// Main render function - returns true if graph was modified
    /// @param label ImGui label for the widget
    /// @param size Widget size (0 = auto-fill)
    bool render(const char* label, ImVec2 size = ImVec2(0, 0));

    // View control

    /// Center view on the graph
    void center_on_graph();

    /// Fit the entire graph in view
    void fit_graph_in_view();

    /// Set zoom level (1.0 = 100%)
    void set_zoom(float zoom);

    /// Get current zoom level
    [[nodiscard]] float zoom() const { return zoom_; }

    /// Set pan offset
    void set_pan(ImVec2 pan);

    /// Get current pan offset
    [[nodiscard]] ImVec2 pan() const { return pan_; }

    // Selection

    /// Get currently selected node (if any)
    [[nodiscard]] std::optional<node_id_t> selected_node() const { return selected_; }

    /// Get currently hovered node (if any)
    [[nodiscard]] std::optional<node_id_t> hovered_node() const { return hovered_; }

    /// Clear selection
    void clear_selection();

    /// Select a specific node
    void select_node(node_id_t id);

    // Edge styling

    /// Set edge style
    void set_edge_style(const edge_style& style);

    /// Get edge style
    [[nodiscard]] const edge_style& get_edge_style() const { return edge_style_; }

    /// Set custom edge renderer
    void set_edge_renderer(edge_render_callback callback);

    // Configuration

    /// Enable/disable node dragging
    void set_dragging_enabled(bool enabled) { dragging_enabled_ = enabled; }

    /// Enable/disable zoom
    void set_zoom_enabled(bool enabled) { zoom_enabled_ = enabled; }

    /// Enable/disable pan
    void set_pan_enabled(bool enabled) { pan_enabled_ = enabled; }

    /// Set minimum zoom level
    void set_min_zoom(float min_zoom) { min_zoom_ = min_zoom; }

    /// Set maximum zoom level
    void set_max_zoom(float max_zoom) { max_zoom_ = max_zoom; }

    /// Set zoom speed (mouse wheel sensitivity)
    void set_zoom_speed(float speed) { zoom_speed_ = speed; }

    /// Enable smooth zoom/pan animations
    void set_smooth_animation(bool enabled) { smooth_animation_ = enabled; }

    // Coordinate conversion

    /// Convert world coordinates to screen coordinates
    [[nodiscard]] ImVec2 world_to_screen(float x, float y) const;

    /// Convert screen coordinates to world coordinates
    [[nodiscard]] ImVec2 screen_to_world(ImVec2 screen) const;

private:
    layout_graph* graph_ = nullptr;
    node_renderer* renderer_ = nullptr;

    // View state
    ImVec2 pan_{0, 0};
    float zoom_ = 1.0f;
    ImVec2 target_pan_{0, 0};
    float target_zoom_ = 1.0f;
    ImVec2 canvas_pos_{0, 0};
    ImVec2 canvas_size_{0, 0};

    // Interaction state
    std::optional<node_id_t> selected_;
    std::optional<node_id_t> hovered_;
    std::optional<node_id_t> dragging_;
    ImVec2 drag_start_pos_{0, 0};
    bool panning_ = false;
    ImVec2 pan_start_mouse_{0, 0};
    ImVec2 pan_start_offset_{0, 0};

    // Edge rendering
    edge_style edge_style_;
    edge_render_callback custom_edge_renderer_;

    // Configuration
    bool dragging_enabled_ = true;
    bool zoom_enabled_ = true;
    bool pan_enabled_ = true;
    float min_zoom_ = 0.1f;
    float max_zoom_ = 10.0f;
    float zoom_speed_ = 0.1f;
    bool smooth_animation_ = true;

    // Internal methods
    void handle_interaction(bool hovered);
    void update_animation();
    void render_edges(ImDrawList* draw_list);
    void render_nodes(ImDrawList* draw_list);
    std::optional<node_id_t> hit_test(ImVec2 screen_pos);
};

} // namespace coast::layout::render

#endif // COAST_GRAPH_CANVAS_H
