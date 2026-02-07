//
// Node renderer for buffer visualization in replay dialog
//

#ifndef COAST_REPLAY_NODE_RENDERER_H
#define COAST_REPLAY_NODE_RENDERER_H

#include "gui_layout/render/node_renderer.h"
#include "gui_vis/core/gl_texture.h"
#include "gui_vis/core/colormap.h"
#include "buffer_log_reader.h"

#include <string>
#include <vector>

namespace coast::replay {

/// Visualization mode for buffer nodes
enum class vis_mode {
    image_2d,        // Reshape to 2D grid, apply colormap
    stats_color,     // Color by aggregate stat (avg, sum, etc.)
    delta_flow       // Show change from previous frame
};

/// Visualization configuration for a buffer node
struct vis_config {
    vis_mode mode = vis_mode::image_2d;
    int width = 28;
    int height = 28;
    int num_channels = 1;  // 1 = grayscale (use colormap), 3 = RGB (no colormap)
    colormap_t colormap = colormap_t::viridis;
    bool auto_normalize = true;
    float manual_min = 0.0f;
    float manual_max = 1.0f;
};

/// Statistics about frame data
enum class stat_type {
    average,
    sum,
    min,
    max,
    nse
};

/// User data stored in node_t::user_data
struct buffer_node_data {
    std::size_t buffer_id = 0;
    std::string buffer_name;
    std::size_t buffer_size = 0;
    vis_config config;
    gl_texture texture;
    std::vector<float> current_data;
    std::vector<float> previous_data;  // For delta mode
    frame_stats current_stats;
    bool data_valid = false;
};

/// Node renderer for buffer replay visualization
class replay_node_renderer : public layout::render::node_renderer {
public:
    replay_node_renderer();
    ~replay_node_renderer() override;

    void render(ImDrawList* draw_list,
                const layout::node_t& node,
                ImVec2 screen_pos,
                float zoom,
                bool selected,
                bool hovered) override;

    ImVec2 get_size(const layout::node_t& node, float zoom) override;

    /// Update buffer data for a node
    void update_node_data(layout::node_t& node,
                          std::span<const float> data,
                          const frame_stats& stats);

    /// Clear data (e.g., when switching frames)
    void clear_node_data(layout::node_t& node);

    /// Set global visualization config
    void set_global_config(const vis_config& config);

    /// Get global config
    [[nodiscard]] const vis_config& global_config() const { return global_config_; }

    /// Apply global config (colormap, normalize, mode) to all nodes in the graph
    void apply_global_config_to_nodes(layout::layout_graph& graph);

    /// Set stat type for stats_color mode
    void set_stat_type(stat_type type) { stat_type_ = type; }

    /// Show node labels
    void set_show_labels(bool show) { show_labels_ = show; }

    /// Set node base size (before zoom)
    void set_node_size(float size) { node_size_ = size; }

private:
    vis_config global_config_;
    stat_type stat_type_ = stat_type::average;
    bool show_labels_ = true;
    float node_size_ = 60.0f;

    // Temporary buffer for pixel data
    std::vector<color_t> pixel_buffer_;

    void update_texture(buffer_node_data& data);
    void render_image_2d(ImDrawList* draw_list, const buffer_node_data& data,
                         ImVec2 screen_pos, float zoom, bool selected, bool hovered);
    void render_stats_color(ImDrawList* draw_list, const buffer_node_data& data,
                            ImVec2 screen_pos, float zoom, bool selected, bool hovered);
    void render_delta_flow(ImDrawList* draw_list, const buffer_node_data& data,
                           ImVec2 screen_pos, float zoom, bool selected, bool hovered);
};

} // namespace coast::replay

#endif // COAST_REPLAY_NODE_RENDERER_H
