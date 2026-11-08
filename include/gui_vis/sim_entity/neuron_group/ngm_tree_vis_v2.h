//
// Neuron tree visualization using gui_layout architecture
// Drop-in replacement for ngm_tree_vis
//

#ifndef COAST_NGM_TREE_VIS_V2_H
#define COAST_NGM_TREE_VIS_V2_H

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/rng/fast_rng.h"
#include "hd_ngm2/hd_ngm2.h"
#include "gui_layout/layout_graph.h"
#include "gui_layout/force_layout.h"
#include "gui_layout/render/graph_canvas.h"
#include "gui_layout/render/node_renderer.h"
#include "gui_vis/core/color_types.h"
#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"

namespace coast {

/// Node type in the dendritic tree
enum class tree_node_type : uint8_t {
    root,    // Neuron root node (orange circle)
    branch,  // Internal branch node (small blue circle)
    leaf     // Leaf segment with synapse texture
};

/// User data stored in node_t::user_data via shared_ptr
struct tree_node_data {
    std::size_t neuron_idx = 0;
    std::size_t branch_idx = 0;
    uint16_t segment_idx = 0;
    tree_node_type type = tree_node_type::branch;
    uint32_t synapse_count = 0;
    gl_texture texture;
    std::vector<color_t> pixel_data;
};

/// Rendering parameters for the tree visualization
struct ngm_tree_vis_v2_params_t {
    // Representation dimensions (for leaf node textures)
    int rep_width  = 28;
    int rep_height = 28;
    int num_channels = 1;  // 1 = grayscale (use colormap), 3 = RGB

    // Background color (RGBA 0-255)
    uint8_t bg_r = 30, bg_g = 30, bg_b = 30, bg_a = 255;

    // Edge appearance
    float edge_thickness_min = 2.0f;
    float edge_thickness_max = 10.0f;
    uint8_t edge_r = 0, edge_g = 121, edge_b = 241, edge_a = 255;

    // Node texture scale (canvas_width / this value)
    float texture_scale_divisor = 1500.0f;

    // Node radii
    float root_node_radius   = 6.0f;
    float branch_node_radius = 3.0f;

    // Root node color (RGBA 0-255)
    uint8_t root_r = 255, root_g = 161, root_b = 0, root_a = 255;

    // Branch node color (RGBA 0-255) - light blue
    uint8_t branch_r = 100, branch_g = 180, branch_b = 255, branch_a = 255;
};

/// Custom node renderer for dendritic tree visualization
class tree_node_renderer : public layout::render::node_renderer {
public:
    explicit tree_node_renderer(const ngm_tree_vis_v2_params_t& params);

    void render(ImDrawList* draw_list,
                const layout::node_t& node,
                ImVec2 screen_pos,
                float zoom,
                bool selected,
                bool hovered) override;

    ImVec2 get_size(const layout::node_t& node, float zoom) override;

    /// Set which neurons' textures to display
    void set_display_neurons(const std::unordered_set<std::size_t>& indices) { display_neurons_ = indices; }
    void add_display_neuron(std::size_t idx) { display_neurons_.insert(idx); }
    void clear_display_neurons() { display_neurons_.clear(); }
    [[nodiscard]] bool is_display_neuron(std::size_t idx) const { return display_neurons_.count(idx) > 0; }

    /// Set parameters reference
    void set_params(const ngm_tree_vis_v2_params_t& params) { params_ = &params; }

    /// Set canvas size for texture scaling
    void set_canvas_size(ImVec2 size) { canvas_size_ = size; }

private:
    const ngm_tree_vis_v2_params_t* params_ = nullptr;
    std::unordered_set<std::size_t> display_neurons_;
    ImVec2 canvas_size_{600, 600};
};

/// Main tree visualization class using gui_layout components
class ngm_tree_vis_v2 {
public:
    ngm_tree_vis_v2();
    explicit ngm_tree_vis_v2(const ngm_tree_vis_v2_params_t& params);
    ~ngm_tree_vis_v2();

    void set_params(const ngm_tree_vis_v2_params_t& params) { params_ = params; }
    [[nodiscard]] const ngm_tree_vis_v2_params_t& get_params() const { return params_; }

    // Colormap dirty flag - set when colormap changes, triggers re-render
    [[nodiscard]] bool is_colormap_dirty() const { return colormap_dirty_; }
    void clear_colormap_dirty() { colormap_dirty_ = false; }

    // Spacing dirty flag - set when spacing slider changes
    [[nodiscard]] bool is_spacing_dirty() const { return spacing_dirty_; }
    void clear_spacing_dirty() { spacing_dirty_ = false; }

    /// Build graph from neuron group hierarchy
    void create_model(const ngm2::neuron_group_t& ng);

    /// Incremental update as dendrites grow
    void update_model(const ngm2::neuron_group_t& ng);

    /// Regenerate textures with current colormap
    void update_representations(const ngm2::neuron_group_t& ng);

    /// Update root node target positions based on tree sizes
    void update_root_positions(const ngm2::neuron_group_t& ng);

    /// Calculate bounding box radius for each neuron's tree and update spacing
    void update_tree_spacing();

    /// Release GPU resources
    void free_resources();

    /// Render in ImGui window. Returns false if window was closed.
    bool render_imgui(const char* label);

private:
    ngm_tree_vis_v2_params_t params_;
    float region_size_ = 500.0f;  // Layout region size (larger for graph_canvas coordinates)

    // Colormap for representations
    colormap_t colormap_ = colormap_t::viridis;
    bool colormap_dirty_ = false;

    // Spacing control for root nodes
    float spacing_multiplier_ = 1.0f;
    bool spacing_dirty_ = false;

    // Selection state (supports multi-select)
    std::unordered_set<std::size_t> selected_neurons_;
    bool show_on_hover_ = false;
    std::size_t hovered_neuron_idx_ = SIZE_MAX;

    // Target positions for each neuron's tree (for smooth animation)
    std::vector<std::array<float, 2>> neuron_target_positions_;

    // gui_layout components
    layout::layout_graph graph_;
    std::unique_ptr<layout::force_layout> layout_;
    layout::render::graph_canvas canvas_;
    tree_node_renderer renderer_;

    // Lookup tables
    struct node_info_t {
        layout::node_id_t node_id = layout::invalid_node_id;
        std::shared_ptr<tree_node_data> data;
    };
    std::vector<node_info_t> nodes_;
    std::vector<std::size_t> neuron_lu_;      // neuron_idx -> first node index
    std::vector<std::size_t> branch_start_lu_; // neuron_idx -> first branch index in branch_lu
    std::vector<std::size_t> branch_lu_;       // flat list of segment bases per branch

    // RNG for initial positioning
    fast_rng_t rgen_{0};
    std::uniform_real_distribution<float> rdis_{-1.0f, 1.0f};

    // Temp buffer for tree traversal
    std::vector<uint8_t> tmp_tree_;

    // Max synapse count for edge thickness normalization
    uint32_t max_synapse_count_ = 1;

    // Flag to fit view on first render
    bool needs_fit_view_ = false;

    // Force layout parameters
    float repulsion_ = 170.0f;
    float attraction_ = 0.05f;

    // Animated settling state
    bool settling_active_ = false;
    int settling_iterations_remaining_ = 0;
    bool initial_settling_ = false;  // True only for first settling after create_model

    // Track initial growth - reset view once after graph stabilizes
    bool initial_view_reset_done_ = false;
    int total_frames_rendered_ = 0;

    // Helper methods
    void get_pixel_data(int width, int height, uint16_t leaf_idx, std::size_t max_idx,
                        const ngm2::dendrite_t::synapses_t& synapses,
                        std::vector<color_t>& output);

    void setup_edge_renderer();
    void animate_tree_positions(float lerp_factor = 0.05f);
    tree_node_data* get_node_data(layout::node_id_t id);

    /// Apply per-neuron repulsion (nodes only repel within their own neuron's tree)
    void apply_per_neuron_repulsion();

    /// Run one step of custom settling with per-neuron repulsion
    void settle_step();

    /// Run custom settling with per-neuron repulsion (blocking)
    void settle_with_per_neuron_forces(int iterations);

    /// Start animated settling (non-blocking, runs in render loop)
    void start_animated_settling(int iterations);
};

} // namespace coast

#endif // COAST_NGM_TREE_VIS_V2_H
