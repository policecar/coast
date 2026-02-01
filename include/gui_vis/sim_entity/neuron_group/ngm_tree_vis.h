//
// Created by jk on 12.08.25.
//

#ifndef NGM_TREE_VIS_H
#define NGM_TREE_VIS_H

#include <vector>
#include <random>
#include <cstdint>
#include "hd_ngm2.h"
#include "../../layout/fbgd.h"
#include "../../core/color_types.h"
#include "../../core/colormap.h"
#include "../../core/gl_texture.h"

namespace coast {

class ngm_tree_vis {
public:
    /**
     * Rendering parameters for the tree visualization.
     */
    struct params_t {
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

private:
    params_t params_;
    float region_size_ = 1.0f;  // Stored from fbgd for rendering

    // View state (zoom and pan) - uses target values for smooth interpolation
    float zoom_level_ = 1.0f;
    float target_zoom_ = 1.0f;
    float pan_x_ = 0.0f;
    float pan_y_ = 0.0f;
    float target_pan_x_ = 0.0f;
    float target_pan_y_ = 0.0f;

    // Colormap for representations
    colormap_t colormap_ = colormap_t::grayscale;
    bool colormap_dirty_ = false;

    // Spacing control for root nodes
    float spacing_multiplier_ = 1.0f;
    bool spacing_dirty_ = false;

    // Target positions for each neuron's tree (for smooth animation)
    std::vector<std::array<float, 2>> neuron_target_positions_;

    // Selection state - which neuron's representations are visible
    // SIZE_MAX means no neuron selected (all representations hidden)
    std::size_t selected_neuron_idx_ = SIZE_MAX;
    bool show_on_hover_ = false;  // Show representations when hovering over root node
    std::size_t hovered_neuron_idx_ = SIZE_MAX;  // Currently hovered neuron

    struct node_info_t {
        std::size_t neuron_idx  = 0;
        std::size_t branch_idx  = 0;
        uint16_t    segment_idx = 0;

        fbgd::node_id_t node_id = 0;
        uint8_t used = 0;

        gl_texture texture;
        std::vector<color_t> pixel_data;

        uint32_t synapse_count = 0;
    };

    std::mt19937 rgen {0};
    std::uniform_real_distribution<float> rdis { -1.0, 1.0 };

    std::vector<node_info_t> nodes;
    std::vector<std::size_t> neuron_lu;
    std::vector<std::size_t> branch_start_lu;
    std::vector<std::size_t> branch_lu;

    [[maybe_unused]] fbgd::node_id_t group_node = 0;

    std::vector<uint8_t> tmp_tree;

    void get_pixel_data(int width, int height, uint16_t leaf_idx, std::size_t max_idx, const ngm2::dendrite_t::synapses_t &synapses, std::vector<color_t> &output);
    static gl_texture get_texture(int width, int height, const color_t *data);

public:
    ngm_tree_vis() = default;
    explicit ngm_tree_vis(const params_t& params) : params_(params) {}
    ~ngm_tree_vis();

    void set_params(const params_t& params) { params_ = params; }
    [[nodiscard]] const params_t& get_params() const { return params_; }

    // Colormap dirty flag - set when colormap changes, triggers re-render
    [[nodiscard]] bool is_colormap_dirty() const { return colormap_dirty_; }
    void clear_colormap_dirty() { colormap_dirty_ = false; }

    // Spacing dirty flag - set when spacing slider changes
    [[nodiscard]] bool is_spacing_dirty() const { return spacing_dirty_; }
    void clear_spacing_dirty() { spacing_dirty_ = false; }

    void create_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_representations(const ngm2::neuron_group_t &ng);

    /**
     * Update root node repulsion radii based on dendritic tree sizes.
     * Larger trees get larger spheres of influence, pushing other neurons away.
     */
    void update_repulsion_radii(fbgd &vis);

    /**
     * Update root node target positions based on tree sizes.
     * Neurons with larger trees get more space in the layout.
     */
    void update_root_positions(fbgd &vis, const ngm2::neuron_group_t &ng);

    /**
     * Animate all nodes towards their target positions.
     * Call this every frame for smooth tree repositioning.
     */
    void animate_tree_positions(fbgd &vis, float lerp_factor = 0.05f);

    void free_resources();

    // Render in ImGui window using ImDrawList. Returns false if window was closed.
    bool render_imgui(
        const char* label,
        const std::vector<fbgd::node>& vis_nodes,
        const std::vector<fbgd::edge>& vis_edges,
        const std::vector<uint32_t>&   vis_node_lu
    );

};

} // coast

#endif //NGM_TREE_VIS_H
