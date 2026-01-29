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

        // Branch node color (RGBA 0-255)
        uint8_t branch_r = 0, branch_g = 0, branch_b = 0, branch_a = 255;
    };

private:
    params_t params_;
    float region_size_ = 1.0f;  // Stored from fbgd for rendering

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

    void create_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_representations(const ngm2::neuron_group_t &ng);

    /**
     * Update root node repulsion radii based on dendritic tree sizes.
     * Larger trees get larger spheres of influence, pushing other neurons away.
     */
    void update_repulsion_radii(fbgd &vis);

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
