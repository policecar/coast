//
// Created by jk on 27.01.26.
//

#ifndef VIS_HINTS_H
#define VIS_HINTS_H

#include <unordered_map>
#include <string>

namespace coast {

struct ngm_flat_vis_hint_t {
    int width  = 1;
    int height = 1;
    int update_interval = 10000;  // Simulation steps between updates (0 = every frame)
};

struct ngm_tree_vis_hint_t {
    // Representation dimensions (for leaf node textures)
    int rep_width  = 28;
    int rep_height = 28;

    // Update interval
    int update_interval = 10000;

    // Background color (RGBA 0-255)
    int bg_r = 30, bg_g = 30, bg_b = 30, bg_a = 255;

    // Edge appearance
    float edge_thickness_min = 2.0f;
    float edge_thickness_max = 10.0f;
    int edge_r = 0, edge_g = 121, edge_b = 241, edge_a = 255;

    // Node texture scale (canvas_width / this value)
    float texture_scale_divisor = 1500.0f;

    // Node radii
    float root_node_radius   = 6.0f;
    float branch_node_radius = 3.0f;

    // Root node color (RGBA 0-255)
    int root_r = 255, root_g = 161, root_b = 0, root_a = 255;

    // Branch node color (RGBA 0-255)
    int branch_r = 0, branch_g = 0, branch_b = 0, branch_a = 255;
};

struct mnist_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 28, .height = 28, .update_interval = 1000};
};

struct dsprites_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 64, .height = 64, .update_interval = 1000};
};

struct buffer_vis_hint_t {
    int update_interval = 100;  // Simulation steps between updates (0 = every frame)
};

struct dataset_hints_t {
    std::unordered_map<std::string,mnist_hint_t>    mnist;
    std::unordered_map<std::string,dsprites_hint_t> dsprites;

    dataset_hints_t()
    {
        mnist["default"];
        dsprites["default"];
    }
};

struct neuron_group_hints_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 50, .height = 1, .update_interval = 10000};
    ngm_tree_vis_hint_t ngm_tree_vis = {};
};

struct ngm2_hints_t {
    std::unordered_map<std::string,neuron_group_hints_t> neuron_groups;

    ngm2_hints_t()
    {
        neuron_groups["default"];
    }
};

struct vis_hints_t {
    dataset_hints_t  datasets;
    ngm2_hints_t     ngm2;
    buffer_vis_hint_t buffer_vis;  // Default hints for io_buffer visualizations
};


}


#endif //VIS_HINTS_H
